import sys
import os
import csv
import aiohttp
import asyncio
import nest_asyncio
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from aiohttp import web

# บังคับให้ Python พ่น Log ออกหน้าจอ Render ทันทีแบบ Real-time
sys.stdout.reconfigure(line_buffering=True)
nest_asyncio.apply()

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG & CREDENTIALS
# ═════════════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN = "8907507767:AAEv4LMGq7vB9Zm76jaLbw3Iy5T66wJxtKw"
TELEGRAM_CHAT_ID   = "5861943388"
LOG_FILE_PATH      = "storm_trade_logs.csv"

YF_SYMBOLS_MAP = {
    'XAU/USD (Gold)': 'GC=F',
    'XAG/USD (Silver)': 'SI=F',
    'BTC/USD': 'BTC-USD',
    'ETH/USD': 'ETH-USD',
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X',
    'USD/CHF': 'USDCHF=X',
    'USD/CAD': 'USDCAD=X',
    'AUD/USD': 'AUDUSD=X',
    'NZD/USD': 'NZDUSD=X',
    'EUR/JPY': 'EURJPY=X',
    'GBP/JPY': 'GBPJPY=X',
    'AUD/JPY': 'AUDJPY=X',
    'CAD/JPY': 'CADJPY=X',
    'CHF/JPY': 'CHFJPY=X',
    'NZD/JPY': 'NZDJPY=X',
    'EUR/GBP': 'EURGBP=X',
    'EUR/AUD': 'EURAUD=X',
    'EUR/CAD': 'EURCAD=X',
    'EUR/CHF': 'EURCHF=X',
    'EUR/NZD': 'EURNZD=X',
    'GBP/AUD': 'GBPAUD=X',
    'GBP/CAD': 'GBPCAD=X',
    'GBP/CHF': 'GBPCHF=X',
    'GBP/NZD': 'GBPNZD=X',
    'AUD/CAD': 'AUDCAD=X',
    'AUD/CHF': 'AUDCHF=X',
    'AUD/NZD': 'AUDNZD=X',
}

TF_CONFIGS = [
    ('5m',  '15m', '30d',  '30d'),
    ('15m', '1h',  '60d',  '1y'),
    ('1h',  '1d',  '730d', '2y')
]

WINRATE_THRESHOLD = 55.0

alerted_tracker = {}
active_trades   = []
live_logs_stream = []
qualified_list  = []
system_status = "กำลังเริ่มขั้นตอน Backtest..."

# ═════════════════════════════════════════════════════════════════════════════
#  HELPER INDICATORS
# ═════════════════════════════════════════════════════════════════════════════
def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-5)
    return 100 - (100 / (1 + rs))

def calc_atr(df, length=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def calc_vwap(df):
    vol = np.where((df['volume'].isna()) | (df['volume'] <= 0), 1.0, df['volume'])
    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    return (hlc3 * vol).cumsum() / vol.cumsum()

# ═════════════════════════════════════════════════════════════════════════════
#  CSV LOGGER & EVENT LOGGING
# ═════════════════════════════════════════════════════════════════════════════
CSV_FIELDS = ['trade_id', 'timestamp', 'pair', 'tf', 'side', 'setup', 'entry', 'sl', 'tp1', 'tp2', 'status', 'pnl_r', 'close_time']

def init_csv():
    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

def log_trade_to_csv(trade_dict):
    init_csv()
    with open(LOG_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(trade_dict)

def update_csv_trade_status(trade_id, status, pnl_r, close_time):
    if not os.path.exists(LOG_FILE_PATH):
        return
    df = pd.read_csv(LOG_FILE_PATH)
    mask = df['trade_id'] == trade_id
    if mask.any():
        df.loc[mask, 'status'] = status
        df.loc[mask, 'pnl_r'] = pnl_r
        df.loc[mask, 'close_time'] = close_time
        df.to_csv(LOG_FILE_PATH, index=False)

init_csv()

def log_event(text):
    now = datetime.now().strftime('%H:%M:%S')
    entry = f"[{now}] {text}"
    print(entry, flush=True)
    live_logs_stream.append(entry)
    if len(live_logs_stream) > 100:
        live_logs_stream.pop(0)

# ═════════════════════════════════════════════════════════════════════════════
#  TELEGRAM SENDER
# ═════════════════════════════════════════════════════════════════════════════
async def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(url, json=payload)
        except Exception as e:
            log_event(f"Telegram error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════
def run_backtest_simulation(df_base, df_htf, stop_atr=1.3, r1=0.8, r2=2.0, max_bars=60):
    if len(df_base) < 100 or len(df_htf) < 50:
        return {'total': 0, 'wr1': 0.0, 'wr2': 0.0, 'win1': 0, 'win2': 0, 'loss': 0}

    df_htf['ema200'] = calc_ema(df_htf['close'], min(200, len(df_htf)-1))
    df_htf['htf_close'] = df_htf['close'].shift(1)
    df_htf['htf_ema']   = df_htf['ema200'].shift(1)
    
    df = pd.merge_asof(
        df_base.sort_index(),
        df_htf[['htf_close', 'htf_ema']].sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )
    
    df['htf_bull'] = df['htf_close'] > df['htf_ema']
    df['htf_bear'] = df['htf_close'] < df['htf_ema']

    df['ema9']  = calc_ema(df['close'], 9)
    df['ema21'] = calc_ema(df['close'], 21)
    df['ema50'] = calc_ema(df['close'], 50)
    df['rsi14'] = calc_rsi(df['close'], 14)
    df['atr14'] = calc_atr(df, 14)
    df['vwap']  = calc_vwap(df)

    safe_open  = np.where(df['open'] == 0, 1e-5, df['open'])
    safe_close = np.where(df['close'] == 0, 1e-5, df['close'])
    safe_high  = np.maximum(df['high'], 1e-5)
    safe_low   = np.where(df['low'] == 0, 1e-5, df['low'])
    prev_close = df['close'].shift(1).fillna(df['open'])

    ln_co  = np.log(safe_close / safe_open)
    ln_hc  = np.log(safe_high / safe_close)
    ln_ho  = np.log(safe_high / safe_open)
    ln_lc  = np.log(safe_low / safe_close)
    ln_lo  = np.log(safe_low / safe_open)
    ln_oc1 = np.log(safe_open / prev_close)

    def calc_f_vol(n):
        v_on = df['ln_oc1'].rolling(n).var(ddof=1)
        v_oc = df['ln_co'].rolling(n).var(ddof=1)
        v_rs = (df['ln_hc'] * df['ln_ho'] + df['ln_lc'] * df['ln_lo']).rolling(n).mean()
        k = 0.34 / (1.34 + (n + 1.0) / max(n - 1.0, 1.0))
        return np.sqrt(np.maximum(v_on + k * v_oc + (1.0 - k) * np.maximum(v_rs, 0.0), 0.0))

    df['ln_co']  = ln_co
    df['ln_hc']  = ln_hc
    df['ln_ho']  = ln_ho
    df['ln_lc']  = ln_lc
    df['ln_lo']  = ln_lo
    df['ln_oc1'] = ln_oc1

    df['vol_yz']   = calc_f_vol(20)
    df['vol_s']    = calc_f_vol(10)
    df['vol_l']    = calc_f_vol(60)
    df['ts_ratio'] = np.where(df['vol_l'] > 0.0, df['vol_s'] / df['vol_l'], 1.0)

    df['vol_pct'] = df['vol_yz'].rolling(150, min_periods=30).apply(
        lambda x: (x[:-1] < x[-1]).sum() * 100.0 / (len(x) - 1) if len(x) > 1 else 50.0,
        raw=True
    ).fillna(50.0)

    regime_list = []
    cmp_dur, regime = 0, 0
    for i in range(len(df)):
        v_pct = df['vol_pct'].iloc[i]
        ts_r  = df['ts_ratio'].iloc[i]
        cmp_dur = cmp_dur + 1 if v_pct <= 25.0 else 0
        press = min(1.0, max(0.0, (35.0 - v_pct) / 35.0) * 0.5 + min(cmp_dur / 40.0, 1.0) * 0.3) * 100.0

        if v_pct >= 72.0 or ts_r >= 1.25:
            new_regime = 2
        elif press >= 49.0 and v_pct <= 35.0:
            new_regime = 1
        elif regime == 2 and v_pct < 72.0:
            new_regime = 3
        elif regime == 3 and v_pct < 45.0 and ts_r < 1.0:
            new_regime = 0
        elif regime == 3:
            new_regime = 3
        else:
            new_regime = 0
        regime = new_regime
        regime_list.append(regime)

    df['is_storm_valid'] = (pd.Series(regime_list, index=df.index) == 1) | (pd.Series(regime_list, index=df.index) == 2)

    total_trades, win_tp1, win_tp2, losses, timeouts = 0, 0, 0, 0, 0
    active = False
    dir_val, stop, t1, t2, open_bar, hit_tp1 = 0, 0.0, 0.0, 0.0, 0, False

    for i in range(30, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        if active:
            hit_stop = (row['low'] <= stop) if dir_val == 1 else (row['high'] >= stop)
            hit1     = (row['high'] >= t1)   if dir_val == 1 else (row['low'] <= t1)
            hit2     = (row['high'] >= t2)   if dir_val == 1 else (row['low'] <= t2)
            timeout  = (i - open_bar) >= max_bars

            if hit1:
                hit_tp1 = True

            if hit2:
                total_trades += 1
                win_tp1 += 1
                win_tp2 += 1
                active = False
            elif hit_stop:
                total_trades += 1
                if hit_tp1:
                    win_tp1 += 1
                else:
                    losses += 1
                active = False
            elif timeout:
                total_trades += 1
                if hit_tp1:
                    win_tp1 += 1
                else:
                    timeouts += 1
                active = False

        if not active:
            trend_up = row['ema21'] > row['ema50'] and df['ema50'].iloc[i] > df['ema50'].iloc[i-3]
            trend_dn = row['ema21'] < row['ema50'] and df['ema50'].iloc[i] < df['ema50'].iloc[i-3]
            touched_up = any(df['low'].iloc[i-3:i+1] <= df['ema21'].iloc[i-3:i+1])
            touched_dn = any(df['high'].iloc[i-3:i+1] >= df['ema21'].iloc[i-3:i+1])
            s_ema_l = trend_up and touched_up and row['close'] > row['ema9'] and row['close'] > row['open'] and row['close'] > prev['high']
            s_ema_s = trend_dn and touched_dn and row['close'] < row['ema9'] and row['close'] < row['open'] and row['close'] < prev['low']

            sw_lo = df['low'].iloc[i-21:i].min()
            sw_hi = df['high'].iloc[i-21:i].max()
            s_sweep_l = row['low'] < sw_lo and row['close'] > sw_lo and row['close'] > row['open']
            s_sweep_s = row['high'] > sw_hi and row['close'] < sw_hi and row['close'] < row['open']

            vwap_cross_l = prev['close'] <= prev['vwap'] and row['close'] > row['vwap']
            vwap_cross_s = prev['close'] >= prev['vwap'] and row['close'] < row['vwap']
            below_run = (df['close'].iloc[i-7:i] < df['vwap'].iloc[i-7:i]).sum() >= 6
            above_run = (df['close'].iloc[i-7:i] > df['vwap'].iloc[i-7:i]).sum() >= 6
            s_vwap_l = vwap_cross_l and below_run
            s_vwap_s = vwap_cross_s and above_run

            raw_l = s_ema_l or s_sweep_l or s_vwap_l
            raw_s = s_ema_s or s_sweep_s or s_vwap_s

            a_plus_l = raw_l and row['is_storm_valid'] and row['htf_bull'] and (row['rsi14'] <= 70.0)
            a_plus_s = raw_s and row['is_storm_valid'] and row['htf_bear'] and (row['rsi14'] >= 30.0)

            if a_plus_l or a_plus_s:
                dir_val = 1 if a_plus_l else -1
                entry = row['close']
                risk = row['atr14'] * stop_atr
                stop = entry - risk if dir_val == 1 else entry + risk
                t1 = entry + risk * r1 if dir_val == 1 else entry - risk * r1
                t2 = entry + risk * r2 if dir_val == 1 else entry - risk * r2
                hit_tp1 = False
                open_bar = i
                active = True

    wr1 = (win_tp1 / total_trades * 100.0) if total_trades > 0 else 0.0
    wr2 = (win_tp2 / total_trades * 100.0) if total_trades > 0 else 0.0
    return {'total': total_trades, 'win1': win_tp1, 'win2': win_tp2, 'loss': losses, 'wr1': wr1, 'wr2': wr2}

# ═════════════════════════════════════════════════════════════════════════════
#  LIVE SCANNER
# ═════════════════════════════════════════════════════════════════════════════
async def check_active_trade_outcomes(data_b):
    global active_trades
    c = data_b['close'].iloc[-1]
    h = data_b['high'].iloc[-1]
    l = data_b['low'].iloc[-1]
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    still_active = []
    for t in active_trades:
        hit_stop = (l <= t['sl'])  if t['side'] == 'BUY' else (h >= t['sl'])
        hit_tp1  = (h >= t['tp1']) if t['side'] == 'BUY' else (l <= t['tp1'])
        hit_tp2  = (h >= t['tp2']) if t['side'] == 'BUY' else (l <= t['tp2'])

        if hit_tp2 and t['status'] != 'HIT_TP2':
            t['status'] = 'CLOSED_TP2'
            update_csv_trade_status(t['trade_id'], 'CLOSED_TP2', 2.0, now_str)
            msg = f"🚀 *[TP2 HIT - FULL WIN]* 🚀\n\n📌 `{t['pair']}` *[{t['tf']}]* ({t['side']})\n🎯 ราคาชน TP2 สำเร็จ (+2.0R) ✅"
            log_event(f"🚀 {t['pair']} [{t['tf']}] Hit TP2 (+2.0R)")
            await send_telegram(msg)

        elif hit_tp1 and t['status'] == 'OPEN':
            t['status'] = 'HIT_TP1'
            update_csv_trade_status(t['trade_id'], 'HIT_TP1', 0.8, now_str)
            msg = f"🎯 *[TP1 HIT]* 🎯\n\n📌 `{t['pair']}` *[{t['tf']}]* ({t['side']})\n✨ ชนเป้า TP1 สำเร็จ (+0.8R)"
            log_event(f"🎯 {t['pair']} [{t['tf']}] Hit TP1 (+0.8R)")
            await send_telegram(msg)
            still_active.append(t)

        elif hit_stop:
            final_status = 'SL_AFTER_TP1' if t['status'] == 'HIT_TP1' else 'CLOSED_SL'
            pnl = 0.8 if t['status'] == 'HIT_TP1' else -1.0
            update_csv_trade_status(t['trade_id'], final_status, pnl, now_str)
            msg = f"🛑 *[STOP LOSS TRIGGERED]*\n\n📌 `{t['pair']}` *[{t['tf']}]* ({t['side']})\nราคาโดนจุด SL | สถานะ: `{final_status}`"
            log_event(f"🛑 {t['pair']} [{t['tf']}] Hit SL")
            await send_telegram(msg)

        else:
            still_active.append(t)

    active_trades = still_active

async def scan_live_pair(name, yf_ticker, tf_base, tf_htf):
    tracker_key = f"{name}_{tf_base}"
    try:
        data_b = yf.download(yf_ticker, period='5d', interval=tf_base, progress=False)
        data_h = yf.download(yf_ticker, period='10d', interval=tf_htf, progress=False)
        
        if isinstance(data_b.columns, pd.MultiIndex):
            data_b.columns = data_b.columns.get_level_values(0)
        if isinstance(data_h.columns, pd.MultiIndex):
            data_h.columns = data_h.columns.get_level_values(0)
            
        data_b.columns = [c.lower() for c in data_b.columns]
        data_h.columns = [c.lower() for c in data_h.columns]
        
        if len(data_b) < 30 or len(data_h) < 30:
            return

        if data_b.index.tz is not None:
            data_b.index = data_b.index.tz_convert('UTC').tz_localize(None)

        await check_active_trade_outcomes(data_b)

        last_time = str(data_b.index[-2])
        c = data_b['close'].iloc[-2]
        atr = calc_atr(data_b, 14).iloc[-2]
        ema9 = calc_ema(data_b['close'], 9).iloc[-2]
        ema21 = calc_ema(data_b['close'], 21).iloc[-2]
        rsi = calc_rsi(data_b['close'], 14).iloc[-2]

        is_buy  = (c > ema9) and (ema9 > ema21) and (rsi <= 70.0)
        is_sell = (c < ema9) and (ema9 < ema21) and (rsi >= 30.0)

        if (is_buy or is_sell) and alerted_tracker.get(tracker_key) != last_time:
            alerted_tracker[tracker_key] = last_time
            side = "BUY" if is_buy else "SELL"
            emoji = "🟢" if is_buy else "🔴"
            risk = atr * 1.3
            sl = c - risk if is_buy else c + risk
            tp1 = c + (risk * 0.8) if is_buy else c - (risk * 0.8)
            tp2 = c + (risk * 2.0) if is_buy else c - (risk * 2.0)
            
            trade_id = f"T_{int(datetime.now().timestamp())}_{name[:3]}"

            new_trade = {
                'trade_id': trade_id,
                'timestamp': last_time,
                'pair': name,
                'tf': tf_base,
                'side': side,
                'setup': 'Storm A+ Trend',
                'entry': round(c, 5),
                'sl': round(sl, 5),
                'tp1': round(tp1, 5),
                'tp2': round(tp2, 5),
                'status': 'OPEN',
                'pnl_r': 0.0,
                'close_time': ''
            }
            log_trade_to_csv(new_trade)
            active_trades.append(new_trade)

            msg = (
                f"⚡ *[STORM A+ {side} SIGNAL]* ⚡\n\n"
                f"📊 *Asset:* `{name}` *(TF: {tf_base})*\n"
                f"🎯 *Direction:* {emoji} *{side}*\n"
                f"🆔 *Trade ID:* `{trade_id}`\n\n"
                f"💵 *Entry:* `{c:.5f}`\n"
                f"🛑 *Stop Loss:* `{sl:.5f}`\n"
                f"🎯 *TP1 (0.8R):* `{tp1:.5f}`\n"
                f"🚀 *TP2 (2.0R):* `{tp2:.5f}`\n"
                f"⏰ *Bar Time:* `{last_time}`"
            )
            log_event(f"🚨 Alert Sent: {name} [{tf_base}] ({side}) | ID: {trade_id}")
            await send_telegram(msg)
            
    except Exception as e:
        pass

# ═════════════════════════════════════════════════════════════════════════════
#  DASHBOARD WEB PAGE (AUTO-REFRESHING UI)
# ═════════════════════════════════════════════════════════════════════════════
async def handle_dashboard(request):
    global system_status
    rows_html = ""
    if os.path.exists(LOG_FILE_PATH):
        try:
            df_logs = pd.read_csv(LOG_FILE_PATH)
            for _, r in df_logs.tail(15).iloc[::-1].iterrows():
                badge_col = "#10b981" if r['side'] == 'BUY' else "#ef4444"
                rows_html += f"""<tr>
                    <td>{r['timestamp']}</td>
                    <td><b>{r['pair']}</b></td>
                    <td>{r['tf']}</td>
                    <td style="color:{badge_col}; font-weight:bold;">{r['side']}</td>
                    <td>{r['entry']}</td>
                    <td>{r['sl']}</td>
                    <td>{r['tp1']}</td>
                    <td><span style="background:#334155; padding:2px 8px; border-radius:4px;">{r['status']}</span></td>
                </tr>"""
        except Exception:
            pass

    logs_text = "\n".join(live_logs_stream[-20:])

    html_content = f"""<!DOCTYPE html>
    <html lang="th">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="refresh" content="5">
        <title>⚡ Storm A+ Live Dashboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0f19; color: #f1f5f9; margin: 0; padding: 20px; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; background: #1e293b; padding: 20px 25px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; }}
            .title {{ font-size: 22px; font-weight: bold; color: #38bdf8; display: flex; align-items: center; gap: 10px; }}
            .badge-live {{ background: #10b981; color: white; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: bold; animation: pulse 2s infinite; }}
            .card {{ background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; }}
            .card-title {{ font-size: 17px; font-weight: bold; margin-bottom: 12px; color: #94a3b8; display: flex; align-items: center; gap: 8px; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
            th {{ background: #0f172a; padding: 12px; color: #94a3b8; font-weight: 600; border-bottom: 2px solid #334155; }}
            td {{ padding: 12px; border-bottom: 1px solid #334155; }}
            .log-terminal {{ background: #000; color: #22c55e; font-family: 'Courier New', monospace; font-size: 13px; padding: 15px; border-radius: 8px; height: 180px; overflow-y: auto; white-space: pre-wrap; line-height: 1.4; }}
            @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">⚡ Storm A+ Live Scanner</div>
                <div><span class="badge-live">● RUNNING 24/7</span></div>
            </div>

            <div class="card">
                <div class="card-title">📡 สถานะการทำงานปัจจุบัน: <span style="color:#f8fafc;">{system_status}</span></div>
                <div class="log-terminal">{logs_text if logs_text else 'กำลังเริ่มระบบ...'}</div>
            </div>

            <div class="card">
                <div class="card-title">📝 รายการออเดอร์ล่าสุด (Trade Logs)</div>
                <table>
                    <thead>
                        <tr>
                            <th>เวลา</th>
                            <th>คู่เงิน</th>
                            <th>TF</th>
                            <th>Side</th>
                            <th>Entry</th>
                            <th>SL</th>
                            <th>TP1</th>
                            <th>สถานะ</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html if rows_html else '<tr><td colspan="8" style="text-align:center; color:#64748b;">ยังไม่มีประวัติการส่งสัญญาณ</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>"""
    return web.Response(text=html_content, content_type="text/html")

async def start_web_server():
    port = int(os.environ.get("PORT", 10000))
    app = web.Application()
    app.router.add_get("/", handle_dashboard)
    app.router.add_get("/health", lambda r: web.Response(text="OK"))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log_event(f"Web Server listening on port {port} (Dashboard ready)")

# ═════════════════════════════════════════════════════════════════════════════
#  ASYNC WORKER
# ═════════════════════════════════════════════════════════════════════════════
async def scanner_worker():
    global system_status, qualified_list
    await asyncio.sleep(2)
    log_event(f"🚀 [PHASE 1] เริ่มต้น Backtest 29 คู่ × 3 Timeframes (87 แบบจำลอง)...")
    system_status = "กำลังทำการ Backtest คัดกรองคู่เงิน (87 รูปแบบ)..."

    results_table = []
    qualified_list = []
    total_count = len(YF_SYMBOLS_MAP) * len(TF_CONFIGS)
    current_idx = 0

    for name, yf_ticker in YF_SYMBOLS_MAP.items():
        for tf_base, tf_htf, p_base, p_htf in TF_CONFIGS:
            current_idx += 1
            try:
                data_b = yf.download(yf_ticker, period=p_base, interval=tf_base, progress=False)
                data_h = yf.download(yf_ticker, period=p_htf,  interval=tf_htf,  progress=False)
                
                if isinstance(data_b.columns, pd.MultiIndex):
                    data_b.columns = data_b.columns.get_level_values(0)
                if isinstance(data_h.columns, pd.MultiIndex):
                    data_h.columns = data_h.columns.get_level_values(0)
                    
                data_b.columns = [c.lower() for c in data_b.columns]
                data_h.columns = [c.lower() for c in data_h.columns]
                
                if data_b.index.tz is not None:
                    data_b.index = data_b.index.tz_convert('UTC').tz_localize(None)
                if data_h.index.tz is not None:
                    data_h.index = data_h.index.tz_convert('UTC').tz_localize(None)
                    
                stats = run_backtest_simulation(data_b, data_h)
                passed = (stats['wr1'] >= WINRATE_THRESHOLD) and (stats['total'] >= 3)
                
                log_event(f"[{current_idx}/{total_count}] {name:<16} [{tf_base}] -> WR: {stats['wr1']:.1f}% ({'✅ ผ่าน' if passed else '❌'})")
                
                results_table.append({
                    'Pair': name,
                    'TF': tf_base,
                    'HTF': tf_htf,
                    'Total': stats['total'],
                    'Wins': stats['win1'],
                    'Losses': stats['loss'],
                    'WinRate_TP1': stats['wr1'],
                    'Passed': passed
                })
                
                if passed:
                    qualified_list.append((name, yf_ticker, tf_base, tf_htf, stats['wr1'], stats['total'], stats['win1'], stats['loss']))
                
                await asyncio.sleep(0.2)
                    
            except Exception as e:
                log_event(f"Error {name} [{tf_base}]: {e}")

    log_event(f"🏆 คัดเลือกคู่ผ่านเกณฑ์ได้ทั้งหมด {len(qualified_list)} รายการ")
    system_status = f"สแกน Real-Time คู่ที่ผ่านเกณฑ์ ({len(qualified_list)} รายการ)"

    # ส่งสรุปเข้า Telegram
    tg_report = f"📊 *[STORM A+ MULTI-TIMEFRAME REPORT]*\n"
    tg_report += f"🎯 *เกณฑ์คัดเลือก:* Win Rate >= {WINRATE_THRESHOLD}%\n"
    tg_report += f"⏱ *Timeframes ทดสอบ:* `5m, 15m, 1h`\n\n"
    tg_report += f"🏆 *รายการที่ผ่านเกณฑ์ ({len(qualified_list)} รายการ):*\n"
    
    if qualified_list:
        qualified_sorted = sorted(qualified_list, key=lambda x: x, reverse=True)
        for name, _, tf, _, wr, tot, w, l in qualified_sorted:
            tg_report += f"• `{name}` *[{tf}]* : *{wr:.1f}%* ({tot} ไม้ | W:{w} L:{l})\n"
    else:
        tg_report += "• ไม่มีคู่ที่ผ่านเกณฑ์ในช่วงเวลานี้\n"
        
    tg_report += f"\n📁 *Trade Logger:* บันทึกไม้ลง `storm_trade_logs.csv` อัตโนมัติ"
    tg_report += f"\n⏰ *รายงาน ณ เวลา:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
    
    await send_telegram(tg_report)

    if not qualified_list:
        log_event("⚠️ ไม่มีคู่ที่ผ่านเกณฑ์")
        return

    log_event("🚀 [PHASE 2] เริ่มระบบ Live Scanner ตรวจสอบทุก 30 วินาที...")
    while True:
        tasks = []
        for name, yf_ticker, tf_base, tf_htf, _, _, _, _ in qualified_list:
            tasks.append(scan_live_pair(name, yf_ticker, tf_base, tf_htf))
        
        await asyncio.gather(*tasks)
        await asyncio.sleep(30)

async def main():
    await start_web_server()
    await scanner_worker()

if __name__ == '__main__':
    asyncio.run(main())
