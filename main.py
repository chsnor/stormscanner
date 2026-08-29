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

sys_stdout = True
nest_asyncio.apply()

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG & CREDENTIALS
# ═════════════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN = "8907507767:AAEv4LMGq7vB9Zm76jaLbw3Iy5T66wJxtKw"
TELEGRAM_CHAT_ID   = "5861943388"
LOG_FILE_PATH      = "storm_trade_logs.csv"

# ลิสต์ 33 รายการเกรด A+ ที่ผ่านการคัดเลือก (Win Rate >= 55%)
A_PLUS_WATCHLIST = [
    # Metals & Crypto
    ('XAU/USD (Gold)', 'GC=F', '5m', '15m', 61.3),
    ('XAU/USD (Gold)', 'GC=F', '15m', '1h', 60.0),
    ('XAG/USD (Silver)', 'SI=F', '5m', '15m', 58.3),
    ('XAG/USD (Silver)', 'SI=F', '15m', '1h', 62.2),
    ('BTC/USD', 'BTC-USD', '5m', '15m', 57.1),
    ('BTC/USD', 'BTC-USD', '15m', '1h', 56.0),
    ('BTC/USD', 'BTC-USD', '1h', '1d', 59.9),
    ('ETH/USD', 'ETH-USD', '5m', '15m', 58.0),
    ('ETH/USD', 'ETH-USD', '1h', '1d', 56.3),
    
    # Forex Crosses & Majors (15m & 5m)
    ('GBP/NZD', 'GBPNZD=X', '15m', '1h', 62.6),
    ('GBP/JPY', 'GBPJPY=X', '5m', '15m', 62.4),
    ('USD/CHF', 'USDCHF=X', '15m', '1h', 59.2),
    ('EUR/AUD', 'EURAUD=X', '15m', '1h', 58.9),
    ('EUR/JPY', 'EURJPY=X', '15m', '1h', 58.5),
    ('USD/CAD', 'USDCAD=X', '15m', '1h', 57.8),
    ('EUR/AUD', 'EURAUD=X', '5m', '15m', 57.4),
    ('EUR/JPY', 'EURJPY=X', '5m', '15m', 57.3),
    ('AUD/USD', 'AUDUSD=X', '15m', '1h', 57.1),
    ('GBP/CAD', 'GBPCAD=X', '15m', '1h', 56.0),
    ('EUR/CAD', 'EURCAD=X', '15m', '1h', 55.6),
    
    # Forex Majors & Crosses (1h)
    ('GBP/AUD', 'GBPAUD=X', '1h', '1d', 58.3),
    ('AUD/CHF', 'AUDCHF=X', '1h', '1d', 58.1),
    ('AUD/CAD', 'AUDCAD=X', '1h', '1d', 57.8),
    ('AUD/JPY', 'AUDJPY=X', '1h', '1d', 57.4),
    ('EUR/AUD', 'EURAUD=X', '1h', '1d', 57.1),
    ('GBP/CHF', 'GBPCHF=X', '1h', '1d', 56.5),
    ('EUR/GBP', 'EURGBP=X', '1h', '1d', 56.0),
    ('NZD/USD', 'NZDUSD=X', '1h', '1d', 55.7),
    ('CHF/JPY', 'CHFJPY=X', '1h', '1d', 55.5),
    ('EUR/NZD', 'EURNZD=X', '1h', '1d', 55.4),
    ('GBP/CAD', 'GBPCAD=X', '1h', '1d', 55.2),
    ('EUR/JPY', 'EURJPY=X', '1h', '1d', 55.1),
    ('GBP/USD', 'GBPUSD=X', '1h', '1d', 55.1),
]

alerted_tracker  = {}
active_trades    = []
live_logs_stream = []

# ═════════════════════════════════════════════════════════════════════════════
#  INDICATORS & CSV LOGGER
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

async def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(url, json=payload)
        except Exception as e:
            log_event(f"Telegram error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
#  LIVE SCANNER & TRADE TRACKER
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

async def scan_live_pair(name, yf_ticker, tf_base, tf_htf, winrate):
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
                f"⭐ *Historical Win Rate:* `{winrate:.1f}%`\n"
                f"🎯 *Direction:* {emoji} *{side}*\n"
                f"🆔 *Trade ID:* `{trade_id}`\n\n"
                f"💵 *Entry:* `{c:.5f}`\n"
                f"🛑 *Stop Loss:* `{sl:.5f}`\n"
                f"🎯 *TP1 (0.8R):* `{tp1:.5f}`\n"
                f"🚀 *TP2 (2.0R):* `{tp2:.5f}`\n"
                f"⏰ *Bar Time:* `{last_time}`\n"
                f"💾 *Saved to Logs:* ✅ `storm_trade_logs.csv`"
            )
            log_event(f"🚨 Alert: {name} [{tf_base}] ({side}) | ID: {trade_id}")
            await send_telegram(msg)
            
    except Exception as e:
        pass

# ═════════════════════════════════════════════════════════════════════════════
#  DASHBOARD WEB PAGE
# ═════════════════════════════════════════════════════════════════════════════
async def handle_dashboard(request):
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
            .badge-live {{ background: #10b981; color: white; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: bold; }}
            .card {{ background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; }}
            .card-title {{ font-size: 17px; font-weight: bold; margin-bottom: 12px; color: #94a3b8; display: flex; align-items: center; gap: 8px; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
            th {{ background: #0f172a; padding: 12px; color: #94a3b8; font-weight: 600; border-bottom: 2px solid #334155; }}
            td {{ padding: 12px; border-bottom: 1px solid #334155; }}
            .log-terminal {{ background: #000; color: #22c55e; font-family: 'Courier New', monospace; font-size: 13px; padding: 15px; border-radius: 8px; height: 180px; overflow-y: auto; white-space: pre-wrap; line-height: 1.4; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">⚡ Storm A+ Live Scanner</div>
                <div><span class="badge-live">● RUNNING 24/7 ({len(A_PLUS_WATCHLIST)} คู่เกรด A+)</span></div>
            </div>

            <div class="card">
                <div class="card-title">📡 Real-Time Logs</div>
                <div class="log-terminal">{logs_text if logs_text else 'ระบบกำลังเฝ้ากราฟ 33 รายการเกรด A+...'}</div>
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
    log_event(f"Web Server listening on port {port} (Live Dashboard Ready)")

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════
async def scanner_worker():
    await asyncio.sleep(2)
    log_event(f"🚀 เริ่มต้นสแกน 33 รายการเกรด A+ แบบ Real-Time ทันที (วนตรวจทุก 30 วิ)...")

    # ส่งสรุปยืนยันการเปิดระบบเข้า Telegram
    summary_msg = f"🏆 *[STORM A+ LIVE SCANNER STARTED]* 🏆\nระบบเริ่มสแกนสด 33 รายการเกรด A+ (Win Rate >= 55%) ตลอด 24 ชม. แล้วครับ!\n\n"
    summary_msg += f"• *ทองคำ & เงิน:* `XAU/USD`, `XAG/USD` (5m, 15m)\n"
    summary_msg += f"• *Crypto:* `BTC/USD`, `ETH/USD` (5m, 15m, 1h)\n"
    summary_msg += f"• *Forex Crosses & Majors:* `GBP/NZD`, `GBP/JPY`, `USD/CHF`, `EUR/AUD` ฯลฯ\n\n"
    summary_msg += f"🌐 *Live Dashboard:* https://stormscanner.onrender.com"
    await send_telegram(summary_msg)

    while True:
        tasks = []
        for name, yf_ticker, tf_base, tf_htf, wr in A_PLUS_WATCHLIST:
            tasks.append(scan_live_pair(name, yf_ticker, tf_base, tf_htf, wr))
        
        await asyncio.gather(*tasks)
        await asyncio.sleep(30)

async def main():
    await asyncio.gather(
        start_web_server(),
        scanner_worker()
    )

if __name__ == '__main__':
    asyncio.run(main())
