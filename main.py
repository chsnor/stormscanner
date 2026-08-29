import os
import csv
import json
import urllib.request
import asyncio
import nest_asyncio
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from aiohttp import web

nest_asyncio.apply()

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG & CREDENTIALS
# ═════════════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
LOG_FILE_PATH      = "storm_trade_logs.csv"

# Storm Scanner Config (matching Pine Script defaults)
VOL_LEN      = 20
STORM_PCT    = 72
WATCH_TH     = 70
AUTO_RSI     = True
RSI_LEN      = 14
RSI_MAX_BUY  = 70.0
RSI_MIN_SELL = 30.0
STOP_ATR     = 1.3
R1           = 0.8
R2           = 2.0
MAX_BARS     = 60
USE_ORB      = False  # Pine default=false; เปิดเฉพาะ Forex/Gold NY Session

# Data periods — เพิ่มให้พอสำหรับ EMA200 + PercentRank300
BASE_PERIOD = {'5m': '7d', '15m': '30d', '1h': '60d'}
HTF_PERIOD  = {'15m': '30d', '1h': '60d', '1d': '2y'}

# ลิสต์ 37 รายการเกรด A+ ที่ผ่านเกณฑ์ Win Rate >= 55% จากการทดสอบ Full Pine Script Logic Engine
A_PLUS_WATCHLIST = [
    # Top Tier (WR >= 60%)
    ('EUR/AUD', 'EURAUD=X', '5m', '15m', 63.4),
    ('XAU/USD (Gold)', 'GC=F', '5m', '15m', 61.6),
    ('GBP/AUD', 'GBPAUD=X', '1h', '1d', 61.2),
    ('XAG/USD (Silver)', 'SI=F', '15m', '1h', 60.6),
    ('AUD/JPY', 'AUDJPY=X', '1h', '1d', 60.3),
    ('EUR/AUD', 'EURAUD=X', '15m', '1h', 60.2),

    # High Tier (WR 58% - 59.9%)
    ('CAD/JPY', 'CADJPY=X', '15m', '1h', 59.9),
    ('GBP/CHF', 'GBPCHF=X', '1h', '1d', 59.8),
    ('NZD/JPY', 'NZDJPY=X', '15m', '1h', 59.5),
    ('EUR/JPY', 'EURJPY=X', '1h', '1d', 59.4),
    ('GBP/NZD', 'GBPNZD=X', '15m', '1h', 59.2),
    ('EUR/NZD', 'EURNZD=X', '1h', '1d', 59.0),
    ('EUR/CAD', 'EURCAD=X', '15m', '1h', 58.9),
    ('AUD/CHF', 'AUDCHF=X', '1h', '1d', 58.8),
    ('CAD/JPY', 'CADJPY=X', '1h', '1d', 58.7),
    ('AUD/CAD', 'AUDCAD=X', '1h', '1d', 58.7),
    ('EUR/GBP', 'EURGBP=X', '1h', '1d', 58.5),
    ('XAG/USD (Silver)', 'SI=F', '5m', '15m', 58.4),
    ('EUR/JPY', 'EURJPY=X', '15m', '1h', 58.3),
    ('XAU/USD (Gold)', 'GC=F', '15m', '1h', 58.3),
    ('AUD/USD', 'AUDUSD=X', '15m', '1h', 58.2),

    # Solid Tier (WR 56% - 57.9%)
    ('USD/JPY', 'USDJPY=X', '1h', '1d', 58.0),
    ('USD/JPY', 'USDJPY=X', '15m', '1h', 57.6),
    ('AUD/NZD', 'AUDNZD=X', '15m', '1h', 57.3),
    ('BTC/USD', 'BTC-USD', '1h', '1d', 57.1),
    ('XAU/USD (Gold)', 'GC=F', '1h', '1d', 57.0),
    ('ETH/USD', 'ETH-USD', '15m', '1h', 56.9),
    ('CHF/JPY', 'CHFJPY=X', '1h', '1d', 56.8),
    ('AUD/USD', 'AUDUSD=X', '1h', '1d', 56.7),
    ('USD/CHF', 'USDCHF=X', '1h', '1d', 56.6),
    ('BTC/USD', 'BTC-USD', '5m', '15m', 56.5),
    ('NZD/USD', 'NZDUSD=X', '1h', '1d', 56.4),
    ('NZD/JPY', 'NZDJPY=X', '1h', '1d', 56.3),

    # Qualified Tier (WR 55% - 55.9%)
    ('EUR/CHF', 'EURCHF=X', '1h', '1d', 55.7),
    ('USD/CHF', 'USDCHF=X', '15m', '1h', 55.6),
    ('GBP/JPY', 'GBPJPY=X', '15m', '1h', 55.2),
    ('GBP/CAD', 'GBPCAD=X', '15m', '1h', 55.0),
]

alerted_tracker  = {}
active_trades    = []
live_logs_stream = []

# ═════════════════════════════════════════════════════════════════════════════
#  INDICATOR FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def calc_ema(series, span):
    """EMA — ตรงกับ Pine ta.ema"""
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi(series, length=14):
    """RSI ใช้ RMA — ตรงกับ Pine ta.rsi"""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calc_atr(df, length=14):
    """ATR ใช้ RMA — ตรงกับ Pine ta.atr"""
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def calc_yang_zhang_vol(df, n):
    """Yang-Zhang Volatility Estimator — ตรงกับ Pine f_vol()"""
    safe_close = df['close'].replace(0, 1e-5)
    safe_open  = df['open'].replace(0, 1e-5)
    safe_low   = df['low'].replace(0, 1e-5)
    safe_high  = df['high'].clip(lower=1e-5)

    lnCO  = np.log(safe_close / safe_open)
    lnHC  = np.log(safe_high / safe_close)
    lnHO  = np.log(safe_high / safe_open)
    lnLC  = np.log(safe_low / safe_close)
    lnLO  = np.log(safe_low / safe_open)
    lnOC1 = np.log(safe_open / safe_close.shift(1).fillna(safe_open))

    vON = lnOC1.rolling(n).std() ** 2
    vOC = lnCO.rolling(n).std() ** 2
    vRS = (lnHC * lnHO + lnLC * lnLO).rolling(n).mean()
    k   = 0.34 / (1.34 + (n + 1.0) / max(n - 1.0, 1.0))

    return np.sqrt((vON + k * vOC + (1.0 - k) * vRS.clip(lower=0)).clip(lower=0))


def calc_percentile_rank(series, length):
    """Percentile rank — ตรงกับ Pine ta.percentrank"""
    values = series.values
    n = len(values)
    result = np.full(n, 50.0)
    for i in range(1, n):
        start = max(0, i - length)
        window = values[start:i]
        valid_mask = ~np.isnan(window)
        valid_count = np.count_nonzero(valid_mask)
        if valid_count > 0 and not np.isnan(values[i]):
            result[i] = np.nansum(window <= values[i]) / valid_count * 100.0
    return pd.Series(result, index=series.index)


def calc_vwap(df):
    """Session VWAP (daily reset) — ตรงกับ Pine ta.vwap"""
    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        return hlc3  # fallback ถ้าไม่มี volume
    volume = df['volume'].clip(lower=1e-10)
    dates = df.index.date
    vwap = pd.Series(index=df.index, dtype=float)
    for date in np.unique(dates):
        mask = dates == date
        vol = volume[mask]
        cum_vol = vol.cumsum()
        vwap[mask] = (hlc3[mask] * vol).cumsum() / cum_vol
    return vwap


def calc_pivot_high(series, left, right):
    """Pivot High — ตรงกับ Pine ta.pivothigh(high, 5, 5)"""
    result = pd.Series(np.nan, index=series.index)
    for i in range(left + right, len(series)):
        pivot_idx = i - right
        pv = series.iloc[pivot_idx]
        lw = series.iloc[pivot_idx - left:pivot_idx]
        rw = series.iloc[pivot_idx + 1:i + 1]
        if len(lw) >= left and len(rw) >= right and (pv >= lw).all() and (pv >= rw).all():
            result.iloc[i] = pv
    return result


def calc_pivot_low(series, left, right):
    """Pivot Low — ตรงกับ Pine ta.pivotlow(low, 5, 5)"""
    result = pd.Series(np.nan, index=series.index)
    for i in range(left + right, len(series)):
        pivot_idx = i - right
        pv = series.iloc[pivot_idx]
        lw = series.iloc[pivot_idx - left:pivot_idx]
        rw = series.iloc[pivot_idx + 1:i + 1]
        if len(lw) >= left and len(rw) >= right and (pv <= lw).all() and (pv <= rw).all():
            result.iloc[i] = pv
    return result


def tp1_first_heuristic(was_hit_tp1, direction, t1_level, hit1, bar_close, bar_open):
    """[F3] Body direction heuristic — ตรงกับ Pine f_tp1First"""
    bar_bullish = bar_close > bar_open
    gap_open_past = (bar_open >= t1_level) if direction == 1 else (bar_open <= t1_level)
    body_match = hit1 and (bar_bullish if direction == 1 else not bar_bullish)
    return was_hit_tp1 or gap_open_past or body_match


# ═════════════════════════════════════════════════════════════════════════════
#  CSV LOGGER
# ═════════════════════════════════════════════════════════════════════════════
CSV_FIELDS = [
    'trade_id', 'timestamp', 'pair', 'tf', 'side', 'setup',
    'entry', 'sl', 'tp1', 'tp2', 'status', 'pnl_r', 'close_time'
]

def init_csv():
    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

def log_trade_to_csv(trade_dict):
    init_csv()
    row = {k: trade_dict.get(k, '') for k in CSV_FIELDS}
    with open(LOG_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

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
#  TELEGRAM
# ═════════════════════════════════════════════════════════════════════════════
def send_telegram_sync(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log_event("⚠️ Telegram token/chat_id not set — skipping send")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message,
                          "parse_mode": "Markdown"}).encode('utf-8')
    req = urllib.request.Request(url, data=payload,
                                headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as e:
        log_event(f"Telegram error: {e}")

async def send_telegram(message):
    await asyncio.to_thread(send_telegram_sync, message)


# ═════════════════════════════════════════════════════════════════════════════
#  VIRTUAL TRACKER & AI OPTIMIZER  (ตรงกับ Pine Section 5)
# ═════════════════════════════════════════════════════════════════════════════
class VirtualStats:
    """ตรงกับ Pine UDT VirtualStats"""
    def __init__(self):
        self.total = 0
        self.win1  = 0
        self.win2  = 0
        self.loss  = 0
        self.tOuts = 0

class VirtualTrade:
    """ตรงกับ Pine UDT VirtualTrade"""
    def __init__(self):
        self.dir      = 0
        self.entry    = 0.0
        self.stop     = 0.0
        self.t1       = 0.0
        self.t2       = 0.0
        self.open_bar = 0
        self.active   = False
        self.hit_tp1  = False


def update_virtual_trade(t, s, trig_l, trig_s, bar_idx,
                         o, h, l, c, atr_val):
    """ตรงกับ Pine method updateVirtual — ครบทุก case"""
    if t.active:
        hit_stop = (l <= t.stop) if t.dir == 1 else (h >= t.stop)
        hit1     = (h >= t.t1)   if t.dir == 1 else (l <= t.t1)
        hit2     = (h >= t.t2)   if t.dir == 1 else (l <= t.t2)
        timeout  = (bar_idx - t.open_bar) >= MAX_BARS

        if hit_stop and hit2:
            # Collision → f_resolveCollision
            sl_first = abs(o - t.stop) < abs(o - t.t2)
            s.total += 1
            if sl_first:
                if tp1_first_heuristic(t.hit_tp1, t.dir, t.t1, hit1, c, o):
                    s.win1 += 1
                else:
                    s.loss += 1
            else:
                s.win1 += 1
                s.win2 += 1
            t.active = False

        elif hit_stop:
            s.total += 1
            if tp1_first_heuristic(t.hit_tp1, t.dir, t.t1, hit1, c, o):
                s.win1 += 1
            else:
                s.loss += 1
            t.active = False

        elif hit2:
            s.total += 1
            s.win1 += 1
            s.win2 += 1
            t.active = False

        elif timeout:
            s.total += 1
            if t.hit_tp1 or hit1:
                s.win1 += 1
            else:
                s.tOuts += 1
            t.active = False

        else:
            if hit1:
                t.hit_tp1 = True

    if not t.active:
        d = 1 if trig_l else (-1 if trig_s else 0)
        if d != 0:
            risk = atr_val * STOP_ATR
            t.dir, t.entry, t.open_bar = d, c, bar_idx
            t.stop = (c - risk) if d == 1 else (c + risk)
            t.t1   = (c + risk * R1) if d == 1 else (c - risk * R1)
            t.t2   = (c + risk * R2) if d == 1 else (c - risk * R2)
            t.hit_tp1, t.active = False, True


# ═════════════════════════════════════════════════════════════════════════════
#  HTF TREND ALIGNMENT  (ตรงกับ Pine Section 3)
# ═════════════════════════════════════════════════════════════════════════════
def get_htf_trend(data_b, data_h):
    """Align HTF EMA200 trend to base timeframe bars.
    Pine ใช้ request.security(..., [close[1], ema200[1]], lookahead_on)
    → ใช้ shift(1) แล้ว forward-fill ลง base bars"""
    n = len(data_b)
    if data_h is None or len(data_h) < 200:
        ema200 = calc_ema(data_b['close'], 200)
        return (data_b['close'] > ema200).values, (data_b['close'] < ema200).values

    htf_ema200 = calc_ema(data_h['close'], 200)
    # shift(1) = Pine's [1] offset → previous confirmed HTF bar
    bull_raw = data_h['close'].shift(1) > htf_ema200.shift(1)
    bear_raw = data_h['close'].shift(1) < htf_ema200.shift(1)

    bull = bull_raw.reindex(data_b.index, method='ffill')
    bear = bear_raw.reindex(data_b.index, method='ffill')

    # Fallback NaN → base EMA200
    base_ema = calc_ema(data_b['close'], 200)
    bull = bull.fillna(data_b['close'] > base_ema)
    bear = bear.fillna(data_b['close'] < base_ema)
    return bull.values.astype(bool), bear.values.astype(bool)


# ═════════════════════════════════════════════════════════════════════════════
#  STORM SIGNAL ENGINE  (ตรงกับ Pine Sections 2-5 ทั้งหมด)
# ═════════════════════════════════════════════════════════════════════════════
def compute_storm_signals(data_b, data_h):
    """
    คำนวณสัญญาณ Storm A+ ครบทุก component ตรงกับ Pine Script:
    Yang-Zhang Vol → Regime → HTF Trend → 5 Setups → Virtual Tracker → AI Optimizer
    Iterate ผ่านทุก bar เพื่อสร้าง stateful logic แล้ว return สัญญาณ bar สุดท้าย
    """
    df = data_b.copy()
    n = len(df)
    if n < 60:
        return {'signal': None}

    # ─── Vectorized Indicators ───────────────────────────────────────────
    df['atr14'] = calc_atr(df, 14)
    df['ema9']  = calc_ema(df['close'], 9)
    df['ema21'] = calc_ema(df['close'], 21)
    df['ema50'] = calc_ema(df['close'], 50)
    df['rsi']   = calc_rsi(df['close'], RSI_LEN)

    # Yang-Zhang Volatility
    df['volYZ']    = calc_yang_zhang_vol(df, VOL_LEN)
    df['volS']     = calc_yang_zhang_vol(df, 10)
    df['volL']     = calc_yang_zhang_vol(df, 60)
    df['tsRatio']  = np.where(df['volL'] > 0, df['volS'] / df['volL'], 1.0)
    df['volPct']   = calc_percentile_rank(df['volYZ'], 300)

    # VVol (Volatility-of-Volatility)
    vol_diff       = df['volYZ'] - df['volYZ'].shift(1)
    df['volDiffStd'] = vol_diff.rolling(20).std()
    df['vvol']     = np.where(df['volYZ'] > 0, df['volDiffStd'] / df['volYZ'], 0.0)
    df['vvolMa']   = pd.Series(df['vvol']).rolling(50).mean()

    # VWAP & Pivot
    df['vwap']      = calc_vwap(df)
    df['pivotHigh'] = calc_pivot_high(df['high'], 5, 5)
    df['pivotLow']  = calc_pivot_low(df['low'], 5, 5)

    # HTF Trend
    htf_bull_arr, htf_bear_arr = get_htf_trend(df, data_h)

    # ─── Extract numpy arrays for fast iteration ─────────────────────────
    close    = df['close'].values
    open_    = df['open'].values
    high     = df['high'].values
    low      = df['low'].values
    atr      = df['atr14'].values
    ema9     = df['ema9'].values
    ema21    = df['ema21'].values
    ema50    = df['ema50'].values
    rsi      = df['rsi'].values
    vol_pct  = df['volPct'].values
    ts_ratio = df['tsRatio'].values
    vvol_v   = df['vvol'].values
    vvol_ma  = df['vvolMa'].values
    vwap     = df['vwap'].values
    ph       = df['pivotHigh'].values
    pl       = df['pivotLow'].values

    # ─── Persistent state (= Pine var) ───────────────────────────────────
    regime   = 0
    cmp_dur  = 0

    below_run = 0
    above_run = 0
    bs_touch_up = 999
    bs_touch_dn = 999

    brt_hi_lvl = np.nan;  brt_hi_brk = False;  brt_hi_bar = -999
    brt_lo_lvl = np.nan;  brt_lo_brk = False;  brt_lo_bar = -999

    t_on  = VirtualTrade();  s_on  = VirtualStats()
    t_off = VirtualTrade();  s_off = VirtualStats()

    # Output for last bar
    sig_out = {'signal': None}

    # ─── Bar-by-bar iteration ────────────────────────────────────────────
    for i in range(n):
        ai = atr[i]
        if np.isnan(ai) or ai <= 0:
            continue

        # ── Compression Duration ──
        vp = vol_pct[i] if not np.isnan(vol_pct[i]) else 50.0
        cmp_dur = cmp_dur + 1 if vp <= 25.0 else 0

        # ── VVol Hot ──
        vvh = (not np.isnan(vvol_ma[i]) and vvol_ma[i] > 0
               and not np.isnan(vvol_v[i]) and vvol_v[i] > vvol_ma[i] * 1.3)

        # ── Storm Pressure ──
        p1 = max(0.0, (35.0 - vp) / 35.0) * 0.5
        p2 = min(cmp_dur / 40.0, 1.0) * 0.3
        p3 = 0.2 if vvh else 0.0
        press = min(1.0, p1 + p2 + p3) * 100.0

        # ── Regime State Machine [F1] ──
        tsr = ts_ratio[i] if not np.isnan(ts_ratio[i]) else 1.0
        if vp >= STORM_PCT or tsr >= 1.25:
            nr = 2
        elif press >= WATCH_TH * 0.7 and vp <= 35.0:
            nr = 1
        elif regime == 2 and vp < STORM_PCT:
            nr = 3
        elif regime == 3 and vp < 45.0 and tsr < 1.0:
            nr = 0
        elif regime == 3:
            nr = 3
        else:
            nr = 0
        regime = nr
        is_storm = (regime == 1 or regime == 2)

        # ── VWAP Runs ──
        prev_below, prev_above = below_run, above_run
        if not np.isnan(vwap[i]):
            below_run = below_run + 1 if close[i] < vwap[i] else 0
            above_run = above_run + 1 if close[i] > vwap[i] else 0

        # ── Setup 1: VWAP Crossover ──
        sVwapL = sVwapS = False
        if i > 0 and not np.isnan(vwap[i]) and not np.isnan(vwap[i - 1]):
            sVwapL = close[i] > vwap[i] and close[i - 1] <= vwap[i - 1] and prev_below >= 6
            sVwapS = close[i] < vwap[i] and close[i - 1] >= vwap[i - 1] and prev_above >= 6

        # ── Setup 2: EMA Bounce ──
        if not np.isnan(ema21[i]) and low[i] <= ema21[i]:
            bs_touch_up = 0
        else:
            bs_touch_up += 1
        if not np.isnan(ema21[i]) and high[i] >= ema21[i]:
            bs_touch_dn = 0
        else:
            bs_touch_dn += 1

        trendUp = (not np.isnan(ema21[i]) and not np.isnan(ema50[i])
                   and ema21[i] > ema50[i]
                   and i >= 3 and not np.isnan(ema50[i - 3]) and ema50[i] > ema50[i - 3])
        trendDn = (not np.isnan(ema21[i]) and not np.isnan(ema50[i])
                   and ema21[i] < ema50[i]
                   and i >= 3 and not np.isnan(ema50[i - 3]) and ema50[i] < ema50[i - 3])

        prevH = high[i - 1] if i > 0 else 0.0
        prevL = low[i - 1]  if i > 0 else float('inf')

        sEmaL = (trendUp and bs_touch_up <= 3
                 and not np.isnan(ema9[i]) and close[i] > ema9[i]
                 and close[i] > open_[i] and close[i] > prevH)
        sEmaS = (trendDn and bs_touch_dn <= 3
                 and not np.isnan(ema9[i]) and close[i] < ema9[i]
                 and close[i] < open_[i] and close[i] < prevL)

        # ── Setup 3: Breakout Retest ──
        if not np.isnan(ph[i]):
            brt_hi_lvl, brt_hi_brk = ph[i], False
        if not np.isnan(pl[i]):
            brt_lo_lvl, brt_lo_brk = pl[i], False

        if not np.isnan(brt_hi_lvl) and not brt_hi_brk and close[i] > brt_hi_lvl:
            brt_hi_brk, brt_hi_bar = True, i
        if not np.isnan(brt_lo_lvl) and not brt_lo_brk and close[i] < brt_lo_lvl:
            brt_lo_brk, brt_lo_bar = True, i

        sBrtL = (brt_hi_brk and i > brt_hi_bar and (i - brt_hi_bar) <= 20
                 and low[i] <= brt_hi_lvl + ai * 0.3
                 and close[i] > brt_hi_lvl and close[i] > open_[i])
        sBrtS = (brt_lo_brk and i > brt_lo_bar and (i - brt_lo_bar) <= 20
                 and high[i] >= brt_lo_lvl - ai * 0.3
                 and close[i] < brt_lo_lvl and close[i] < open_[i])

        if sBrtL:
            brt_hi_brk = False
        if sBrtS:
            brt_lo_brk = False

        # ── Setup 4: Liquidity Sweep ──
        sSweepL = sSweepS = False
        if i >= 21:
            swLo = np.min(low[i - 20:i])    # ta.lowest(low,20)[1]
            swHi = np.max(high[i - 20:i])   # ta.highest(high,20)[1]
            sSweepL = low[i] < swLo and close[i] > swLo and close[i] > open_[i]
            sSweepS = high[i] > swHi and close[i] < swHi and close[i] < open_[i]

        # ── Setup 5: ORB (skip by default) ──
        sOrbL = sOrbS = False
        # ORB logic เปิดใช้เมื่อ USE_ORB=True (สำหรับ Forex/Gold NY Session)

        # ── Raw & Filtered Triggers ──
        rawL = sVwapL or sEmaL or sBrtL or sSweepL or sOrbL
        rawS = sVwapS or sEmaS or sBrtS or sSweepS or sOrbS

        htf_bull = htf_bull_arr[i] if i < len(htf_bull_arr) else True
        htf_bear = htf_bear_arr[i] if i < len(htf_bear_arr) else False
        rv = rsi[i] if not np.isnan(rsi[i]) else 50.0

        aplus_l_on  = rawL and is_storm and htf_bull and rv <= RSI_MAX_BUY
        aplus_s_on  = rawS and is_storm and htf_bear and rv >= RSI_MIN_SELL
        aplus_l_off = rawL and is_storm and htf_bull
        aplus_s_off = rawS and is_storm and htf_bear

        # ── Virtual Tracker ──
        update_virtual_trade(t_on,  s_on,  aplus_l_on,  aplus_s_on,
                             i, open_[i], high[i], low[i], close[i], ai)
        update_virtual_trade(t_off, s_off, aplus_l_off, aplus_s_off,
                             i, open_[i], high[i], low[i], close[i], ai)

        # ── AI Optimizer [F2] ──
        wr_on  = (s_on.win1 / s_on.total * 100) if s_on.total > 0 else 0.0
        wr_off = (s_off.win1 / s_off.total * 100) if s_off.total > 0 else 0.0
        prefer_on = True
        if AUTO_RSI:
            if s_on.total >= 5 and s_off.total >= 5:
                if wr_off > wr_on:
                    prefer_on = False
                elif wr_off == wr_on and s_off.total > s_on.total:
                    prefer_on = False
            elif s_off.total >= 5 and s_on.total < 5:
                if wr_off >= 60.0:
                    prefer_on = False
            elif s_on.total >= 5 and s_off.total < 5:
                if wr_on < 40.0:
                    prefer_on = False

        a_plus_long  = aplus_l_on if prefer_on else aplus_l_off
        a_plus_short = aplus_s_on if prefer_on else aplus_s_off

        # ── Record setups ──
        setups = []
        if sVwapL or sVwapS: setups.append('VWAP')
        if sEmaL  or sEmaS:  setups.append('EMA')
        if sBrtL  or sBrtS:  setups.append('BRT')
        if sSweepL or sSweepS: setups.append('SWEEP')

        sig = 'BUY' if a_plus_long else ('SELL' if a_plus_short else None)

        # อัปเดต output (จะ overwrite ทุก bar → สุดท้ายคือ bar สุดท้าย)
        sig_out = {
            'signal':     sig,
            'setups':     setups if sig else [],
            'regime':     regime,
            'vol_pct':    round(vp, 1),
            'rsi':        round(rv, 1),
            'atr':        ai,
            'prefer_on':  prefer_on,
            'wr_on':      round(wr_on, 1),
            'wr_off':     round(wr_off, 1),
            'trades_on':  s_on.total,
            'trades_off': s_off.total,
            'htf_trend':  'BULL' if htf_bull else 'BEAR',
        }

    return sig_out


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE TRADE TRACKER  (แก้ไข: +collision, +timeout, +tp1First)
# ═════════════════════════════════════════════════════════════════════════════
async def check_active_trade_outcomes(confirmed, pair_name, tf_base):
    """ตรวจผลลัพธ์ trade ที่เปิดอยู่ ตาม Pine Section 6 logic"""
    global active_trades
    still_active = []
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for t in active_trades:
        # ข้าม trade ที่ไม่ใช่คู่/TF นี้
        if t['pair'] != pair_name or t['tf'] != tf_base:
            still_active.append(t)
            continue

        last_checked = pd.Timestamp(t.get('last_checked_bar', t['open_bar_time']))
        new_bars = confirmed[confirmed.index > last_checked]
        if len(new_bars) == 0:
            still_active.append(t)
            continue

        # นับ bars ทั้งหมดตั้งแต่เปิด trade
        open_ts = pd.Timestamp(t['open_bar_time'])
        total_bars_since_open = len(confirmed[confirmed.index > open_ts])
        already_checked = total_bars_since_open - len(new_bars)

        closed = False
        direction = 1 if t['side'] == 'BUY' else -1

        for bar_count, (idx, bar) in enumerate(new_bars.iterrows()):
            if closed:
                break
            h, l, o, c = bar['high'], bar['low'], bar['open'], bar['close']
            elapsed = already_checked + bar_count + 1

            hit_stop = (l <= t['sl'])  if direction == 1 else (h >= t['sl'])
            hit_tp1  = (h >= t['tp1']) if direction == 1 else (l <= t['tp1'])
            hit_tp2  = (h >= t['tp2']) if direction == 1 else (l <= t['tp2'])
            timeout  = elapsed >= MAX_BARS

            # ─ ลำดับเดียวกับ Pine: collision → stop → tp2 → timeout → continue ─
            if hit_stop and hit_tp2:
                sl_first = abs(o - t['sl']) < abs(o - t['tp2'])
                if sl_first:
                    tp1w = tp1_first_heuristic(
                        t.get('hit_tp1', False), direction, t['tp1'], hit_tp1, c, o)
                    status = 'WIN_TP1_SL' if tp1w else 'CLOSED_SL'
                    pnl = 0.8 if tp1w else -1.0
                else:
                    status, pnl = 'CLOSED_TP2', 2.0
                update_csv_trade_status(t['trade_id'], status, pnl, now_str)
                em = "🚀" if pnl > 0 else "🛑"
                await send_telegram(
                    f"{em} *[{status}]* `{t['pair']}` *[{t['tf']}]* ({t['side']}) | PnL: {pnl:+.1f}R")
                log_event(f"{em} {t['pair']} [{t['tf']}] {status} ({pnl:+.1f}R)")
                closed = True

            elif hit_stop:
                tp1w = tp1_first_heuristic(
                    t.get('hit_tp1', False), direction, t['tp1'], hit_tp1, c, o)
                status = 'WIN_TP1_SL' if tp1w else 'CLOSED_SL'
                pnl = 0.8 if tp1w else -1.0
                update_csv_trade_status(t['trade_id'], status, pnl, now_str)
                em = "🎯" if tp1w else "🛑"
                await send_telegram(
                    f"{em} *[{status}]* `{t['pair']}` *[{t['tf']}]* ({t['side']}) | PnL: {pnl:+.1f}R")
                log_event(f"{em} {t['pair']} [{t['tf']}] {status} ({pnl:+.1f}R)")
                closed = True

            elif hit_tp2:
                update_csv_trade_status(t['trade_id'], 'CLOSED_TP2', 2.0, now_str)
                await send_telegram(
                    f"🚀 *[TP2 FULL WIN]* `{t['pair']}` *[{t['tf']}]* ({t['side']}) | +2.0R ✅")
                log_event(f"🚀 {t['pair']} [{t['tf']}] TP2 (+2.0R)")
                closed = True

            elif timeout:
                tp1_in = t.get('hit_tp1', False) or hit_tp1
                status = 'TIMEOUT_TP1' if tp1_in else 'TIMEOUT'
                pnl = 0.8 if tp1_in else 0.0
                update_csv_trade_status(t['trade_id'], status, pnl, now_str)
                await send_telegram(
                    f"⏰ *[{status}]* `{t['pair']}` *[{t['tf']}]* ({t['side']}) | PnL: {pnl:+.1f}R")
                log_event(f"⏰ {t['pair']} [{t['tf']}] {status} ({pnl:+.1f}R)")
                closed = True

            else:
                if hit_tp1 and not t.get('hit_tp1', False):
                    t['hit_tp1'] = True
                    t['status'] = 'HIT_TP1'
                    update_csv_trade_status(t['trade_id'], 'HIT_TP1', 0.8, now_str)
                    await send_telegram(
                        f"🎯 *[TP1 HIT]* `{t['pair']}` *[{t['tf']}]* ({t['side']}) | +0.8R ✨")
                    log_event(f"🎯 {t['pair']} [{t['tf']}] TP1 (+0.8R)")

        t['last_checked_bar'] = str(new_bars.index[-1])
        if not closed:
            still_active.append(t)

    active_trades = still_active


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE SCANNER  (ใช้ compute_storm_signals แทน EMA cross เดิม)
# ═════════════════════════════════════════════════════════════════════════════
async def scan_live_pair(name, yf_ticker, tf_base, tf_htf, winrate):
    tracker_key = f"{name}_{tf_base}"
    try:
        bp = BASE_PERIOD.get(tf_base, '7d')
        hp = HTF_PERIOD.get(tf_htf, '60d')

        data_b = yf.download(yf_ticker, period=bp, interval=tf_base, progress=False)
        data_h = yf.download(yf_ticker, period=hp, interval=tf_htf, progress=False)

        # Clean MultiIndex columns (yfinance v0.2+)
        for d in (data_b, data_h):
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
        data_b.columns = [c.lower() for c in data_b.columns]
        data_h.columns = [c.lower() for c in data_h.columns]

        if len(data_b) < 60 or len(data_h) < 30:
            return

        # Normalize timezone
        if data_b.index.tz is not None:
            data_b.index = data_b.index.tz_convert('UTC').tz_localize(None)
        if data_h.index.tz is not None:
            data_h.index = data_h.index.tz_convert('UTC').tz_localize(None)

        # ใช้ confirmed bars only (ตัด bar สุดท้ายที่อาจยังไม่ปิด)
        confirmed = data_b.iloc[:-1]
        if len(confirmed) < 60:
            return

        # 1) ตรวจผลลัพธ์ trades ที่เปิดอยู่
        await check_active_trade_outcomes(confirmed, name, tf_base)

        # 2) คำนวณสัญญาณ Storm A+
        result = compute_storm_signals(confirmed, data_h)
        if result.get('signal') is None:
            return

        last_bar_time = pd.Timestamp(confirmed.index[-1])
        last_time = str(last_bar_time)
        if alerted_tracker.get(tracker_key) == last_time:
            return  # ส่งสัญญาณ bar นี้ไปแล้ว

        # ─── FRESHNESS GUARD: ป้องกันการแจ้งเตือนแท่งเก่าย้อนหลัง (เช่น วันหยุด/ตลาดปิด) ───
        # กำหนดอายุแท่งสูงสุดที่ยอมรับได้ (ไม่เกิน 2-3 แท่งย้อนหลัง)
        max_age_map = {'5m': 25 * 60, '15m': 60 * 60, '1h': 3 * 3600}
        max_allowed_age = max_age_map.get(tf_base, 3 * 3600)
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        bar_age_seconds = (now_utc - last_bar_time.to_pydatetime()).total_seconds()

        if bar_age_seconds > max_allowed_age:
            # แท่งเก่าเกินไป (ตลาดปิด เช่น Forex/ทองคำช่วงเสาร์-อาทิตย์ หรือบอทเพิ่ง Start)
            # บันทึกว่ารับทราบแล้ว แต่ไม่ส่ง Alert และไม่เปิดออเดอร์ใหม่
            alerted_tracker[tracker_key] = last_time
            return

        alerted_tracker[tracker_key] = last_time
        side  = result['signal']
        c     = confirmed['close'].iloc[-1]
        risk  = result['atr'] * STOP_ATR
        sl    = c - risk if side == 'BUY' else c + risk
        tp1   = (c + risk * R1) if side == 'BUY' else (c - risk * R1)
        tp2   = (c + risk * R2) if side == 'BUY' else (c - risk * R2)
        emoji = "🟢" if side == "BUY" else "🔴"
        trade_id = f"T_{int(datetime.now().timestamp())}_{name[:3]}"

        new_trade = {
            'trade_id': trade_id, 'timestamp': last_time,
            'pair': name, 'tf': tf_base, 'side': side,
            'setup': ', '.join(result['setups']) or 'Storm A+',
            'entry': round(c, 5), 'sl': round(sl, 5),
            'tp1': round(tp1, 5), 'tp2': round(tp2, 5),
            'status': 'OPEN', 'pnl_r': 0.0, 'close_time': '',
            # runtime-only fields (not in CSV)
            'open_bar_time': last_time,
            'hit_tp1': False,
            'last_checked_bar': last_time,
        }
        log_trade_to_csv(new_trade)
        active_trades.append(new_trade)

        msg = (
            f"⚡ *[STORM A+ {side} SIGNAL]* ⚡\n\n"
            f"📊 *Asset:* `{name}` *(TF: {tf_base})*\n"
            f"🔥 *Setups:* {', '.join(result['setups'])}\n"
            f"🌪️ *Regime:* {result['regime']} | *Vol%:* {result['vol_pct']}\n"
            f"📈 *HTF:* {result['htf_trend']} | *RSI:* {result['rsi']}\n"
            f"🧠 *AI:* {'RSI ON' if result['prefer_on'] else 'RSI OFF'} "
            f"(On:{result['wr_on']}%/{result['trades_on']}t "
            f"Off:{result['wr_off']}%/{result['trades_off']}t)\n"
            f"⭐ *Historical WR:* `{winrate:.1f}%`\n"
            f"🎯 *Direction:* {emoji} *{side}*\n"
            f"🆔 *ID:* `{trade_id}`\n\n"
            f"💵 *Entry:* `{c:.5f}`\n"
            f"🛑 *SL:* `{sl:.5f}`\n"
            f"🎯 *TP1 ({R1}R):* `{tp1:.5f}`\n"
            f"🚀 *TP2 ({R2}R):* `{tp2:.5f}`\n"
            f"⏰ *Bar:* `{last_time}`\n"
            f"💾 *Logged:* ✅ `{LOG_FILE_PATH}`"
        )
        log_event(f"🚨 {name} [{tf_base}] {side} | {result['setups']} | "
                  f"Regime:{result['regime']} Vol%:{result['vol_pct']} "
                  f"HTF:{result['htf_trend']} RSI:{result['rsi']}")
        await send_telegram(msg)

    except Exception as e:
        log_event(f"⚠️ {name} [{tf_base}] Error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  DASHBOARD WEB PAGE  (เดิม + เพิ่ม regime/AI info)
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
                    <td>{r.get('setup','')}</td>
                    <td>{r['entry']}</td>
                    <td>{r['sl']}</td>
                    <td>{r['tp1']}</td>
                    <td><span style="background:#334155; padding:2px 8px;
                        border-radius:4px;">{r['status']}</span></td>
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
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                   Roboto, sans-serif; background: #0b0f19; color: #f1f5f9;
                   margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ display: flex; justify-content: space-between;
                      align-items: center; background: #1e293b;
                      padding: 20px 25px; border-radius: 12px;
                      border: 1px solid #334155; margin-bottom: 20px; }}
            .title {{ font-size: 22px; font-weight: bold; color: #38bdf8;
                     display: flex; align-items: center; gap: 10px; }}
            .badge-live {{ background: #10b981; color: white;
                          padding: 4px 12px; border-radius: 20px;
                          font-size: 13px; font-weight: bold; }}
            .card {{ background: #1e293b; padding: 20px; border-radius: 12px;
                    border: 1px solid #334155; margin-bottom: 20px; }}
            .card-title {{ font-size: 17px; font-weight: bold;
                          margin-bottom: 12px; color: #94a3b8; }}
            table {{ width: 100%; border-collapse: collapse;
                    text-align: left; font-size: 14px; }}
            th {{ background: #0f172a; padding: 12px; color: #94a3b8;
                 font-weight: 600; border-bottom: 2px solid #334155; }}
            td {{ padding: 12px; border-bottom: 1px solid #334155; }}
            .log-terminal {{ background: #000; color: #22c55e;
                            font-family: 'Courier New', monospace;
                            font-size: 13px; padding: 15px; border-radius: 8px;
                            height: 180px; overflow-y: auto; white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">⚡ Storm A+ Live Scanner (Full Pine Logic)</div>
                <div><span class="badge-live">● RUNNING ({len(A_PLUS_WATCHLIST)} A+ pairs)</span></div>
            </div>
            <div class="card">
                <div class="card-title">📡 Real-Time Logs</div>
                <div class="log-terminal">{logs_text if logs_text else
                    'ระบบกำลังเฝ้ากราฟ A+ pairs...'}</div>
            </div>
            <div class="card">
                <div class="card-title">📝 รายการออเดอร์ล่าสุด</div>
                <table>
                    <thead><tr>
                        <th>เวลา</th><th>คู่เงิน</th><th>TF</th>
                        <th>Side</th><th>Setup</th><th>Entry</th>
                        <th>SL</th><th>TP1</th><th>สถานะ</th>
                    </tr></thead>
                    <tbody>
                        {rows_html if rows_html else
                         '<tr><td colspan="9" style="text-align:center; '
                         'color:#64748b;">ยังไม่มีสัญญาณ</td></tr>'}
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
    log_event(f"Web Server listening on port {port} (Full Pine Logic Edition)")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════
async def scanner_worker():
    await asyncio.sleep(2)
    log_event(f"🚀 Storm A+ Scanner started — {len(A_PLUS_WATCHLIST)} pairs "
              f"(Full Pine Logic: YZ-Vol + Regime + HTF + 5 Setups + AI Optimizer)")

    summary_msg = (
        f"🏆 *[STORM A+ LIVE SCANNER STARTED]* 🏆\n"
        f"ระบบ Full Pine Logic Edition เริ่มทำงานแล้ว!\n\n"
        f"🌪️ *Engine:* Yang-Zhang Vol → Regime → HTF Trend → "
        f"5 Setups → Virtual Tracker → AI Optimizer\n"
        f"📊 *Pairs:* {len(A_PLUS_WATCHLIST)} A+ (WR >= 55%)\n"
        f"⚙️ *Config:* Storm%={STORM_PCT} StopATR={STOP_ATR} "
        f"TP1={R1}R TP2={R2}R Timeout={MAX_BARS}bars\n\n"
        f"🌐 *Dashboard:* https://stormscanner.onrender.com"
    )
    await send_telegram(summary_msg)

    while True:
        tasks = [scan_live_pair(name, ticker, tf_b, tf_h, wr)
                 for name, ticker, tf_b, tf_h, wr in A_PLUS_WATCHLIST]
        await asyncio.gather(*tasks)
        await asyncio.sleep(30)


async def main():
    await asyncio.gather(start_web_server(), scanner_worker())


if __name__ == '__main__':
    asyncio.run(main())
