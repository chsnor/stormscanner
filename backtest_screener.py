import os
import json
import urllib.request
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG & CREDENTIALS
# ═════════════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8907507767:AAEv4LMGq7vB9Zm76jaLbw3Iy5T66wJxtKw")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "5861943388")
WINRATE_THRESHOLD  = 55.0
MIN_TRADES         = 3

# Strategy Parameters (ตรงกับ Pine Script เป๊ะ)
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

# รายชื่อ 29 สินทรัพย์ทั้งหมด
YF_SYMBOLS_MAP = {
    # Metals & Crypto
    'XAU/USD (Gold)': 'GC=F',
    'XAG/USD (Silver)': 'SI=F',
    'BTC/USD': 'BTC-USD',
    'ETH/USD': 'ETH-USD',

    # Forex Majors
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X',
    'USD/CHF': 'USDCHF=X',
    'USD/CAD': 'USDCAD=X',
    'AUD/USD': 'AUDUSD=X',
    'NZD/USD': 'NZDUSD=X',

    # Forex Crosses (JPY)
    'EUR/JPY': 'EURJPY=X',
    'GBP/JPY': 'GBPJPY=X',
    'AUD/JPY': 'AUDJPY=X',
    'CAD/JPY': 'CADJPY=X',
    'CHF/JPY': 'CHFJPY=X',
    'NZD/JPY': 'NZDJPY=X',

    # Forex Crosses (EUR & GBP)
    'EUR/GBP': 'EURGBP=X',
    'EUR/AUD': 'EURAUD=X',
    'EUR/CAD': 'EURCAD=X',
    'EUR/CHF': 'EURCHF=X',
    'EUR/NZD': 'EURNZD=X',
    'GBP/AUD': 'GBPAUD=X',
    'GBP/CAD': 'GBPCAD=X',
    'GBP/CHF': 'GBPCHF=X',
    'GBP/NZD': 'GBPNZD=X',

    # Forex Crosses (AUD)
    'AUD/CAD': 'AUDCAD=X',
    'AUD/CHF': 'AUDCHF=X',
    'AUD/NZD': 'AUDNZD=X',
}

TF_CONFIGS = [
    ('5m',  '15m', '30d',  '30d'),
    ('15m', '1h',  '60d',  '1y'),
    ('1h',  '1d',  '730d', '2y')
]

# ═════════════════════════════════════════════════════════════════════════════
#  INDICATORS (EXACT PINE SCRIPT MATH)
# ═════════════════════════════════════════════════════════════════════════════
def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def calc_atr(df, length=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()

def calc_yang_zhang_vol(df, n):
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

def calc_percentile_rank(series, length=300):
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
    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        return hlc3
    volume = df['volume'].clip(lower=1e-10)
    dates = df.index.date
    vwap = pd.Series(index=df.index, dtype=float)
    for date in np.unique(dates):
        mask = dates == date
        vol = volume[mask]
        cum_vol = vol.cumsum()
        vwap[mask] = (hlc3[mask] * vol).cumsum() / cum_vol
    return vwap

def calc_pivot_high(series, left=5, right=5):
    result = pd.Series(np.nan, index=series.index)
    for i in range(left + right, len(series)):
        pivot_idx = i - right
        pv = series.iloc[pivot_idx]
        lw = series.iloc[pivot_idx - left:pivot_idx]
        rw = series.iloc[pivot_idx + 1:i + 1]
        if len(lw) >= left and len(rw) >= right and (pv >= lw).all() and (pv >= rw).all():
            result.iloc[i] = pv
    return result

def calc_pivot_low(series, left=5, right=5):
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
    bar_bullish = bar_close > bar_open
    gap_open_past = (bar_open >= t1_level) if direction == 1 else (bar_open <= t1_level)
    body_match = hit1 and (bar_bullish if direction == 1 else not bar_bullish)
    return was_hit_tp1 or gap_open_past or body_match

# ═════════════════════════════════════════════════════════════════════════════
#  VIRTUAL TRACKER (FOR AI OPTIMIZER)
# ═════════════════════════════════════════════════════════════════════════════
class VirtualStats:
    def __init__(self):
        self.total = 0
        self.win1  = 0
        self.win2  = 0
        self.loss  = 0
        self.tOuts = 0

class VirtualTrade:
    def __init__(self):
        self.dir      = 0
        self.entry    = 0.0
        self.stop     = 0.0
        self.t1       = 0.0
        self.t2       = 0.0
        self.open_bar = 0
        self.active   = False
        self.hit_tp1  = False

def update_virtual_trade(t, s, trig_l, trig_s, bar_idx, o, h, l, c, atr_val):
    if t.active:
        hit_stop = (l <= t.stop) if t.dir == 1 else (h >= t.stop)
        hit1     = (h >= t.t1)   if t.dir == 1 else (l <= t.t1)
        hit2     = (h >= t.t2)   if t.dir == 1 else (l <= t.t2)
        timeout  = (bar_idx - t.open_bar) >= MAX_BARS

        if hit_stop and hit2:
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
#  FULL PINE SCRIPT BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════
def run_backtest_simulation(df_base, df_htf):
    if len(df_base) < 60 or len(df_htf) < 30:
        return {'total': 0, 'win1': 0, 'win2': 0, 'loss': 0, 'timeouts': 0, 'wr1': 0.0, 'wr2': 0.0}

    df = df_base.copy()
    n = len(df)

    # 1. Base Indicators
    df['atr14'] = calc_atr(df, 14)
    df['ema9']  = calc_ema(df['close'], 9)
    df['ema21'] = calc_ema(df['close'], 21)
    df['ema50'] = calc_ema(df['close'], 50)
    df['rsi']   = calc_rsi(df['close'], RSI_LEN)

    # 2. Yang-Zhang Volatility Engine
    df['volYZ']   = calc_yang_zhang_vol(df, VOL_LEN)
    df['volS']    = calc_yang_zhang_vol(df, 10)
    df['volL']    = calc_yang_zhang_vol(df, 60)
    df['tsRatio'] = np.where(df['volL'] > 0, df['volS'] / df['volL'], 1.0)
    df['volPct']  = calc_percentile_rank(df['volYZ'], 300)

    vol_diff = df['volYZ'] - df['volYZ'].shift(1)
    df['volDiffStd'] = vol_diff.rolling(20).std()
    df['vvol']       = np.where(df['volYZ'] > 0, df['volDiffStd'] / df['volYZ'], 0.0)
    df['vvolMa']     = pd.Series(df['vvol']).rolling(50).mean()

    # 3. Setups & VWAP
    df['vwap']      = calc_vwap(df)
    df['pivotHigh'] = calc_pivot_high(df['high'], 5, 5)
    df['pivotLow']  = calc_pivot_low(df['low'], 5, 5)

    # 4. HTF Trend (EMA200)
    if len(df_htf) >= 200:
        htf_ema200 = calc_ema(df_htf['close'], 200)
        htf_bull_raw = df_htf['close'].shift(1) > htf_ema200.shift(1)
        htf_bear_raw = df_htf['close'].shift(1) < htf_ema200.shift(1)
        bull_s = htf_bull_raw.reindex(df.index, method='ffill')
        bear_s = htf_bear_raw.reindex(df.index, method='ffill')
        base_ema = calc_ema(df['close'], 200)
        htf_bull_arr = bull_s.fillna(df['close'] > base_ema).values.astype(bool)
        htf_bear_arr = bear_s.fillna(df['close'] < base_ema).values.astype(bool)
    else:
        base_ema = calc_ema(df['close'], 200)
        htf_bull_arr = (df['close'] > base_ema).values.astype(bool)
        htf_bear_arr = (df['close'] < base_ema).values.astype(bool)

    # Convert to numpy arrays for fast execution
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

    # Persistent States (Exact Pine Var)
    regime = 0
    cmp_dur = 0
    below_run = 0
    above_run = 0
    bs_touch_up = 999
    bs_touch_dn = 999
    brt_hi_lvl = np.nan; brt_hi_brk = False; brt_hi_bar = -999
    brt_lo_lvl = np.nan; brt_lo_brk = False; brt_lo_bar = -999

    # Virtual Trackers (AI Optimizer)
    t_on  = VirtualTrade(); s_on  = VirtualStats()
    t_off = VirtualTrade(); s_off = VirtualStats()

    # Main Strategy Backtest Tracker
    total_trades, win_tp1, win_tp2, losses, timeouts = 0, 0, 0, 0, 0
    active = False
    cur_dir, cur_stop, cur_t1, cur_t2, cur_open_bar, cur_hit_tp1 = 0, 0.0, 0.0, 0.0, 0, False

    for i in range(n):
        ai = atr[i]
        if np.isnan(ai) or ai <= 0:
            continue

        # ── Compression Duration & VVol ──
        vp = vol_pct[i] if not np.isnan(vol_pct[i]) else 50.0
        cmp_dur = cmp_dur + 1 if vp <= 25.0 else 0
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

        # ── Setup 1: VWAP Cross ──
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
            swLo = np.min(low[i - 20:i])
            swHi = np.max(high[i - 20:i])
            sSweepL = low[i] < swLo and close[i] > swLo and close[i] > open_[i]
            sSweepS = high[i] > swHi and close[i] < swHi and close[i] < open_[i]

        # ── Triggers ──
        rawL = sVwapL or sEmaL or sBrtL or sSweepL
        rawS = sVwapS or sEmaS or sBrtS or sSweepS

        htf_bull = htf_bull_arr[i] if i < len(htf_bull_arr) else True
        htf_bear = htf_bear_arr[i] if i < len(htf_bear_arr) else False
        rv = rsi[i] if not np.isnan(rsi[i]) else 50.0

        aplus_l_on  = rawL and is_storm and htf_bull and (rv <= RSI_MAX_BUY)
        aplus_s_on  = rawS and is_storm and htf_bear and (rv >= RSI_MIN_SELL)
        aplus_l_off = rawL and is_storm and htf_bull
        aplus_s_off = rawS and is_storm and htf_bear

        # Update Virtual Trackers
        update_virtual_trade(t_on,  s_on,  aplus_l_on,  aplus_s_on,  i, open_[i], high[i], low[i], close[i], ai)
        update_virtual_trade(t_off, s_off, aplus_l_off, aplus_s_off, i, open_[i], high[i], low[i], close[i], ai)

        # AI Optimizer Decision [F2]
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

        # ── Main Strategy Simulation (Exact Pine Section 6) ──
        if active:
            hit_stop = (low[i] <= cur_stop) if cur_dir == 1 else (high[i] >= cur_stop)
            hit1     = (high[i] >= cur_t1)  if cur_dir == 1 else (low[i] <= cur_t1)
            hit2     = (high[i] >= cur_t2)  if cur_dir == 1 else (low[i] <= cur_t2)
            timeout  = (i - cur_open_bar) >= MAX_BARS

            if hit_stop and hit2:
                sl_first = abs(open_[i] - cur_stop) < abs(open_[i] - cur_t2)
                total_trades += 1
                if sl_first:
                    if tp1_first_heuristic(cur_hit_tp1, cur_dir, cur_t1, hit1, close[i], open_[i]):
                        win_tp1 += 1
                    else:
                        losses += 1
                else:
                    win_tp1 += 1
                    win_tp2 += 1
                active = False
            elif hit_stop:
                total_trades += 1
                if tp1_first_heuristic(cur_hit_tp1, cur_dir, cur_t1, hit1, close[i], open_[i]):
                    win_tp1 += 1
                else:
                    losses += 1
                active = False
            elif hit2:
                total_trades += 1
                win_tp1 += 1
                win_tp2 += 1
                active = False
            elif timeout:
                total_trades += 1
                if cur_hit_tp1 or hit1:
                    win_tp1 += 1
                else:
                    timeouts += 1
                active = False
            else:
                if hit1:
                    cur_hit_tp1 = True

        if not active:
            trigger_dir = 1 if a_plus_long else (-1 if a_plus_short else 0)
            if trigger_dir != 0:
                risk = ai * STOP_ATR
                cur_dir = trigger_dir
                cur_open_bar = i
                cur_stop = (close[i] - risk) if trigger_dir == 1 else (close[i] + risk)
                cur_t1   = (close[i] + risk * R1) if trigger_dir == 1 else (close[i] - risk * R1)
                cur_t2   = (close[i] + risk * R2) if trigger_dir == 1 else (close[i] - risk * R2)
                cur_hit_tp1 = False
                active = True

    wr1 = (win_tp1 / total_trades * 100.0) if total_trades > 0 else 0.0
    wr2 = (win_tp2 / total_trades * 100.0) if total_trades > 0 else 0.0
    return {
        'total': total_trades,
        'win1': win_tp1,
        'win2': win_tp2,
        'loss': losses,
        'timeouts': timeouts,
        'wr1': wr1,
        'wr2': wr2
    }

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ No Telegram token/chat_id provided.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10):
            print("📨 ส่งรายงานผลเข้า Telegram สำเร็จเรียบร้อยแล้ว!")
    except Exception as e:
        print(f"Telegram error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"🚀 เริ่มต้นการ Backtest คัดกรองคู่เงิน (Full Pine Script Logic Engine)")
    print(f"📌 29 สินทรัพย์ × 3 Timeframes = 87 รูปแบบ | เกณฑ์ Win Rate >= {WINRATE_THRESHOLD}% (ขั้นต่ำ {MIN_TRADES} trades)\n")

    results_table = []
    qualified_list = []
    total_count = len(YF_SYMBOLS_MAP) * len(TF_CONFIGS)
    current_idx = 0

    for name, yf_ticker in YF_SYMBOLS_MAP.items():
        for tf_base, tf_htf, p_base, p_htf in TF_CONFIGS:
            current_idx += 1
            try:
                print(f"[{current_idx:>2}/{total_count}] {name:<18} [{tf_base:>3}] ...", end=" ", flush=True)
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
                passed = (stats['wr1'] >= WINRATE_THRESHOLD) and (stats['total'] >= MIN_TRADES)

                status_str = f"✅ ผ่าน ({stats['win1']}/{stats['total']})" if passed else "❌ ไม่ผ่าน"
                print(f"WR: {stats['wr1']:>5.1f}% (TP2: {stats['wr2']:>5.1f}%) | Trades: {stats['total']:>2} -> {status_str}", flush=True)

                results_table.append({
                    'Pair': name,
                    'Ticker': yf_ticker,
                    'TF': tf_base,
                    'TF_HTF': tf_htf,
                    'Total': stats['total'],
                    'Wins': stats['win1'],
                    'Win_TP2': stats['win2'],
                    'Losses': stats['loss'],
                    'Timeouts': stats['timeouts'],
                    'WinRate_TP1': stats['wr1'],
                    'WinRate_TP2': stats['wr2'],
                    'Passed': passed
                })

                if passed:
                    qualified_list.append((name, yf_ticker, tf_base, tf_htf, stats['wr1'], stats['total'], stats['win1'], stats['loss']))

            except Exception as e:
                print(f"Error: {e}")

    # แสดงตารางสรุป
    if results_table:
        df_res = pd.DataFrame(results_table).sort_values(by=['Passed', 'WinRate_TP1', 'Total'], ascending=[False, False, False])
        print("\n" + "="*95)
        print(f"📊 ตารางอันดับ WINRATE สูงสุด (จาก 87 รูปแบบ)")
        print("="*95)
        for idx, r in df_res.iterrows():
            badge = "⭐ [ผ่านเกณฑ์]" if r['Passed'] else "   [ไม่ผ่าน] "
            print(f"{badge} {r['Pair']:<18} ({r['TF']:>3}) | WR1: {r['WinRate_TP1']:>5.1f}% | WR2: {r['WinRate_TP2']:>5.1f}% | Trades: {r['Total']:>2} (W1:{r['Wins']}, W2:{r['Win_TP2']}, L:{r['Losses']}, TO:{r['Timeouts']})")

    # จัดเรียงตาม Win Rate สูงสุด
    qualified_list_sorted = sorted(qualified_list, key=lambda x: (x[4], x[5]), reverse=True)

    print("\n" + "="*95)
    print(f"🏆 คู่เงินและ Timeframe ที่ผ่านเกณฑ์ Win Rate >= {WINRATE_THRESHOLD}% มีทั้งหมด {len(qualified_list_sorted)} รายการ:")
    for name, ticker, tf, tf_h, wr, tot, w, l in qualified_list_sorted:
        print(f"   ⭐ ('{name}', '{ticker}', '{tf}', '{tf_h}', {wr:.1f}),  # {tot} ไม้ (W:{w} L:{l})")
    print("="*95)

    # ส่งสรุปเข้า Telegram
    tg_report = f"📊 *[STORM A+ WINRATE RECOMMENDATION (PINE ACCURATE)]*\n"
    tg_report += f"🎯 *เกณฑ์คัดเลือก:* Win Rate >= {WINRATE_THRESHOLD}% (ขั้นต่ำ {MIN_TRADES} ไม้)\n"
    tg_report += f"⏱ *Timeframes ทดสอบ:* `5m, 15m, 1h`\n"
    tg_report += f"🌪️ *Engine:* Full Yang-Zhang + Regime + HTF EMA200 + 5 Setups + AI Optimizer\n\n"
    tg_report += f"🏆 *รายชื่อคู่เงินเกรด A+ ที่ผ่านเกณฑ์ ({len(qualified_list_sorted)} รายการ):*\n\n"

    if qualified_list_sorted:
        for name, ticker, tf, tf_h, wr, tot, w, l in qualified_list_sorted:
            tg_report += f"• `{name}` *[{tf}]* ➡️ Win Rate: *{wr:.1f}%* (ชนะ {w}/{tot} ไม้)\n"
    else:
        tg_report += "• ไม่มีคู่ที่ผ่านเกณฑ์ในช่วงเวลานี้\n"

    tg_report += f"\n⏰ *รายงาน ณ เวลา:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"

    print("\nกำลังส่งรายงานเข้า Telegram...")
    send_telegram(tg_report)

    # บันทึกผลลัพธ์ลงไฟล์ json สำหรับนำไปใช้อัปเดต A_PLUS_WATCHLIST
    output_watchlist = [
        (name, ticker, tf, tf_h, round(wr, 1))
        for name, ticker, tf, tf_h, wr, tot, w, l in qualified_list_sorted
    ]
    with open("scanned_watchlist.json", "w", encoding="utf-8") as f:
        json.dump(output_watchlist, f, ensure_ascii=False, indent=4)
    print("\n💾 บันทึกผลลัพธ์ลง `scanned_watchlist.json` เรียบร้อยแล้ว!")
