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

nest_asyncio.apply()

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG & CREDENTIALS
# ═════════════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN = "8907507767:AAEv4LMGq7vB9Zm76jaLbw3Iy5T66wJxtKw"
TELEGRAM_CHAT_ID   = "5861943388"
LOG_FILE_PATH      = "storm_trade_logs.csv"

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
    ('5m',  '15m', '60d',  '60d'),
    ('15m', '1h',  '60d',  '1y'),
    ('1h',  '1d',  '730d', '2y')
]

WINRATE_THRESHOLD = 55.0

alerted_tracker = {}
active_trades   = []

# ═════════════════════════════════════════════════════════════════════════════
#  HELPER INDICATORS (PURE PANDAS / NUMPY)
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
#  CSV LOGGER
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
            print(f"Telegram error: {e}")

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
        v_on = df['ln_oc1'].
