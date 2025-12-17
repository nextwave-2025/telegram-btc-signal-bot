import os
import time
import ccxt
import numpy as np
import pandas as pd
import urllib.parse
import urllib.request

from dataclasses import dataclass
from typing import Dict, Tuple
from zoneinfo import ZoneInfo


# =========================================================
# CONFIG
# =========================================================

TIMEFRAME_ENTRY = "15m"
TIMEFRAME_BIAS = "1h"
TIMEFRAME_CTX_4H = "4h"
TIMEFRAME_CTX_D1 = "1d"

EMA_FAST = 20
EMA_SLOW = 50

RSI_LOWER = 34
RSI_UPPER = 64

# RSI Extremes for SCALPING
RSI_SCALP_LONG = 28
RSI_SCALP_SHORT = 72

VOL_MA_LEN = 20
VOL_RATIO = 0.10

MAX_SL_PCT = 0.02
SCALP_SL_PCT = 0.009

ENTRY_PAD_PCT = 0.0006
RR_TARGETS = (1, 2, 3)

COOLDOWN_MINUTES = 45
SCALP_COOLDOWN_MINUTES = 90

LOCAL_TZ = "Europe/Berlin"

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "SUI/USDT",
    "DOGE/USDT",
    "XRP/USDT",
    "HYPE/USDT",
    "FARTCOIN/USDT",
]


# =========================================================
# FOOTER
# =========================================================

FOOTER_TEXT = (
    "<b>©️ Copyright by crypto_mistik</b>\n"
    "⚠️ Kein Financial Advice – Krypto ist hochriskant\n"
)


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class Bias:
    direction: str


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }).encode()
    urllib.request.urlopen(url, data=data, timeout=10)


# =========================================================
# HELPERS
# =========================================================

def to_bybit(sym): return sym if ":" in sym else f"{sym}:USDT"

def local_time(ts):
    return ts.to_pydatetime().astimezone(
        ZoneInfo(LOCAL_TZ)
    ).strftime("%Y-%m-%d %H:%M:%S %Z")

def ema(series, n): return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def minutes_between(a, b):
    return abs((a.to_pydatetime() - b.to_pydatetime()).total_seconds()) / 60


# =========================================================
# STRATEGY CORE
# =========================================================

def compute_bias(df):
    e20 = ema(df["close"], EMA_FAST).iloc[-1]
    e50 = ema(df["close"], EMA_SLOW).iloc[-1]
    if e20 > e50: return Bias("LONG")
    if e20 < e50: return Bias("SHORT")
    return Bias("NEUTRAL")


def entry_scalp_rsi(df15) -> Tuple[bool, str, Dict]:
    close = df15["close"]
    low = df15["low"]
    high = df15["high"]
    vol = df15["volume"]

    r = rsi(close).iloc[-1]
    c0 = close.iloc[-1]
    c1 = close.iloc[-2]

    v = vol.iloc[-1]
    vma = vol.rolling(VOL_MA_LEN).mean().iloc[-1]
    if vma <= 0 or v < vma * VOL_RATIO:
        return False, "", {}

    # LONG SCALP
    if r <= RSI_SCALP_LONG and c0 > low.iloc[-2]:
        return True, "SCALP_LONG", {"close": c0, "rsi": r}

    # SHORT SCALP
    if r >= RSI_SCALP_SHORT and c0 < high.iloc[-2]:
        return True, "SCALP_SHORT", {"close": c0, "rsi": r}

    return False, "", {}


# =========================================================
# TRADE PLAN
# =========================================================

def scalp_trade_plan(side, price):
    if side == "SCALP_LONG":
        sl = price * (1 - SCALP_SL_PCT)
        tp = price + (price - sl) * 1.2
    else:
        sl = price * (1 + SCALP_SL_PCT)
        tp = price - (sl - price) * 1.2
    return sl, tp


# =========================================================
# EXCHANGE
# =========================================================

def fetch_df(ex, sym, tf, limit=300):
    data = ex.fetch_ohlcv(sym, tf, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df


# =========================================================
# MAIN
# =========================================================

def main():
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    symbols = [to_bybit(s) for s in SYMBOLS]
    ex.load_markets()

    last_seen = {}
    last_scalp = {}

    print("✅ BOT AKTIV – Trend + RSI-Scalping", flush=True)

    while True:
        try:
            for sym in symbols:
                df15_raw = fetch_df(ex, sym, TIMEFRAME_ENTRY)
                df15 = df15_raw.iloc[:-1]
                ts = df15.index[-1]

                if last_seen.get(sym) == ts:
                    continue
                last_seen[sym] = ts

                ok, scalp_type, info = entry_scalp_rsi(df15)
                if not ok:
                    continue

                key = f"{sym}|{scalp_type}"
                if key in last_scalp and minutes_between(ts, last_scalp[key]) < SCALP_COOLDOWN_MINUTES:
                    continue
                last_scalp[key] = ts

                sl, tp = scalp_trade_plan(scalp_type, info["close"])

                head = "🟢 RSI SCALP LONG" if scalp_type == "SCALP_LONG" else "🔴 RSI SCALP SHORT"

                msg = (
                    f"{head}\n\n"
                    f"📊 Pair: {sym.replace(':USDT','').replace('/','')}\n"
                    f"🕒 15m Close: {local_time(ts)}\n"
                    f"📍 RSI: {info['rsi']:.2f}\n\n"
                    f"🎯 Entry: {info['close']:.4f}\n"
                    f"🛑 SL: {sl:.4f}\n"
                    f"✅ TP: {tp:.4f}\n\n"
                    f"{FOOTER_TEXT}"
                )

                print(msg, flush=True)
                send_telegram(msg)

            time.sleep(5)

        except Exception as e:
            print("ERROR:", e, flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
