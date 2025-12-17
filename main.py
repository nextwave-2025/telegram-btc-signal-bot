import os
import time
import ccxt
import numpy as np
import pandas as pd
import urllib.parse
import urllib.request

from typing import Dict, Tuple
from dataclasses import dataclass
from zoneinfo import ZoneInfo


# =========================================================
# CONFIG
# =========================================================

TIMEFRAME_ENTRY = "15m"
TIMEFRAME_BIAS = "1h"       # ✅ Daytrading-Bias
TIMEFRAME_CTX_4H = "4h"
TIMEFRAME_CTX_D1 = "1d"

EMA_FAST = 20
EMA_SLOW = 50

RSI_LOWER = 34
RSI_UPPER = 64

VOL_MA_LEN = 20
VOL_RATIO = 0.12

MAX_SL_PCT = 0.02
ENTRY_PAD_PCT = 0.0006
RR_TARGETS = (1, 2, 3)

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
# DATA STRUCTURES
# =========================================================

@dataclass
class Bias:
    direction: str   # LONG / SHORT / NEUTRAL


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ Telegram OFF (BOT_TOKEN / CHAT_ID missing)", flush=True)
        return

    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception as e:
        print(f"⚠️ Telegram error: {e}", flush=True)


# =========================================================
# HELPERS
# =========================================================

def to_bybit(sym: str) -> str:
    return sym if ":" in sym else f"{sym}:USDT"

def local_time(ts) -> str:
    return ts.to_pydatetime().astimezone(
        ZoneInfo(LOCAL_TZ)
    ).strftime("%Y-%m-%d %H:%M:%S %Z")

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


# =========================================================
# STRATEGY CORE
# =========================================================

def compute_bias(df) -> Bias:
    e20 = ema(df["close"], EMA_FAST).iloc[-1]
    e50 = ema(df["close"], EMA_SLOW).iloc[-1]
    if e20 > e50:
        return Bias("LONG")
    if e20 < e50:
        return Bias("SHORT")
    return Bias("NEUTRAL")

def check_entry(df15, bias_dir) -> Tuple[bool, Dict]:
    close = df15["close"]
    vol = df15["volume"]

    e20 = ema(close, EMA_FAST).iloc[-1]
    e50 = ema(close, EMA_SLOW).iloc[-1]
    r = rsi(close).iloc[-1]

    v = vol.iloc[-1]
    vma = vol.rolling(VOL_MA_LEN).mean().iloc[-1]
    vol_ratio = v / vma if vma > 0 else 0

    info = {
        "close": float(close.iloc[-1]),
        "ema20": float(e20),
        "ema50": float(e50),
        "rsi": float(r),
        "vol_ratio": float(vol_ratio),
    }

    vol_ok = vol_ratio >= VOL_RATIO

    if bias_dir == "SHORT":
        ok = close.iloc[-1] < e20 and r > RSI_LOWER and vol_ok
        return ok, info

    if bias_dir == "LONG":
        ok = close.iloc[-1] > e20 and r < RSI_UPPER and vol_ok
        return ok, info

    return False, info


# =========================================================
# TRADE PLAN
# =========================================================

def trade_plan(side, price):
    if side == "SHORT":
        entry_lo = price
        entry_hi = price * (1 + ENTRY_PAD_PCT)
        entry = (entry_lo + entry_hi) / 2
        sl = entry * (1 + MAX_SL_PCT)
        risk = sl - entry
        tps = [entry - risk * r for r in RR_TARGETS]
    else:
        entry_lo = price * (1 - ENTRY_PAD_PCT)
        entry_hi = price
        entry = (entry_lo + entry_hi) / 2
        sl = entry * (1 - MAX_SL_PCT)
        risk = entry - sl
        tps = [entry + risk * r for r in RR_TARGETS]

    return entry_lo, entry_hi, sl, tps


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
    markets = ex.load_markets()
    symbols = [s for s in symbols if s in markets]

    print("✅ BOT AKTIV – Daytrading (1h Bias / 15m Entry)", flush=True)

    last_seen = {}

    while True:
        try:
            for s in symbols:
                df15_raw = fetch_df(ex, s, TIMEFRAME_ENTRY)
                if len(df15_raw) < 3:
                    continue
                df15 = df15_raw.iloc[:-1]
                candle_ts = df15.index[-1]

                if last_seen.get(s) == candle_ts:
                    continue
                last_seen[s] = candle_ts

                df1h = fetch_df(ex, s, TIMEFRAME_BIAS)
                df4h = fetch_df(ex, s, TIMEFRAME_CTX_4H)
                dfd = fetch_df(ex, s, TIMEFRAME_CTX_D1)

                bias_1h = compute_bias(df1h)
                bias_4h = compute_bias(df4h)
                bias_d1 = compute_bias(dfd)

                ok, info = check_entry(df15, bias_1h.direction)

                print(f"[{s}] 1h={bias_1h.direction} ok={ok}", flush=True)

                if not ok:
                    continue

                side = bias_1h.direction
                entry_lo, entry_hi, sl, tps = trade_plan(side, info["close"])

                warn = ""
                if side != bias_4h.direction:
                    warn += "⚠️ Gegen 4h-Trend\n"
                if side != bias_d1.direction:
                    warn += "⚠️ Gegen Tages-Trend\n"

                head = "🟢 LONG" if side == "LONG" else "🔴 SHORT"

                msg = (
                    f"{head} SIGNAL\n\n"
                    f"📊 Pair: {s.replace(':USDT','').replace('/','')}\n"
                    f"🕒 15m Close: {local_time(candle_ts)}\n\n"
                    f"🧭 Bias: 1h={bias_1h.direction} | 4h={bias_4h.direction} | D1={bias_d1.direction}\n"
                    f"{warn}\n"
                    f"🎯 Entry: {entry_lo:.4f} – {entry_hi:.4f}\n"
                    f"🛑 SL: {sl:.4f} (≤2%)\n"
                    f"✅ TP1: {tps[0]:.4f}\n"
                    f"✅ TP2: {tps[1]:.4f}\n"
                    f"✅ TP3: {tps[2]:.4f}\n\n"
                    f"<b>©️ Copyright by crypto_mistik</b>"
                )

                print("\n" + msg + "\n", flush=True)
                send_telegram(msg)

            time.sleep(5)

        except Exception as e:
            print(f"⚠️ ERROR: {e}", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
