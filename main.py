import os
import time
import ccxt
import numpy as np
import pandas as pd
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Tuple, List
from zoneinfo import ZoneInfo
from datetime import datetime


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

RSI_SCALP_LONG = 28
RSI_SCALP_SHORT = 72

VOL_MA_LEN = 20
VOL_RATIO = 0.12

ATR_LEN = 14
ATR_MULTIPLIER = 1.5

CRV_MIN = 2.0

COOLDOWN_MINUTES = 45
SCALP_COOLDOWN_MINUTES = 90

QUIET_HOURS_START = 22
QUIET_HOURS_END = 7

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
    "<b>©️ Copyright by crypto_mistik</b>\n\n"
    "⚠️ <b>Kein Financial Advice</b>\n"
    "Kryptowährungen sind hochvolatil. Der Handel erfolgt auf eigenes Risiko.\n"
    "Die Nutzung dieses AI-Trading-Bots oder das Kopieren von Code ohne gültigen "
    "<b>Premium-Zugang</b> ist untersagt und kann rechtlich verfolgt werden."
)


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class Bias:
    direction: str  # LONG / SHORT / NEUTRAL


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

def to_bybit(sym: str) -> str:
    return sym if ":" in sym else f"{sym}:USDT"

def german_time(ts: pd.Timestamp) -> str:
    return ts.to_pydatetime().astimezone(
        ZoneInfo(LOCAL_TZ)
    ).strftime("%d.%m.%Y – %H:%M Uhr")

def in_quiet_hours(ts: pd.Timestamp) -> bool:
    hour = ts.to_pydatetime().astimezone(ZoneInfo(LOCAL_TZ)).hour
    return hour >= QUIET_HOURS_START or hour < QUIET_HOURS_END

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def atr(df: pd.DataFrame, n: int = 14) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])

def minutes_between(a, b) -> float:
    return abs((a.to_pydatetime() - b.to_pydatetime()).total_seconds()) / 60.0


# =========================================================
# MARKET STRUCTURE
# =========================================================

def swing_levels(df: pd.DataFrame, lookback: int = 40) -> Tuple[List[float], List[float]]:
    highs = df["high"].tail(lookback)
    lows = df["low"].tail(lookback)
    return sorted(highs.nlargest(3)), sorted(lows.nsmallest(3))

def zone_text(levels: List[float]) -> str:
    if not levels:
        return "—"
    return f"{min(levels):.4f} – {max(levels):.4f}"


# =========================================================
# STRATEGY CORE
# =========================================================

def compute_bias(df: pd.DataFrame) -> Bias:
    e20 = ema(df["close"], EMA_FAST).iloc[-1]
    e50 = ema(df["close"], EMA_SLOW).iloc[-1]
    if e20 > e50:
        return Bias("LONG")
    if e20 < e50:
        return Bias("SHORT")
    return Bias("NEUTRAL")


def trend_entry_15m(df15: pd.DataFrame, side: str) -> Tuple[bool, str, Dict]:
    close = df15["close"]
    vol = df15["volume"]

    c0 = close.iloc[-1]
    c1 = close.iloc[-2]

    e20 = ema(close, EMA_FAST).iloc[-1]
    e50 = ema(close, EMA_SLOW).iloc[-1]
    r = rsi(close).iloc[-1]

    v = vol.iloc[-1]
    vma = vol.rolling(VOL_MA_LEN).mean().iloc[-1]
    vol_ratio = v / vma if vma > 0 else 0

    info = {
        "close": float(c0),
        "ema20": float(e20),
        "ema50": float(e50),
        "rsi": float(r),
        "vol_ratio": float(vol_ratio)
    }

    if vol_ratio < VOL_RATIO:
        return False, "LOW_VOLUME", info

    if side == "SHORT":
        if r <= RSI_LOWER:
            return False, "RSI_BLOCK", info
        if c1 > e20 and c0 < e20:
            return True, "SAFE_PULLBACK", info

    if side == "LONG":
        if r >= RSI_UPPER:
            return False, "RSI_BLOCK", info
        if c1 < e20 and c0 > e20:
            return True, "SAFE_PULLBACK", info

    return False, "NO_TRIGGER", info


# =========================================================
# MAIN
# =========================================================

def fetch_df(ex, sym, tf, limit=300):
    data = ex.fetch_ohlcv(sym, tf, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df


def main():
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    symbols = [to_bybit(s) for s in SYMBOLS]
    ex.load_markets()

    last_seen: Dict[str, pd.Timestamp] = {}
    last_signal: Dict[str, pd.Timestamp] = {}

    print("✅ BOT AKTIV – FINAL VERSION", flush=True)

    while True:
        try:
            for sym in symbols:
                df15_raw = fetch_df(ex, sym, TIMEFRAME_ENTRY)
                df15 = df15_raw.iloc[:-1]
                ts = df15.index[-1]

                if last_seen.get(sym) == ts:
                    continue
                last_seen[sym] = ts

                df1h = fetch_df(ex, sym, TIMEFRAME_BIAS)
                df4h = fetch_df(ex, sym, TIMEFRAME_CTX_4H)

                bias_1h = compute_bias(df1h)
                bias_4h = compute_bias(df4h)

                if bias_1h.direction not in ("LONG", "SHORT"):
                    continue

                ok, entry_type, info = trend_entry_15m(df15, bias_1h.direction)
                if not ok:
                    continue

                if in_quiet_hours(ts):
                    print("⏸ Quiet Hours – Alert unterdrückt", flush=True)
                    continue

                key = f"{sym}|{bias_1h.direction}"
                if key in last_signal and minutes_between(ts, last_signal[key]) < COOLDOWN_MINUTES:
                    continue
                last_signal[key] = ts

                atr_val = atr(df15)
                sl_dist = atr_val * ATR_MULTIPLIER

                if bias_1h.direction == "SHORT":
                    sl = info["close"] + sl_dist
                    tp1 = info["close"] - sl_dist
                    tp2 = info["close"] - sl_dist * 2
                else:
                    sl = info["close"] - sl_dist
                    tp1 = info["close"] + sl_dist
                    tp2 = info["close"] + sl_dist * 2

                crv = abs((tp2 - info["close"]) / (info["close"] - sl))
                if crv < CRV_MIN:
                    continue

                highs, lows = swing_levels(df4h)
                support = zone_text(lows)
                resistance = zone_text(highs)

                head = "🟢 LONG" if bias_1h.direction == "LONG" else "🔴 SHORT"
                pair = sym.replace(":USDT","").replace("/","")

                msg = (
                    f"{head} SETUP ({entry_type})\n\n"
                    f"📊 Pair: {pair}\n"
                    f"🕒 Zeit: {german_time(ts)}\n\n"
                    f"🧭 Bias: 1h={bias_1h.direction} | 4h={bias_4h.direction}\n\n"
                    f"💰 Close: {info['close']:.4f}\n"
                    f"📍 RSI(14): {info['rsi']:.2f}\n\n"
                    f"🧱 Support: {support}\n"
                    f"🧱 Resistance: {resistance}\n\n"
                    f"🎯 Entry: {info['close']:.4f}\n"
                    f"🛑 SL (ATR 1.5x): {sl:.4f}\n"
                    f"✅ TP1: {tp1:.4f} (1R)\n"
                    f"✅ TP2: {tp2:.4f} (2R)\n"
                    f"➖ TP3: optional (Struktur)\n\n"
                    f"{FOOTER_TEXT}"
                )

                print(msg, flush=True)
                send_telegram(msg)

            time.sleep(5)

        except Exception as e:
            print("⚠️ ERROR:", e, flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
