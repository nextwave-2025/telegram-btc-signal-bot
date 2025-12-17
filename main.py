import os
import time
import ccxt
import numpy as np
import pandas as pd

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from zoneinfo import ZoneInfo


# =========================================================
# CONFIG
# =========================================================

TIMEFRAME_ENTRY = "15m"
TIMEFRAME_BIAS = "4h"

EMA_FAST = 20
EMA_SLOW = 50

RSI_LOWER = 34
RSI_UPPER = 64

VOL_MA_LEN = 20
VOL_RATIO = 0.12        # realistisch für 15m

MAX_SL_PCT = 0.02       # max 2%
ENTRY_PAD_PCT = 0.0006
RR_TARGETS = (1, 2, 3)

LOCAL_TZ = "Europe/Berlin"

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


@dataclass
class Zone:
    low: float
    high: float
    touches: int = 0
    rejections: int = 0
    strength: float = 0.0


# =========================================================
# HELPERS
# =========================================================

def to_bybit_linear(sym: str) -> str:
    return sym if ":" in sym else f"{sym}:USDT"


def to_local_time(ts) -> str:
    dt = ts.to_pydatetime()
    return dt.astimezone(ZoneInfo(LOCAL_TZ)).strftime("%Y-%m-%d %H:%M:%S %Z")


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# =========================================================
# STRATEGY
# =========================================================

def compute_bias(df4h: pd.DataFrame) -> Bias:
    e20 = ema(df4h["close"], EMA_FAST).iloc[-1]
    e50 = ema(df4h["close"], EMA_SLOW).iloc[-1]
    if e20 > e50:
        return Bias("LONG")
    if e20 < e50:
        return Bias("SHORT")
    return Bias("NEUTRAL")


def check_entry(df15: pd.DataFrame, bias_dir: str) -> Tuple[bool, Dict]:
    close = df15["close"]
    vol = df15["volume"]

    e20 = ema(close, EMA_FAST).iloc[-1]
    e50 = ema(close, EMA_SLOW).iloc[-1]
    r = rsi(close, 14).iloc[-1]

    v = vol.iloc[-1]
    vma = vol.rolling(VOL_MA_LEN).mean().iloc[-1]

    vol_ratio = (v / vma) if vma > 0 else 0.0
    vol_ok = v >= vma * VOL_RATIO

    info = {
        "close": float(close.iloc[-1]),
        "ema20": float(e20),
        "ema50": float(e50),
        "rsi": float(r),
        "vol": float(v),
        "volma": float(vma),
        "vol_ratio": float(vol_ratio),
    }

    if bias_dir == "SHORT":
        ok = close.iloc[-1] < e20 < e50 and r > RSI_LOWER and vol_ok
        return ok, info

    if bias_dir == "LONG":
        ok = close.iloc[-1] > e20 > e50 and r < RSI_UPPER and vol_ok
        return ok, info

    return False, info


# =========================================================
# TRADE PLAN
# =========================================================

def trade_plan(side: str, close: float) -> Dict:
    if side == "SHORT":
        entry_low = close
        entry_high = close * (1 + ENTRY_PAD_PCT)
        entry = (entry_low + entry_high) / 2
        sl = entry * (1 + MAX_SL_PCT)
        risk = sl - entry
        tps = [entry - risk * r for r in RR_TARGETS]
    else:
        entry_low = close * (1 - ENTRY_PAD_PCT)
        entry_high = close
        entry = (entry_low + entry_high) / 2
        sl = entry * (1 - MAX_SL_PCT)
        risk = entry - sl
        tps = [entry + risk * r for r in RR_TARGETS]

    return {
        "entry": (entry_low, entry_high),
        "sl": sl,
        "tp1": tps[0],
        "tp2": tps[1],
        "tp3": tps[2],
    }


# =========================================================
# EXCHANGE
# =========================================================

def make_exchange():
    return ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })


def fetch_df(exchange, symbol, tf, limit=300):
    data = exchange.fetch_ohlcv(symbol, tf, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df


# =========================================================
# MAIN
# =========================================================

def main():
    exchange = make_exchange()
    symbols = [to_bybit_linear(s) for s in SYMBOLS]

    markets = exchange.load_markets()
    symbols = [s for s in symbols if s in markets]

    print(f"✅ BOT AKTIV – 15m Candle Close | Symbols: {symbols}", flush=True)

    last_seen: Dict[str, pd.Timestamp] = {}

    while True:
        try:
            for symbol in symbols:
                df15_raw = fetch_df(exchange, symbol, TIMEFRAME_ENTRY)
                df4h = fetch_df(exchange, symbol, TIMEFRAME_BIAS)

                if len(df15_raw) < 3:
                    continue

                # 🔴 WICHTIG: letzte GESCHLOSSENE Candle
                df15 = df15_raw.iloc[:-1]
                candle_ts = df15.index[-1]

                if last_seen.get(symbol) == candle_ts:
                    continue
                last_seen[symbol] = candle_ts

                bias = compute_bias(df4h)
                ok, info = check_entry(df15, bias.direction)

                print(
                    f"[{symbol}] {candle_ts} bias={bias.direction} "
                    f"ok={ok} rsi={info['rsi']:.2f} "
                    f"vol_ratio={info['vol_ratio']:.2f}",
                    flush=True
                )

                if not ok:
                    continue

                side = bias.direction
                plan = trade_plan(side, info["close"])
                head = "🟢 LONG" if side == "LONG" else "🔴 SHORT"

                print(
                    f"\n{head} SIGNAL {symbol.replace(':USDT','')}\n"
                    f"🕒 {to_local_time(candle_ts)}\n"
                    f"Entry: {plan['entry'][0]:.4f} – {plan['entry'][1]:.4f}\n"
                    f"SL: {plan['sl']:.4f}\n"
                    f"TP1: {plan['tp1']:.4f}\n"
                    f"TP2: {plan['tp2']:.4f}\n"
                    f"TP3: {plan['tp3']:.4f}\n",
                    flush=True
                )

            time.sleep(5)

        except Exception as e:
            print(f"⚠️ BOT ERROR: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
