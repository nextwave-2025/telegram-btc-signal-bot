import os
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import ccxt

from telegram import Bot

# =========================
# CONFIG
# =========================

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT"]

ENTRY_TF = "15m"   # entry timeframe
BIAS_TF = "4h"     # bias timeframe

EMA_FAST = 20
EMA_SLOW = 50

RSI_UPPER = 64
RSI_LOWER = 34

VOL_MA_LEN = 20

# Prevent spam (per symbol)
SIGNAL_COOLDOWN_SECONDS = 60 * 30  # 30 minutes

# We poll often, but only act on NEW 15m candle close
POLL_SECONDS = 30

# Exchange
EXCHANGE_ID = "bybit"
MARKET_TYPE = "linear"  # USDT Perps on Bybit


# =========================
# INDICATORS
# =========================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def to_df(ohlcv) -> pd.DataFrame:
    # ohlcv: [timestamp, open, high, low, close, volume]
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def last_closed_candle_ts(df: pd.DataFrame) -> pd.Timestamp:
    # ccxt returns closed candles; last row is last closed
    return df["ts"].iloc[-1]


# =========================
# SIGNAL LOGIC
# =========================

@dataclass
class Bias:
    direction: str  # "LONG" or "SHORT" or "NEUTRAL"
    ema_fast: float
    ema_slow: float

def compute_bias(df_4h: pd.DataFrame) -> Bias:
    close = df_4h["close"]
    e20 = float(ema(close, EMA_FAST).iloc[-1])
    e50 = float(ema(close, EMA_SLOW).iloc[-1])

    if e20 > e50:
        direction = "LONG"
    elif e20 < e50:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    return Bias(direction=direction, ema_fast=e20, ema_slow=e50)

def check_entry(df_15m: pd.DataFrame, bias_dir: str) -> Tuple[bool, str, Dict]:
    """
    V1 "setup intakt" rules:
    - 4h bias from EMA20 vs EMA50
    - 15m confirmation at candle close:
        LONG:
          - Close > EMA20 and EMA20 > EMA50
          - RSI between 34 and 64 (avoid extremes)
          - Volume > VolumeMA(20)
        SHORT:
          - Close < EMA20 and EMA20 < EMA50
          - RSI between 34 and 64
          - Volume > VolumeMA(20)
    """
    close = df_15m["close"]
    vol = df_15m["volume"]

    e20_series = ema(close, EMA_FAST)
    e50_series = ema(close, EMA_SLOW)
    rsi_series = rsi(close, 14)
    volma_series = vol.rolling(VOL_MA_LEN).mean()

    c = float(close.iloc[-1])
    e20 = float(e20_series.iloc[-1])
    e50 = float(e50_series.iloc[-1])
    r = float(rsi_series.iloc[-1])
    v = float(vol.iloc[-1])

    vma_val = volma_series.iloc[-1]
    if np.isnan(vma_val):
        vma = float(np.mean(vol.tail(VOL_MA_LEN)))
    else:
        vma = float(vma_val)

    info = {"close": c, "ema20": e20, "ema50": e50, "rsi": r, "vol": v, "volma": vma}

    if bias_dir == "LONG":
        conds = [c > e20, e20 > e50, r < RSI_UPPER, r > RSI_LOWER, v > vma]
        ok = all(conds)
        reason = "15m LONG confirm: close>EMA20, EMA20>EMA50, RSI 34-64, vol>MA"
        return ok, reason, info

    if bias_dir == "SHORT":
        conds = [c < e20, e20 < e50, r > RSI_LOWER, r < RSI_UPPER, v > vma]
        ok = all(conds)
        reason = "15m SHORT confirm: close<EMA20, EMA20<EMA50, RSI 34-64, vol>MA"
        return ok, reason, info

    return False, "4h bias neutral (EMA20 ~= EMA50)", info


# =========================
# TELEGRAM
# =========================

def format_signal(symbol: str, direction: str, reason: str, info: Dict, bias: Bias, candle_ts: str) -> str:
    return (
        f"🚨 SIGNAL ({direction})\n\n"
        f"📊 Pair: {symbol.replace('/', '')}\n"
        f"🕒 Entry TF: {ENTRY_TF} (Candle Close)\n"
        f"🧭 Bias TF: {BIAS_TF} ({bias.direction})\n"
        f"🕯️ Candle: {candle_ts}\n\n"
        f"💰 Close: {info['close']:.6f}\n"
        f"📈 EMA{EMA_FAST}: {info['ema20']:.6f}\n"
        f"📉 EMA{EMA_SLOW}: {info['ema50']:.6f}\n"
        f"📍 RSI(14): {info['rsi']:.2f} (Bands {RSI_LOWER}/{RSI_UPPER})\n"
        f"📦 Vol: {info['vol']:.2f} | VolMA({VOL_MA_LEN}): {info['volma']:.2f}\n\n"
        f"✅ Setup: {reason}\n"
        f"⚠️ Hinweis: V1 Engine (Liquidity/News folgen als Filter)"
    )

async def send_telegram(bot: Bot, chat_id: str, text: str) -> None:
    await bot.send_message(chat_id=chat_id, text=text)


# =========================
# MAIN LOOP
# =========================

def make_exchange():
    ex_class = getattr(ccxt, EXCHANGE_ID)
    return ex_class({
        "enableRateLimit": True,
        "options": {"defaultType": MARKET_TYPE},
    })

async def run():
    bot_token = (os.getenv("BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("CHAT_ID") or "").strip()

    if not bot_token or not chat_id:
        raise RuntimeError("Missing BOT_TOKEN or CHAT_ID in Railway Variables")

    bot = Bot(token=bot_token)
    ex = make_exchange()

    last_seen_candle: Dict[str, pd.Timestamp] = {}
    last_signal_time: Dict[str, float] = {}

    # Startup message should never crash the process
    try:
        await send_telegram(bot, chat_id, "✅ Signal-Bot gestartet (V1). Warte auf 15m Candle-Closes…")
    except Exception as e:
        print(f"Startup message failed: {type(e).__name__}: {e}", flush=True)

    while True:
        try:
            for symbol in SYMBOLS:
                # Fetch 15m + 4h
                ohlcv_15m = ex.fetch_ohlcv(symbol, timeframe=ENTRY_TF, limit=200)
                ohlcv_4h = ex.fetch_ohlcv(symbol, timeframe=BIAS_TF, limit=200)

                df15 = to_df(ohlcv_15m)
                df4h = to_df(ohlcv_4h)

                candle_ts = last_closed_candle_ts(df15)

                # Candle-close only: act only when new 15m candle appears
                if symbol in last_seen_candle and candle_ts == last_seen_candle[symbol]:
                    continue
                last_seen_candle[symbol] = candle_ts

                bias = compute_bias(df4h)
                ok, reason, info = check_entry(df15, bias.direction)

                # Cooldown
                now = time.time()
                if now - last_signal_time.get(symbol, 0) < SIGNAL_COOLDOWN_SECONDS:
                    continue

                if ok and bias.direction in ("LONG", "SHORT"):
                    text = format_signal(
                        symbol=symbol,
                        direction=bias.direction,
                        reason=reason,
                        info=info,
                        bias=bias,
                        candle_ts=str(candle_ts),
                    )
                    await send_telegram(bot, chat_id, text)
                    last_signal_time[symbol] = now

        except Exception as e:
            err = f"⚠️ Bot error: {type(e).__name__}: {e}"
            print(err, flush=True)
            # Do NOT crash if Telegram fails
            try:
                await send_telegram(bot, chat_id, err)
            except Exception:
                pass

        await asyncio.sleep(POLL_SECONDS)

if __name__ == "__main__":
    asyncio.run(run())
