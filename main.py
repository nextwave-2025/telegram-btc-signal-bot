import os
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import ccxt

from telegram import Bot


# =========================
# CONFIG
# =========================

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT"]

ENTRY_TF = "15m"   # entry timeframe (Candle close only)
BIAS_TF = "4h"     # bias timeframe

EMA_FAST = 20
EMA_SLOW = 50

RSI_UPPER = 64
RSI_LOWER = 34

VOL_MA_LEN = 20

# Prevent signal spam per symbol
SIGNAL_COOLDOWN_SECONDS = 60 * 30  # 30 minutes

# We poll frequently, but we ONLY act when a new 15m candle closed
POLL_SECONDS = 30

# Exchange
EXCHANGE_ID = "bybit"
MARKET_TYPE = "linear"  # USDT Perps on Bybit

# --- Liquidity cluster settings (Orderbook) ---
ORDERBOOK_LIMIT = 200  # levels per side
LIQ_LOOKAROUND_PCT = 0.006  # analyze +/- 0.6% around current price
LIQ_BIN_PCT = 0.0015        # cluster bin size ~0.15% of price
LIQ_DOMINANCE_RATIO = 1.25  # require 25% more liquidity on the preferred side
LIQ_WALL_MULTIPLIER = 2.0   # wall if bin volume > avg_bin_vol * multiplier


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
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def last_closed_candle_ts(df: pd.DataFrame) -> pd.Timestamp:
    return df["ts"].iloc[-1]


# =========================
# SIGNAL LOGIC (V1 core)
# =========================

@dataclass
class Bias:
    direction: str  # "LONG" / "SHORT" / "NEUTRAL"
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
    vma = float(volma_series.iloc[-1]) if not np.isnan(volma_series.iloc[-1]) else float(np.mean(vol.tail(VOL_MA_LEN)))

    info = {"close": c, "ema20": e20, "ema50": e50, "rsi": r, "vol": v, "volma": vma}

    if bias_dir == "LONG":
        ok = (
            c > e20 and
            e20 > e50 and
            (RSI_LOWER < r < RSI_UPPER) and
            v > vma
        )
        reason = "15m confirms LONG: close>EMA20, EMA20>EMA50, RSI 34-64, vol>MA"
        return ok, reason, info

    if bias_dir == "SHORT":
        ok = (
            c < e20 and
            e20 < e50 and
            (RSI_LOWER < r < RSI_UPPER) and
            v > vma
        )
        reason = "15m confirms SHORT: close<EMA20, EMA20<EMA50, RSI 34-64, vol>MA"
        return ok, reason, info

    return False, "Bias neutral (4h EMA20 ~= EMA50)", info


# =========================
# LIQUIDITY CLUSTERS (Orderbook)
# =========================

@dataclass
class LiquidityInfo:
    mid: float
    liq_below: float
    liq_above: float
    dominance: float          # below/above ratio
    top_bid_wall_price: Optional[float]
    top_ask_wall_price: Optional[float]
    top_bid_wall_size: float
    top_ask_wall_size: float
    lookaround_pct: float
    bin_pct: float

def _bin_levels(levels, mid: float, bin_size: float, lo: float, hi: float) -> Dict[int, float]:
    """
    levels: list of [price, size]
    returns bins: bin_index -> summed_size
    """
    bins: Dict[int, float] = {}
    for price, size in levels:
        if price < lo or price > hi:
            continue
        idx = int((price - lo) // bin_size)
        bins[idx] = bins.get(idx, 0.0) + float(size)
    return bins

def analyze_liquidity(ex, symbol: str) -> LiquidityInfo:
    ob = ex.fetch_order_book(symbol, limit=ORDERBOOK_LIMIT)
    bids = ob.get("bids", [])  # [price, amount]
    asks = ob.get("asks", [])

    if not bids or not asks:
        raise RuntimeError("Empty orderbook")

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0

    lo = mid * (1.0 - LIQ_LOOKAROUND_PCT)
    hi = mid * (1.0 + LIQ_LOOKAROUND_PCT)

    bin_size = mid * LIQ_BIN_PCT

    bid_bins = _bin_levels(bids, mid, bin_size, lo, hi)
    ask_bins = _bin_levels(asks, mid, bin_size, lo, hi)

    # Total liquidity below and above mid
    # For bids: mostly below mid, for asks: mostly above mid — but we enforce by price comparison
    liq_below = 0.0
    for price, size in bids:
        p = float(price)
        if lo <= p < mid:
            liq_below += float(size)

    liq_above = 0.0
    for price, size in asks:
        p = float(price)
        if mid < p <= hi:
            liq_above += float(size)

    dominance = (liq_below / liq_above) if liq_above > 0 else float("inf")

    # Identify walls using binned size vs average bin size
    all_bid_bin_sizes = list(bid_bins.values()) or [0.0]
    all_ask_bin_sizes = list(ask_bins.values()) or [0.0]

    avg_bid = float(np.mean(all_bid_bin_sizes))
    avg_ask = float(np.mean(all_ask_bin_sizes))

    # Get strongest wall bin
    top_bid_wall_idx = max(bid_bins, key=lambda k: bid_bins[k]) if bid_bins else None
    top_ask_wall_idx = max(ask_bins, key=lambda k: ask_bins[k]) if ask_bins else None

    top_bid_wall_size = float(bid_bins[top_bid_wall_idx]) if top_bid_wall_idx is not None else 0.0
    top_ask_wall_size = float(ask_bins[top_ask_wall_idx]) if top_ask_wall_idx is not None else 0.0

    # Convert bin idx to representative price (bin center)
    def bin_center(idx: int) -> float:
        return lo + (idx + 0.5) * bin_size

    top_bid_wall_price = None
    if top_bid_wall_idx is not None and avg_bid > 0 and top_bid_wall_size >= avg_bid * LIQ_WALL_MULTIPLIER:
        top_bid_wall_price = float(bin_center(top_bid_wall_idx))

    top_ask_wall_price = None
    if top_ask_wall_idx is not None and avg_ask > 0 and top_ask_wall_size >= avg_ask * LIQ_WALL_MULTIPLIER:
        top_ask_wall_price = float(bin_center(top_ask_wall_idx))

    return LiquidityInfo(
        mid=mid,
        liq_below=liq_below,
        liq_above=liq_above,
        dominance=dominance,
        top_bid_wall_price=top_bid_wall_price,
        top_ask_wall_price=top_ask_wall_price,
        top_bid_wall_size=top_bid_wall_size,
        top_ask_wall_size=top_ask_wall_size,
        lookaround_pct=LIQ_LOOKAROUND_PCT,
        bin_pct=LIQ_BIN_PCT,
    )

def liquidity_allows(direction: str, liq: LiquidityInfo) -> Tuple[bool, str]:
    """
    LONG requires:
      - liq_below >= liq_above * LIQ_DOMINANCE_RATIO
      - and ideally a bid wall below mid (optional but strong)
    SHORT requires:
      - liq_above >= liq_below * LIQ_DOMINANCE_RATIO
      - and ideally an ask wall above mid
    """
    if direction == "LONG":
        if liq.liq_above <= 0:
            return True, "Liquidity OK: no asks in range"
        if liq.liq_below >= liq.liq_above * LIQ_DOMINANCE_RATIO:
            if liq.top_bid_wall_price is not None and liq.top_bid_wall_price < liq.mid:
                return True, f"Liquidity OK: bids dominate ({liq.dominance:.2f}x) + bid wall at {liq.top_bid_wall_price:.4f}"
            return True, f"Liquidity OK: bids dominate ({liq.dominance:.2f}x)"
        return False, f"Liquidity blocks LONG: bids/asks ratio {liq.dominance:.2f}x (< {LIQ_DOMINANCE_RATIO}x)"

    if direction == "SHORT":
        if liq.liq_below <= 0:
            return True, "Liquidity OK: no bids in range"
        ratio = (liq.liq_above / liq.liq_below) if liq.liq_below > 0 else float("inf")
        if liq.liq_above >= liq.liq_below * LIQ_DOMINANCE_RATIO:
            if liq.top_ask_wall_price is not None and liq.top_ask_wall_price > liq.mid:
                return True, f"Liquidity OK: asks dominate ({ratio:.2f}x) + ask wall at {liq.top_ask_wall_price:.4f}"
            return True, f"Liquidity OK: asks dominate ({ratio:.2f}x)"
        return False, f"Liquidity blocks SHORT: asks/bids ratio {ratio:.2f}x (< {LIQ_DOMINANCE_RATIO}x)"

    return False, "Liquidity: neutral direction"


# =========================
# TELEGRAM
# =========================

def format_signal(symbol: str, direction: str, reason: str, info: Dict, bias: Bias, candle_ts: str,
                  liq_reason: str, liq: LiquidityInfo) -> str:
    wall_line = ""
    if direction == "LONG" and liq.top_bid_wall_price is not None:
        wall_line = f"🧱 Bid wall: ~{liq.top_bid_wall_price:.4f}\n"
    if direction == "SHORT" and liq.top_ask_wall_price is not None:
        wall_line = f"🧱 Ask wall: ~{liq.top_ask_wall_price:.4f}\n"

    return (
        f"🚨 SIGNAL ({direction})\n\n"
        f"📊 Pair: {symbol.replace('/', '')}\n"
        f"🕒 Entry TF: {ENTRY_TF} (Candle Close)\n"
        f"🧭 Bias TF: {BIAS_TF} ({bias.direction})\n"
        f"🕯️ Candle: {candle_ts}\n\n"
        f"💰 Close: {info['close']:.4f}\n"
        f"📈 EMA{EMA_FAST}: {info['ema20']:.4f}\n"
        f"📉 EMA{EMA_SLOW}: {info['ema50']:.4f}\n"
        f"📍 RSI(14): {info['rsi']:.2f} (Bands {RSI_LOWER}/{RSI_UPPER})\n"
        f"📦 Vol: {info['vol']:.2f} | VolMA({VOL_MA_LEN}): {info['volma']:.2f}\n\n"
        f"🌊 Liquidity (±{liq.lookaround_pct*100:.2f}% range):\n"
        f"Below: {liq.liq_below:.2f} | Above: {liq.liq_above:.2f}\n"
        f"{wall_line}"
        f"✅ Liquidity: {liq_reason}\n\n"
        f"✅ Setup: {reason}\n"
        f"⚠️ Hinweis: V2 (Liquidity Filter aktiv, News Filter folgt)"
    )

async def send_telegram(bot: Bot, chat_id: str, text: str) -> None:
    await bot.send_message(chat_id=chat_id, text=text)


# =========================
# EXCHANGE
# =========================

def make_exchange():
    ex_class = getattr(ccxt, EXCHANGE_ID)
    return ex_class({
        "enableRateLimit": True,
        "options": {"defaultType": MARKET_TYPE},
    })


# =========================
# MAIN LOOP
# =========================

async def run():
    bot_token = (os.getenv("BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("CHAT_ID") or "").strip()  # IMPORTANT

    if not bot_token or not chat_id:
        raise RuntimeError("Missing BOT_TOKEN or CHAT_ID in Railway Variables")

    bot = Bot(token=bot_token)
    ex = make_exchange()

    last_seen_candle: Dict[str, pd.Timestamp] = {}
    last_signal_time: Dict[str, float] = {}

    try:
        await send_telegram(bot, chat_id, "✅ Signal-Bot gestartet (V2). 15m Close + 4h Bias + Liquidity Filter aktiv.")
    except Exception as e:
        print(f"Startup Telegram failed: {type(e).__name__}: {e}", flush=True)

    while True:
        try:
            for symbol in SYMBOLS:
                # Fetch candles
                ohlcv_15m = ex.fetch_ohlcv(symbol, timeframe=ENTRY_TF, limit=220)
                ohlcv_4h = ex.fetch_ohlcv(symbol, timeframe=BIAS_TF, limit=220)

                df15 = to_df(ohlcv_15m)
                df4h = to_df(ohlcv_4h)

                candle_ts = last_closed_candle_ts(df15)

                # Candle close only
                if symbol in last_seen_candle and candle_ts == last_seen_candle[symbol]:
                    continue
                last_seen_candle[symbol] = candle_ts

                bias = compute_bias(df4h)
                ok_entry, reason, info = check_entry(df15, bias.direction)

                if not (ok_entry and bias.direction in ("LONG", "SHORT")):
                    continue

                # Cooldown
                now = time.time()
                if now - last_signal_time.get(symbol, 0) < SIGNAL_COOLDOWN_SECONDS:
                    continue

                # Liquidity filter (Orderbook)
                liq = analyze_liquidity(ex, symbol)
                ok_liq, liq_reason = liquidity_allows(bias.direction, liq)

                if not ok_liq:
                    # optional: comment out if you don't want "filtered" messages
                    # print(f"{symbol} filtered by liquidity: {liq_reason}", flush=True)
                    continue

                text = format_signal(
                    symbol=symbol,
                    direction=bias.direction,
                    reason=reason,
                    info=info,
                    bias=bias,
                    candle_ts=str(candle_ts),
                    liq_reason=liq_reason,
                    liq=liq,
                )
                await send_telegram(bot, chat_id, text)
                last_signal_time[symbol] = now

        except Exception as e:
            err = f"⚠️ Bot error: {type(e).__name__}: {e}"
            print(err, flush=True)
            try:
                await send_telegram(bot, chat_id, err)
            except Exception:
                pass

        await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    asyncio.run(run())
