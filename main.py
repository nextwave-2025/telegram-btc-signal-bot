import os
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import ccxt

from telegram import Bot


# =========================
# CONFIG
# =========================

DEBUG_LOGS = True  # prints to Railway logs

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT"]  # add "DOGE/USDT" if you want

ENTRY_TF = "15m"   # entry timeframe (candle close only)
BIAS_TF = "4h"     # bias timeframe

EMA_FAST = 20
EMA_SLOW = 50

RSI_UPPER = 64
RSI_LOWER = 34

VOL_MA_LEN = 20

SIGNAL_COOLDOWN_SECONDS = 60 * 30  # 30 minutes per symbol
POLL_SECONDS = 30                  # we poll often, but act only on NEW 15m candle

# Exchange
EXCHANGE_ID = "bybit"
MARKET_TYPE = "linear"  # USDT perps on Bybit via ccxt

# --- Liquidity Cluster V3 (Orderbook) ---
ORDERBOOK_LIMIT = 200
LIQ_LOOKAROUND_PCT = 0.008     # +/- 0.8% around mid
LIQ_BIN_PCT = 0.0015           # bin size ~0.15% of price
LIQ_WALL_MULTIPLIER = 2.2      # wall if bin size > avg_bin * multiplier

# Multi-snapshot fake wall filter
OB_SNAPSHOTS = 3
OB_SNAPSHOT_DELAY_SEC = 0.6
WALL_PERSIST_MIN = 2           # wall must appear in >=2 snapshots
WALL_PRICE_TOLERANCE_BINS = 1  # match walls across snapshots within +/-1 bin

# Direction rules
DOMINANCE_RATIO_NEAR_WALL = 1.15  # if near wall exists, allow with slightly weaker dominance
DOMINANCE_RATIO_NO_WALL = 1.40    # if no near wall, require stronger dominance
MAX_WALL_DISTANCE_PCT = 0.005     # wall considered "near" if within 0.5% from mid in direction
TOP_WALLS_TO_REPORT = 3


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
# CORE SIGNAL LOGIC
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
        ok = (c > e20) and (e20 > e50) and (RSI_LOWER < r < RSI_UPPER) and (v > vma)
        reason = "15m confirms LONG: close>EMA20, EMA20>EMA50, RSI 34-64, vol>MA"
        return ok, reason, info

    if bias_dir == "SHORT":
        ok = (c < e20) and (e20 < e50) and (RSI_LOWER < r < RSI_UPPER) and (v > vma)
        reason = "15m confirms SHORT: close<EMA20, EMA20<EMA50, RSI 34-64, vol>MA"
        return ok, reason, info

    return False, "Bias neutral (4h EMA20 ~= EMA50)", info


# =========================
# LIQUIDITY CLUSTER V3
# =========================

@dataclass
class Wall:
    price: float
    size: float
    distance_pct: float

@dataclass
class LiquidityInfo:
    mid: float
    liq_below: float
    liq_above: float
    dominance_below_over_above: float
    dominance_above_over_below: float
    bid_walls_below: List[Wall]
    ask_walls_above: List[Wall]
    has_near_bid_wall: bool
    has_near_ask_wall: bool

def _bounds(mid: float) -> Tuple[float, float]:
    lo = mid * (1.0 - LIQ_LOOKAROUND_PCT)
    hi = mid * (1.0 + LIQ_LOOKAROUND_PCT)
    return lo, hi

def _bin_size(mid: float) -> float:
    return mid * LIQ_BIN_PCT

def _bin_index(price: float, lo: float, bin_size: float) -> int:
    return int((price - lo) // bin_size)

def _bin_center(idx: int, lo: float, bin_size: float) -> float:
    return lo + (idx + 0.5) * bin_size

def _bin_levels(levels, lo: float, hi: float, bin_size: float) -> Dict[int, float]:
    bins: Dict[int, float] = {}
    for price, size in levels:
        p = float(price)
        if p < lo or p > hi:
            continue
        idx = _bin_index(p, lo, bin_size)
        bins[idx] = bins.get(idx, 0.0) + float(size)
    return bins

def _extract_wall_bins(bins: Dict[int, float]) -> List[Tuple[int, float]]:
    sizes = list(bins.values())
    if not sizes:
        return []
    avg = float(np.mean(sizes))
    if avg <= 0:
        return []
    out = [(idx, float(sz)) for idx, sz in bins.items() if float(sz) >= avg * LIQ_WALL_MULTIPLIER]
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def _merge_persistent_walls(walls_per_snapshot: List[List[Tuple[int, float]]]) -> Dict[int, float]:
    # idx -> {"count": int, "max": float}
    track: Dict[int, Dict[str, float]] = {}

    for snap_walls in walls_per_snapshot:
        matched_this_snapshot = set()
        for idx, sz in snap_walls:
            match = None
            for existing_idx in track.keys():
                if abs(existing_idx - idx) <= WALL_PRICE_TOLERANCE_BINS and existing_idx not in matched_this_snapshot:
                    match = existing_idx
                    break

            if match is None:
                track[idx] = {"count": 1.0, "max": float(sz)}
                matched_this_snapshot.add(idx)
            else:
                track[match]["count"] += 1.0
                track[match]["max"] = max(track[match]["max"], float(sz))
                matched_this_snapshot.add(match)

    persistent: Dict[int, float] = {}
    for idx, meta in track.items():
        if int(meta["count"]) >= WALL_PERSIST_MIN:
            persistent[idx] = float(meta["max"])

    return persistent

def _sum_liquidity_band(bids, asks, mid: float, lo: float, hi: float) -> Tuple[float, float]:
    below = 0.0
    for price, size in bids:
        p = float(price)
        if lo <= p < mid:
            below += float(size)

    above = 0.0
    for price, size in asks:
        p = float(price)
        if mid < p <= hi:
            above += float(size)

    return below, above

def analyze_liquidity_v3(ex, symbol: str) -> LiquidityInfo:
    snapshots = []
    for i in range(OB_SNAPSHOTS):
        ob = ex.fetch_order_book(symbol, limit=ORDERBOOK_LIMIT)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            raise RuntimeError("Empty orderbook")

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        snapshots.append((mid, bids, asks))

        if i < OB_SNAPSHOTS - 1:
            time.sleep(OB_SNAPSHOT_DELAY_SEC)

    mid, bids, asks = snapshots[-1]
    lo, hi = _bounds(mid)
    bs = _bin_size(mid)

    liq_below, liq_above = _sum_liquidity_band(bids, asks, mid, lo, hi)
    dom_ba = (liq_below / liq_above) if liq_above > 0 else float("inf")
    dom_ab = (liq_above / liq_below) if liq_below > 0 else float("inf")

    bid_wall_candidates: List[List[Tuple[int, float]]] = []
    ask_wall_candidates: List[List[Tuple[int, float]]] = []

    for (_m, b, a) in snapshots:
        bid_bins = _bin_levels(b, lo, hi, bs)
        ask_bins = _bin_levels(a, lo, hi, bs)
        bid_wall_candidates.append(_extract_wall_bins(bid_bins))
        ask_wall_candidates.append(_extract_wall_bins(ask_bins))

    persistent_bid = _merge_persistent_walls(bid_wall_candidates)
    persistent_ask = _merge_persistent_walls(ask_wall_candidates)

    bid_walls_below: List[Wall] = []
    for idx, sz in persistent_bid.items():
        price = _bin_center(idx, lo, bs)
        if price >= mid:
            continue
        dist = (mid - price) / mid
        bid_walls_below.append(Wall(price=float(price), size=float(sz), distance_pct=float(dist)))
    bid_walls_below.sort(key=lambda w: w.size, reverse=True)
    bid_walls_below = bid_walls_below[:TOP_WALLS_TO_REPORT]

    ask_walls_above: List[Wall] = []
    for idx, sz in persistent_ask.items():
        price = _bin_center(idx, lo, bs)
        if price <= mid:
            continue
        dist = (price - mid) / mid
        ask_walls_above.append(Wall(price=float(price), size=float(sz), distance_pct=float(dist)))
    ask_walls_above.sort(key=lambda w: w.size, reverse=True)
    ask_walls_above = ask_walls_above[:TOP_WALLS_TO_REPORT]

    has_near_bid = any(w.distance_pct <= MAX_WALL_DISTANCE_PCT for w in bid_walls_below)
    has_near_ask = any(w.distance_pct <= MAX_WALL_DISTANCE_PCT for w in ask_walls_above)

    return LiquidityInfo(
        mid=float(mid),
        liq_below=float(liq_below),
        liq_above=float(liq_above),
        dominance_below_over_above=float(dom_ba),
        dominance_above_over_below=float(dom_ab),
        bid_walls_below=bid_walls_below,
        ask_walls_above=ask_walls_above,
        has_near_bid_wall=has_near_bid,
        has_near_ask_wall=has_near_ask,
    )

def liquidity_allows_v3(direction: str, liq: LiquidityInfo) -> Tuple[bool, str]:
    if direction == "LONG":
        required = DOMINANCE_RATIO_NEAR_WALL if liq.has_near_bid_wall else DOMINANCE_RATIO_NO_WALL
        if liq.dominance_below_over_above >= required:
            return True, f"Liquidity OK LONG: bids/asks={liq.dominance_below_over_above:.2f}x (req {required}x)"
        return False, f"Liquidity BLOCK LONG: bids/asks={liq.dominance_below_over_above:.2f}x (<{required}x)"

    if direction == "SHORT":
        required = DOMINANCE_RATIO_NEAR_WALL if liq.has_near_ask_wall else DOMINANCE_RATIO_NO_WALL
        if liq.dominance_above_over_below >= required:
            return True, f"Liquidity OK SHORT: asks/bids={liq.dominance_above_over_below:.2f}x (req {required}x)"
        return False, f"Liquidity BLOCK SHORT: asks/bids={liq.dominance_above_over_below:.2f}x (<{required}x)"

    return False, "Liquidity: neutral direction"


# =========================
# TELEGRAM FORMATTING
# =========================

def _fmt_walls(label: str, walls: List[Wall]) -> str:
    if not walls:
        return f"{label}: none"
    parts = []
    for w in walls:
        parts.append(f"{w.price:.4f} ({w.size:.2f}, {w.distance_pct*100:.2f}%)")
    return f"{label}: " + " | ".join(parts)

def format_signal(symbol: str, direction: str, candle_ts: str, bias: Bias, reason: str, info: Dict,
                  liq_reason: str, liq: LiquidityInfo) -> str:
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
        f"🌊 Liquidity V3 (Bybit L2, ±{LIQ_LOOKAROUND_PCT*100:.2f}%):\n"
        f"Below: {liq.liq_below:.2f} | Above: {liq.liq_above:.2f}\n"
        f"{_fmt_walls('🧱 Bid walls below', liq.bid_walls_below)}\n"
        f"{_fmt_walls('🧱 Ask walls above', liq.ask_walls_above)}\n"
        f"✅ {liq_reason}\n\n"
        f"✅ Setup: {reason}\n"
        f"⚠️ Hinweis: V3 (Liquidity Fake-Wall Filter aktiv; News/SR folgen)"
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
# MAIN
# =========================

async def run():
    bot_token = (os.getenv("BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("CHAT_ID") or "").strip()

    if not bot_token or not chat_id:
        raise RuntimeError("Missing BOT_TOKEN or CHAT_ID in Railway Variables")

    bot = Bot(token=bot_token)
    ex = make_exchange()

    last_seen_candle: Dict[str, pd.Timestamp] = {}
    last_signal_time: Dict[str, float] = {}

    print("Bot booting…", flush=True)
    try:
        await send_telegram(bot, chat_id, "✅ Signal-Bot gestartet (V3). 15m Close + 4h Bias + Liquidity V3 aktiv.")
        print("Startup message sent ✅", flush=True)
    except Exception as e:
        print(f"Startup Telegram failed: {type(e).__name__}: {e}", flush=True)

    while True:
        try:
            for symbol in SYMBOLS:
                if DEBUG_LOGS:
                    print(f"Checking {symbol}", flush=True)

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

                if DEBUG_LOGS:
                    print(
                        f"[{symbol}] candle={candle_ts} bias={bias.direction} entry_ok={ok_entry} "
                        f"close={info['close']:.4f} rsi={info['rsi']:.2f} "
                        f"ema20={info['ema20']:.4f} ema50={info['ema50']:.4f} "
                        f"vol={info['vol']:.2f} volma={info['volma']:.2f}",
                        flush=True
                    )

                if not (ok_entry and bias.direction in ("LONG", "SHORT")):
                    continue

                now = time.time()
                if now - last_signal_time.get(symbol, 0) < SIGNAL_COOLDOWN_SECONDS:
                    if DEBUG_LOGS:
                        print(f"[{symbol}] cooldown active, skipping", flush=True)
                    continue

                liq = analyze_liquidity_v3(ex, symbol)

                if DEBUG_LOGS:
                    print(
                        f"[{symbol}] liq below={liq.liq_below:.2f} above={liq.liq_above:.2f} "
                        f"dom_ba={liq.dominance_below_over_above:.2f}x "
                        f"dom_ab={liq.dominance_above_over_below:.2f}x "
                        f"near_bid_wall={liq.has_near_bid_wall} near_ask_wall={liq.has_near_ask_wall}",
                        flush=True
                    )

                ok_liq, liq_reason = liquidity_allows_v3(bias.direction, liq)
                if not ok_liq:
                    if DEBUG_LOGS:
                        print(f"[{symbol}] FILTERED: {liq_reason}", flush=True)
                    continue

                text = format_signal(
                    symbol=symbol,
                    direction=bias.direction,
                    candle_ts=str(candle_ts),
                    bias=bias,
                    reason=reason,
                    info=info,
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
