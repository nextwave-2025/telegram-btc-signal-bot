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

DEBUG_LOGS = True
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT", "DOGE/USDT"]

ENTRY_TF = "15m"   # entry timeframe (Candle close only)
BIAS_TF = "4h"     # bias timeframe

EMA_FAST = 20
EMA_SLOW = 50

RSI_UPPER = 64
RSI_LOWER = 34

VOL_MA_LEN = 20

SIGNAL_COOLDOWN_SECONDS = 60 * 30  # 30 min
POLL_SECONDS = 30

EXCHANGE_ID = "bybit"
MARKET_TYPE = "linear"  # USDT Perps on Bybit

# --- Orderbook / Liquidity V3 ---
ORDERBOOK_LIMIT = 200
LIQ_LOOKAROUND_PCT = 0.008     # analyze +/-0.8% around mid
LIQ_BIN_PCT = 0.0015           # cluster bin width ~0.15% of price
LIQ_WALL_MULTIPLIER = 2.2      # wall if bin_size > avg_bin_size * multiplier

# Multiple snapshots to filter fake walls
OB_SNAPSHOTS = 3
OB_SNAPSHOT_DELAY_SEC = 0.6
WALL_PERSIST_MIN = 2           # wall must appear in >=2 snapshots
WALL_PRICE_TOLERANCE_BINS = 1  # match walls across snapshots within +-1 bin

# Direction rules
# LONG: need bids dominate + persistent bid wall BELOW near price
# SHORT: need asks dominate + persistent ask wall ABOVE near price
DOMINANCE_RATIO_NEAR_WALL = 1.15  # if near wall exists, dominance requirement is easier
DOMINANCE_RATIO_NO_WALL = 1.40    # if no near wall, be stricter
MAX_WALL_DISTANCE_PCT = 0.005     # near wall must be within 0.5% in the direction
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
# SIGNAL LOGIC (Core)
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
# LIQUIDITY CLUSTER V3
# =========================

@dataclass
class Wall:
    price: float
    size: float
    distance_pct: float  # distance from mid in %

@dataclass
class LiquidityInfo:
    mid: float
    liq_below: float
    liq_above: float
    dominance_below_over_above: float
    dominance_above_over_below: float
    bid_walls_below: List[Wall]   # persistent walls below mid
    ask_walls_above: List[Wall]   # persistent walls above mid
    has_near_bid_wall: bool
    has_near_ask_wall: bool
    lookaround_pct: float
    bin_pct: float
    snapshots: int
    wall_persist_min: int

def _bin_size(mid: float) -> float:
    return mid * LIQ_BIN_PCT

def _bounds(mid: float) -> Tuple[float, float]:
    lo = mid * (1.0 - LIQ_LOOKAROUND_PCT)
    hi = mid * (1.0 + LIQ_LOOKAROUND_PCT)
    return lo, hi

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

def _sum_liquidity_in_band(bids, asks, mid: float, lo: float, hi: float) -> Tuple[float, float]:
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

    return liq_below, liq_above

def _extract_wall_bins(bins: Dict[int, float], wall_multiplier: float) -> List[Tuple[int, float]]:
    sizes = list(bins.values())
    if not sizes:
        return []
    avg = float(np.mean(sizes))
    if avg <= 0:
        return []
    out = [(idx, float(sz)) for idx, sz in bins.items() if float(sz) >= avg * wall_multiplier]
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def _merge_persistent_walls(wall_candidates_per_snapshot: List[List[Tuple[int, float]]]) -> Dict[int, float]:
    """
    Match walls across snapshots by bin index tolerance.
    Returns: representative_bin_idx -> aggregated_size (max size observed)
    """
    persistent: Dict[int, Dict[str, float]] = {}  # idx -> {"count": c, "max": m}
    for snapshot_walls in wall_candidates_per_snapshot:
        used = set()
        for idx, sz in snapshot_walls:
            # try match existing within tolerance
            match_idx = None
            for existing_idx in persistent.keys():
                if abs(existing_idx - idx) <= WALL_PRICE_TOLERANCE_BINS and existing_idx not in used:
                    match_idx = existing_idx
                    break
            if match_idx is None:
                persistent[idx] = {"count": 1.0, "max": float(sz)}
                used.add(idx)
            else:
                persistent[match_idx]["count"] += 1.0
                persistent[match_idx]["max"] = max(persistent[match_idx]["max"], float(sz))
                used.add(match_idx)

    # filter by persistence
    out: Dict[int, float] = {}
    for idx, meta in persistent.items():
        if int(meta["count"]) >= WALL_PERSIST_MIN:
            out[idx] = float(meta["max"])
    return out

def analyze_liquidity_v3(ex, symbol: str) -> LiquidityInfo:
    # Take multiple snapshots to reduce fake walls
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

    # Use last snapshot mid as reference
    mid, bids, asks = snapshots[-1]
    lo, hi = _bounds(mid)
    bin_size = _bin_size(mid)

    # Total liquidity in band (last snapshot)
    liq_below, liq_above = _sum_liquidity_in_band(bids, asks, mid, lo, hi)
    dominance_ba = (liq_below / liq_above) if liq_above > 0 else float("inf")
    dominance_ab = (liq_above / liq_below) if liq_below > 0 else float("inf")

    # Build bins & wall candidates per snapshot
    bid_wall_candidates = []
    ask_wall_candidates = []
    for (m, b, a) in snapshots:
        # keep same lo/hi/bin_size based on latest mid to avoid drift issues
        bid_bins = _bin_levels(b, lo, hi, bin_size)
        ask_bins = _bin_levels(a, lo, hi, bin_size)

        bid_wall_candidates.append(_extract_wall_bins(bid_bins, LIQ_WALL_MULTIPLIER))
        ask_wall_candidates.append(_extract_wall_bins(ask_bins, LIQ_WALL_MULTIPLIER))

    persistent_bid = _merge_persistent_walls(bid_wall_candidates)
    persistent_ask = _merge_persistent_walls(ask_wall_candidates)

    # Convert persistent bins to walls with distance filters
    bid_walls_below: List[Wall] = []
    for idx, sz in persistent_bid.items():
        price = _bin_center(idx, lo, bin_size)
        if price >= mid:
            continue
        dist_pct = (mid - price) / mid
        bid_walls_below.append(Wall(price=float(price), size=float(sz), distance_pct=float(dist_pct)))
    bid_walls_below.sort(key=lambda w: w.size, reverse=True)

    ask_walls_above: List[Wall] = []
    for idx, sz in persistent_ask.items():
        price = _bin_center(idx, lo, bin_size)
        if price <= mid:
            continue
        dist_pct = (price - mid) / mid
        ask_walls_above.append(Wall(price=float(price), size=float(sz), distance_pct=float(dist_pct)))
    ask_walls_above.sort(key=lambda w: w.size, reverse=True)

    has_near_bid_wall = any(w.distance_pct <= MAX_WALL_DISTANCE_PCT for w in bid_walls_below)
    has_near_ask_wall = any(w.distance_pct <= MAX_WALL_DISTANCE_PCT for w in ask_walls_above)

    return LiquidityInfo(
        mid=float(mid),
        liq_below=float(liq_below),
        liq_above=float(liq_above),
        dominance_below_over_above=float(dominance_ba),
        dominance_above_over_below=float(dominance_ab),
        bid_walls_below=bid_walls_below[:TOP_WALLS_TO_REPORT],
        ask_walls_above=ask_walls_above[:TOP_WALLS_TO_REPORT],
        has_near_bid_wall=has_near_bid_wall,
        has_near_ask_wall=has_near_ask_wall,
        lookaround_pct=LIQ_LOOKAROUND_PCT,
        bin_pct=LIQ_BIN_PCT,
        snapshots=OB_SNAPSHOTS,
        wall_persist_min=WALL_PERSIST_MIN,
    )

def liquidity_allows_v3(direction: str, liq: LiquidityInfo) -> Tuple[bool, str]:
    if direction == "LONG":
        required = DOMINANCE_RATIO_NEAR_WALL if liq.has_near_bid_wall else DOMINANCE_RATIO_NO_WALL
        if liq.dominance_below_over_above >= required:
            if liq.has_near_bid_wall:
                return True, f"Liquidity OK LONG: bids dominate {liq.dominance_below_over_above:.2f}x + near bid wall"
            return True, f"Liquidity OK LONG: bids dominate {liq.dominance_below_over_above:.2f}x"
        return False, f"Liquidity BLOCK LONG: bids/asks {liq.dominance_below_over_above:.2f}x (<{required}x)"

    if direction == "SHORT":
        required = DOMINANCE_RATIO_NEAR_WALL if liq.has_near_ask_wall else DOMINANCE_RATIO_NO_WALL
        if liq.dominance_above_over_below >= required:
            if liq.has_near_ask_wall:
                return True, f"Liquidity OK SHORT: asks dominate {liq.dominance_above_over_below:.2f}x + near ask wall"
            return True, f"Liquidity OK SHORT: asks dominate {liq.dominance_above_over_below:.2f}x"
        return False, f"Liquidity BLOCK SHORT: asks/bids {liq.dominance_above_over_below:.2f}x (<{required}x)"

    return False, "Liquidity: neutral direction"


# =========================
# TELEGRAM
# =========================

def _format_walls(walls: List[Wall], side_label: str) -> str:
    if not walls:
        return f"{side_label}: none\n"
    lines = []
    for w in walls:
        lines.append(f"{w.price:.4f} ({w.size:.2f}, {w.distance_pct*100:.2f}%)")
    return f"{side_label}: " + " | ".join(lines) + "\n"

def format_signal(symbol: str, direction: str, reason: str, info: Dict, bias: Bias, candle_ts: str,
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
        f"🌊 Liquidity V3 (Bybit L2, ±{liq.lookaround_pct*100:.2f}%, snapshots={liq.snapshots}, persist≥{liq.wall_persist_min}):\n"
        f"Below: {liq.liq_below:.2f} | Above: {liq.liq_above:.2f}\n"
        f"{_format_walls(liq.bid_walls_below, '🧱 Bid walls below')}"
        f"{_format_walls(liq.ask_walls_above, '🧱 Ask walls above')}"
        f"✅ Liquidity: {liq_reason}\n\n"
        f"✅ Setup: {reason}\n"
        f"⚠️ Hinweis: V3 (Fake-Wall Filter aktiv; News/SR folgen als nächster Schritt)"
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
    chat_id = (os.getenv("CHAT_ID") or "").strip()

    if not bot_token or not chat_id:
        raise RuntimeError("Missing BOT_TOKEN or CHAT_ID in Railway Variables")

    bot = Bot(token=bot_token)
    ex = make_exchange()

    last_seen_candle: Dict[str, pd.Timestamp] = {}
    last_signal_time: Dict[str, float] = {}

    try:
        await send_telegram(bot, chat_id, "✅ Signal-Bot gestartet (V3). 15m Close + 4h Bias + Liquidity V3 (Fake-Wall Filter) aktiv.")
    except Exception as e:
        print(f"Startup Telegram failed: {type(e).__name__}: {e}", flush=True)

   while True:
    try:
        for symbol in SYMBOLS:
            print(f"Checking {symbol}", flush=True)

            ohlcv_15m = ex.fetch_ohlcv(symbol, timeframe=ENTRY_TF, limit=220)
            ohlcv_4h = ex.fetch_ohlcv(symbol, timeframe=BIAS_TF, limit=220)

            df15 = to_df(ohlcv_15m)
            df4h = to_df(ohlcv_4h)

            candle_ts = last_closed_candle_ts(df15)

            if symbol in last_seen_candle and candle_ts == last_seen_candle[symbol]:
                continue
            last_seen_candle[symbol] = candle_ts

            bias = compute_bias(df4h)
            ok_entry, reason, info = check_entry(df15, bias.direction)

            if DEBUG_LOGS:
                print(
                    f"[{symbol}] entry_ok={ok_entry} bias={bias.direction} "
                    f"rsi={info['rsi']:.2f} "
                    f"ema20={info['ema20']:.4f} ema50={info['ema50']:.4f} "
                    f"vol={info['vol']:.2f} volma={info['volma']:.2f}",
                    flush=True
                )

            if not (ok_entry and bias.direction in ("LONG", "SHORT")):
                continue

            now = time.time()
            if now - last_signal_time.get(symbol, 0) < SIGNAL_COOLDOWN_SECONDS:
                continue

            liq = analyze_liquidity_v3(ex, symbol)

            if DEBUG_LOGS:
                print(
                    f"[{symbol}] liq below={liq.liq_below:.2f} above={liq.liq_above:.2f} "
                    f"dom_ba={liq.dominance_below_over_above:.2f}x "
                    f"near_bid_wall={liq.has_near_bid_wall} "
                    f"near_ask_wall={liq.has_near_ask_wall}",
                    flush=True
                )

            ok_liq, liq_reason = liquidity_allows_v3(bias.direction, liq)
            if not ok_liq:
                print(f"[{symbol}] FILTERED: {liq_reason}", flush=True)
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






