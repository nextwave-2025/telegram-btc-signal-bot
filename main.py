import os
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import ccxt
from telegram import Bot


# =========================
# CONFIG
# =========================

DEBUG_LOGS = True

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT", "DOGE/USDT", "XRP/USDT", "HYPE/USDT", "FARTCOIN/USDT", "1000PEPE/USDT"]

ENTRY_TF = "15m"   # candle close only
BIAS_TF = "4h"     # bias + zones

EMA_FAST = 20
EMA_SLOW = 50
RSI_UPPER = 64
RSI_LOWER = 34
VOL_MA_LEN = 20

SIGNAL_COOLDOWN_SECONDS = 60 * 30
POLL_SECONDS = 30

EXCHANGE_ID = "bybit"
MARKET_TYPE = "linear"

# ---- Liquidity V3 ----
ORDERBOOK_LIMIT = 200
LIQ_LOOKAROUND_PCT = 0.008
LIQ_BIN_PCT = 0.0015
LIQ_WALL_MULTIPLIER = 2.2

OB_SNAPSHOTS = 3
OB_SNAPSHOT_DELAY_SEC = 0.6
WALL_PERSIST_MIN = 2
WALL_PRICE_TOLERANCE_BINS = 1

DOMINANCE_RATIO_NEAR_WALL = 1.15
DOMINANCE_RATIO_NO_WALL = 1.40
MAX_WALL_DISTANCE_PCT = 0.005
TOP_WALLS_TO_REPORT = 3

# ---- S/R Zones (4h) ----
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
ZONE_CLUSTER_PCT = 0.0018      # 0.18%
MIN_ZONE_TOUCHES = 3
NEAR_ZONE_PCT = 0.0015         # "near zone" ~0.15%

# ---- Break rules ----
REQUIRE_VOL_ON_BREAK = True
ALLOW_BREAK_ON_NEUTRAL_BIAS = True  # allow break signals even if 4h EMA bias is neutral


# =========================
# INDICATORS
# =========================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
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
    direction: str  # LONG / SHORT / NEUTRAL
    ema_fast: float
    ema_slow: float


def compute_bias(df_4h: pd.DataFrame) -> Bias:
    close = df_4h["close"]
    e20 = float(ema(close, EMA_FAST).iloc[-1])
    e50 = float(ema(close, EMA_SLOW).iloc[-1])
    if e20 > e50:
        d = "LONG"
    elif e20 < e50:
        d = "SHORT"
    else:
        d = "NEUTRAL"
    return Bias(direction=d, ema_fast=e20, ema_slow=e50)


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

    # --- REALISTIC volume gate for your current vol/volma behavior ---
    VOL_RATIO = 0.12  # 0.08 = mehr Trades | 0.15 = strenger
    vol_ratio = (v / vma) if vma > 0 else 0.0
    vol_ok = v >= vma * VOL_RATIO

    info = {
        "close": c, "ema20": e20, "ema50": e50, "rsi": r,
        "vol": v, "volma": vma, "vol_ratio": vol_ratio
    }

    # LONG: Trend up + RSI not overheated + volume ok
    if bias_dir == "LONG":
        rsi_ok = (r < RSI_UPPER)      # statt (34<r<64) -> weniger Blockaden
        ok = (c > e20) and (e20 > e50) and rsi_ok and vol_ok
        reason = "15m confirms LONG" if ok else f"LONG blocked (rsi_ok={rsi_ok}, vol_ratio={vol_ratio:.2f})"
        return ok, reason, info

    # SHORT: Trend down + RSI not oversold + volume ok
    if bias_dir == "SHORT":
        rsi_ok = (r > RSI_LOWER)      # statt (34<r<64) -> BTC/SOL blocken nicht mehr wegen RSI>64
        ok = (c < e20) and (e20 < e50) and rsi_ok and vol_ok
        reason = "15m confirms SHORT" if ok else f"SHORT blocked (rsi_ok={rsi_ok}, vol_ratio={vol_ratio:.2f})"
        return ok, reason, info

    return False, "Bias neutral", info





# =========================
# SUPPORT / RESISTANCE ZONES (4h)
# =========================

def find_pivots(df: pd.DataFrame, left: int, right: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    lows = df["low"].values
    highs = df["high"].values
    piv_lows: List[Tuple[int, float]] = []
    piv_highs: List[Tuple[int, float]] = []

    for i in range(left, len(df) - right):
        if lows[i] == np.min(lows[i - left:i + right + 1]):
            piv_lows.append((i, float(lows[i])))
        if highs[i] == np.max(highs[i - left:i + right + 1]):
            piv_highs.append((i, float(highs[i])))
    return piv_lows, piv_highs


def build_zones(levels: List[float], price: float, cluster_pct: float) -> List[Tuple[float, float, int]]:
    if not levels:
        return []
    levels = sorted(levels)
    threshold = price * cluster_pct

    zones: List[Tuple[float, float, int]] = []
    cur = [levels[0]]
    for lv in levels[1:]:
        if abs(lv - cur[-1]) <= threshold:
            cur.append(lv)
        else:
            zones.append((min(cur), max(cur), len(cur)))
            cur = [lv]
    zones.append((min(cur), max(cur), len(cur)))
    zones.sort(key=lambda z: z[2], reverse=True)  # by touches
    return zones


def zone_strength_rejections(df: pd.DataFrame, zl: float, zh: float, is_support: bool) -> int:
    closes = df["close"].values
    lows = df["low"].values
    highs = df["high"].values
    rej = 0
    for i in range(len(df)):
        if is_support:
            touched = lows[i] <= zh and lows[i] >= zl * 0.995
            if touched and closes[i] > zh:
                rej += 1
        else:
            touched = highs[i] >= zl and highs[i] <= zh * 1.005
            if touched and closes[i] < zl:
                rej += 1
    return rej


@dataclass
class Zone:
    low: float
    high: float
    touches: int
    rejections: int
    strength: float
    kind: str  # SUPPORT or RESISTANCE


def best_zones_4h(df4h: pd.DataFrame, current_price: float) -> Tuple[Optional[Zone], Optional[Zone]]:
    piv_lows, piv_highs = find_pivots(df4h, PIVOT_LEFT, PIVOT_RIGHT)
    low_levels = [p for _, p in piv_lows][-250:]
    high_levels = [p for _, p in piv_highs][-250:]

    support_z = build_zones(low_levels, current_price, ZONE_CLUSTER_PCT)
    resist_z = build_zones(high_levels, current_price, ZONE_CLUSTER_PCT)

    sup_best: Optional[Zone] = None
    res_best: Optional[Zone] = None

    df_tail = df4h.tail(400)

    for zl, zh, touches in support_z:
        if touches < MIN_ZONE_TOUCHES:
            continue
        rej = zone_strength_rejections(df_tail, zl, zh, is_support=True)
        strength = touches + 1.5 * rej
        sup_best = Zone(low=zl, high=zh, touches=touches, rejections=rej, strength=strength, kind="SUPPORT")
        break

    for zl, zh, touches in resist_z:
        if touches < MIN_ZONE_TOUCHES:
            continue
        rej = zone_strength_rejections(df_tail, zl, zh, is_support=False)
        strength = touches + 1.5 * rej
        res_best = Zone(low=zl, high=zh, touches=touches, rejections=rej, strength=strength, kind="RESISTANCE")
        break

    return sup_best, res_best


def is_near_zone(price: float, z: Zone, near_pct: float) -> bool:
    band = price * near_pct
    return (z.low - band) <= price <= (z.high + band)


# =========================
# LIQUIDITY V3 (Orderbook)
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
    persistent: Dict[int, Dict[str, float]] = {}
    for snapshot_walls in walls_per_snapshot:
        used = set()
        for idx, sz in snapshot_walls:
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

    out: Dict[int, float] = {}
    for idx, meta in persistent.items():
        if int(meta["count"]) >= WALL_PERSIST_MIN:
            out[idx] = float(meta["max"])
    return out


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
    bin_size = _bin_size(mid)

    liq_below, liq_above = _sum_liquidity_in_band(bids, asks, mid, lo, hi)
    dom_ba = (liq_below / liq_above) if liq_above > 0 else float("inf")
    dom_ab = (liq_above / liq_below) if liq_below > 0 else float("inf")

    bid_wall_candidates = []
    ask_wall_candidates = []
    for _, b, a in snapshots:
        bid_bins = _bin_levels(b, lo, hi, bin_size)
        ask_bins = _bin_levels(a, lo, hi, bin_size)
        bid_wall_candidates.append(_extract_wall_bins(bid_bins))
        ask_wall_candidates.append(_extract_wall_bins(ask_bins))

    persistent_bid = _merge_persistent_walls(bid_wall_candidates)
    persistent_ask = _merge_persistent_walls(ask_wall_candidates)

    bid_walls_below: List[Wall] = []
    for idx, sz in persistent_bid.items():
        price = _bin_center(idx, lo, bin_size)
        if price >= mid:
            continue
        dist_pct = (mid - price) / mid
        bid_walls_below.append(Wall(price=float(price), size=float(sz), distance_pct=float(dist_pct)))
    bid_walls_below.sort(key=lambda w: w.size, reverse=True)
    bid_walls_below = bid_walls_below[:TOP_WALLS_TO_REPORT]

    ask_walls_above: List[Wall] = []
    for idx, sz in persistent_ask.items():
        price = _bin_center(idx, lo, bin_size)
        if price <= mid:
            continue
        dist_pct = (price - mid) / mid
        ask_walls_above.append(Wall(price=float(price), size=float(sz), distance_pct=float(dist_pct)))
    ask_walls_above.sort(key=lambda w: w.size, reverse=True)
    ask_walls_above = ask_walls_above[:TOP_WALLS_TO_REPORT]

    has_near_bid_wall = any(w.distance_pct <= MAX_WALL_DISTANCE_PCT for w in bid_walls_below)
    has_near_ask_wall = any(w.distance_pct <= MAX_WALL_DISTANCE_PCT for w in ask_walls_above)

    return LiquidityInfo(
        mid=float(mid),
        liq_below=float(liq_below),
        liq_above=float(liq_above),
        dominance_below_over_above=float(dom_ba),
        dominance_above_over_below=float(dom_ab),
        bid_walls_below=bid_walls_below,
        ask_walls_above=ask_walls_above,
        has_near_bid_wall=has_near_bid_wall,
        has_near_ask_wall=has_near_ask_wall,
    )


def liquidity_allows_v3(direction: str, liq: LiquidityInfo) -> Tuple[bool, str]:
    if direction == "LONG":
        required = DOMINANCE_RATIO_NEAR_WALL if liq.has_near_bid_wall else DOMINANCE_RATIO_NO_WALL
        if liq.dominance_below_over_above >= required:
            extra = " + near bid wall" if liq.has_near_bid_wall else ""
            return True, f"Liquidity OK LONG: bids/asks={liq.dominance_below_over_above:.2f}x (req {required}x){extra}"
        return False, f"Liquidity BLOCK LONG: bids/asks={liq.dominance_below_over_above:.2f}x (<{required}x)"

    if direction == "SHORT":
        required = DOMINANCE_RATIO_NEAR_WALL if liq.has_near_ask_wall else DOMINANCE_RATIO_NO_WALL
        if liq.dominance_above_over_below >= required:
            extra = " + near ask wall" if liq.has_near_ask_wall else ""
            return True, f"Liquidity OK SHORT: asks/bids={liq.dominance_above_over_below:.2f}x (req {required}x){extra}"
        return False, f"Liquidity BLOCK SHORT: asks/bids={liq.dominance_above_over_below:.2f}x (<{required}x)"

    return False, "Liquidity: neutral direction"


# =========================
# TELEGRAM
# =========================

def _walls_line(label: str, walls: List[Wall]) -> str:
    if not walls:
        return f"{label}: none"
    parts = [f"{w.price:.4f} ({w.size:.2f}, {w.distance_pct*100:.2f}%)" for w in walls]
    return f"{label}: " + " | ".join(parts)


def _zone_line(z: Optional[Zone]) -> str:
    if not z:
        return "none"
    return f"{z.low:.4f}–{z.high:.4f} (touches={z.touches}, rej={z.rejections}, strength={z.strength:.1f})"


def format_signal(symbol: str, title: str, info: Dict, bias: Bias, candle_ts: str,
                  liq_reason: str, liq: LiquidityInfo,
                  sup: Optional[Zone], res: Optional[Zone],
                  extra: str) -> str:
    return (
        f"🚨 {title}\n\n"
        f"📊 Pair: {symbol.replace('/', '')}\n"
        f"🕒 Entry TF: {ENTRY_TF} (Candle Close)\n"
        f"🧭 Bias TF: {BIAS_TF} ({bias.direction})\n"
        f"🕯️ Candle: {candle_ts}\n\n"
        f"💰 Close: {info['close']:.4f}\n"
        f"📈 EMA{EMA_FAST}: {info['ema20']:.4f}\n"
        f"📉 EMA{EMA_SLOW}: {info['ema50']:.4f}\n"
        f"📍 RSI(14): {info['rsi']:.2f} (Bands {RSI_LOWER}/{RSI_UPPER})\n"
        f"📦 Vol: {info['vol']:.2f} | VolMA({VOL_MA_LEN}): {info['volma']:.2f}\n\n"
        f"🧱 4h Zones:\n"
        f"Support: {_zone_line(sup)}\n"
        f"Resistance: {_zone_line(res)}\n\n"
        f"🌊 Liquidity V3 (Bybit L2, ±{LIQ_LOOKAROUND_PCT*100:.2f}%, snap={OB_SNAPSHOTS}, persist≥{WALL_PERSIST_MIN})\n"
        f"Below: {liq.liq_below:.2f} | Above: {liq.liq_above:.2f}\n"
        f"{_walls_line('🧱 Bid walls below', liq.bid_walls_below)}\n"
        f"{_walls_line('🧱 Ask walls above', liq.ask_walls_above)}\n"
        f"✅ Liquidity: {liq_reason}\n\n"
        f"{extra}\n"
        f"⚠️ Hinweis: Automatisches Signal (kein Financial Advice)"
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
        raise RuntimeError("Missing BOT_TOKEN or CHAT_ID (Railway Variables)")

    bot = Bot(token=bot_token)
    ex = make_exchange()

    last_seen_candle: Dict[str, pd.Timestamp] = {}
    last_signal_time: Dict[str, float] = {}

    print("Bot booting…", flush=True)
    try:
        await send_telegram(bot, chat_id, "✅ Signal-Bot gestartet (V4.1: Support-Breakdown + Resistance-Breakout).")
        print("Startup message sent ✅", flush=True)
    except Exception as e:
        print(f"Startup Telegram failed: {type(e).__name__}: {e}", flush=True)

    while True:
        try:
            for symbol in SYMBOLS:
                if DEBUG_LOGS:
                    print(f"Checking {symbol}", flush=True)

                ohlcv_15m = ex.fetch_ohlcv(symbol, timeframe=ENTRY_TF, limit=220)
                ohlcv_4h = ex.fetch_ohlcv(symbol, timeframe=BIAS_TF, limit=500)

                df15 = to_df(ohlcv_15m)
                df4h = to_df(ohlcv_4h)

                candle_ts = last_closed_candle_ts(df15)

                # Candle close only
                if symbol in last_seen_candle and candle_ts == last_seen_candle[symbol]:
                    continue
                last_seen_candle[symbol] = candle_ts

                bias = compute_bias(df4h)
                last_price = float(df15["close"].iloc[-1])

                sup, res = best_zones_4h(df4h, current_price=last_price)
                ok_entry, entry_reason, info = check_entry(df15, bias.direction)

                if DEBUG_LOGS:
                    print(
                        f"[{symbol}] candle={candle_ts} bias={bias.direction} entry_ok={ok_entry} "
                        f"close={info['close']:.4f} rsi={info['rsi']:.2f} "
                        f"vol={info['vol']:.2f} volma={info['volma']:.2f} vol_ratio={info.get('vol_ratio', 0):.2f} "
                        f"sup={_zone_line(sup)} res={_zone_line(res)}",
                        flush=True
                    )

                # Cooldown per symbol
                now = time.time()
                if now - last_signal_time.get(symbol, 0) < SIGNAL_COOLDOWN_SECONDS:
                    continue

                # Liquidity snapshot
                liq = analyze_liquidity_v3(ex, symbol)

                if DEBUG_LOGS:
                    print(
                        f"[{symbol}] liq dom_ba={liq.dominance_below_over_above:.2f}x dom_ab={liq.dominance_above_over_below:.2f}x "
                        f"near_bid_wall={liq.has_near_bid_wall} near_ask_wall={liq.has_near_ask_wall}",
                        flush=True
                    )

                vol_ok = True
                if REQUIRE_VOL_ON_BREAK:
                    vol_ok = float(info["vol"]) > float(info["volma"])

                allow_break = (bias.direction in ("LONG", "SHORT")) or (ALLOW_BREAK_ON_NEUTRAL_BIAS and bias.direction == "NEUTRAL")

                # ==========================
                # 1) SUPPORT BREAKDOWN -> SHORT
                # ==========================
                support_break = False
                if sup is not None:
                    support_break = float(info["close"]) < sup.low

                if support_break and vol_ok and allow_break and (bias.direction in ("SHORT", "NEUTRAL")):
                    ok_liq, liq_reason = liquidity_allows_v3("SHORT", liq)
                    if ok_liq:
                        extra = (
                            "🔻 SUPPORT BREAKDOWN:\n"
                            f"- 15m Close unter Support ({sup.low:.4f}–{sup.high:.4f})\n"
                            f"- Volumen {'OK' if vol_ok else 'NICHT OK'} (vol > VolMA)\n"
                            "- Erwartung: erhöhte Dump-Wahrscheinlichkeit / Liquidity unten\n"
                        )
                        text = format_signal(
                            symbol=symbol,
                            title="BREAKDOWN SHORT (Support gebrochen)",
                            info=info,
                            bias=bias,
                            candle_ts=str(candle_ts),
                            liq_reason=liq_reason,
                            liq=liq,
                            sup=sup,
                            res=res,
                            extra=extra,
                        )
                        await send_telegram(bot, chat_id, text)
                        last_signal_time[symbol] = now
                        continue

                # ==========================
                # 2) RESISTANCE BREAKOUT -> LONG (NEU)
                # ==========================
                resistance_break = False
                if res is not None:
                    resistance_break = float(info["close"]) > res.high

                if resistance_break and vol_ok and allow_break and (bias.direction in ("LONG", "NEUTRAL")):
                    ok_liq, liq_reason = liquidity_allows_v3("LONG", liq)
                    if ok_liq:
                        extra = (
                            "🔺 RESISTANCE BREAKOUT:\n"
                            f"- 15m Close über Resistance ({res.low:.4f}–{res.high:.4f})\n"
                            f"- Volumen {'OK' if vol_ok else 'NICHT OK'} (vol > VolMA)\n"
                            "- Erwartung: erhöhte Pump-/Squeeze-Wahrscheinlichkeit / Liquidity oben\n"
                        )
                        text = format_signal(
                            symbol=symbol,
                            title="BREAKOUT LONG (Resistance gebrochen)",
                            info=info,
                            bias=bias,
                            candle_ts=str(candle_ts),
                            liq_reason=liq_reason,
                            liq=liq,
                            sup=sup,
                            res=res,
                            extra=extra,
                        )
                        await send_telegram(bot, chat_id, text)
                        last_signal_time[symbol] = now
                        continue

                # ==========================
                # 3) Normal entries with zone gating
                # ==========================
                if not (ok_entry and bias.direction in ("LONG", "SHORT")):
                    continue

                if bias.direction == "LONG" and sup is not None:
                    if not is_near_zone(float(info["close"]), sup, NEAR_ZONE_PCT):
                        continue

                if bias.direction == "SHORT" and res is not None:
                    if not is_near_zone(float(info["close"]), res, NEAR_ZONE_PCT):
                        continue

                ok_liq, liq_reason = liquidity_allows_v3(bias.direction, liq)
                if not ok_liq:
                    continue

                extra = f"✅ Setup: {entry_reason}\n✅ Zone-Filter: {'Support' if bias.direction=='LONG' else 'Resistance'} nahe"
                text = format_signal(
                    symbol=symbol,
                    title=f"SIGNAL ({bias.direction})",
                    info=info,
                    bias=bias,
                    candle_ts=str(candle_ts),
                    liq_reason=liq_reason,
                    liq=liq,
                    sup=sup,
                    res=res,
                    extra=extra,
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




