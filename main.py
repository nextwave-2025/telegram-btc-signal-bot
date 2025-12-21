import os
import time
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import ccxt
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


# =========================
# CONFIG
# =========================

LOCAL_TZ = "Europe/Berlin"

TF_ENTRY = "15m"
TF_ZONES = "1h"
TF_BIAS  = "4h"
TF_DAILY = "1d"

EMA_FAST = 20
EMA_SLOW = 50
RSI_LEN  = 14

# Quality > quantity filters
VOL_MA_LEN = 20
VOL_RATIO_MIN_SETUP = 1.15
VOL_RATIO_IF_RSI_EXCEPTION = 1.30

ATR_LEN = 14
ATR_MULT = 1.5

# Candle strength (break/retest confirmation)
MIN_BODY_TO_RANGE = 0.55
MIN_RANGE_ATR_MULT = 0.70

# Retest quality: wick rejection (RELAXED but still quality)
RETEST_MIN_WICK_RATIO = 0.25
RETEST_MAX_BODY_RATIO = 0.80

# NEW: allow "near miss" to count as retest touch (proximity)
RETEST_PROX_PCT = 0.0010  # 0.10% of price

# Micro trend softness (avoid being blocked by tiny EMA flips)
MICRO_TREND_ENABLED = True
EMA_ALIGN_TOL_PCT = 0.001        # 0.10%
EMA_SLOPE_LEN = 3

# RSI filters
RSI_SHORT_MIN = 42.0
RSI_LONG_MAX  = 72.0
QUALITY_LONG_NEEDS_RSI50_CROSS = True

# Entries
ENTRY_PAD_PCT = 0.0006
PULLBACK_PAD_PCT = 0.0012
PULLBACK_VALID_CANDLES = 6

# Risk
MAX_SL_PCT = 0.02
RR_TARGETS = (1, 2, 3)

# Pivot zones
PIV_LEFT = 2
PIV_RIGHT = 2
LOOKBACK_1H = 180
LOOKBACK_4H = 140
ZONE_PAD_PCT_1H = 0.0010
ZONE_PAD_PCT_4H = 0.0012

# =========================
# ZONE FREEZE
# =========================
ZONE_FREEZE_ENABLED = True
ZONE_FREEZE_CANDLES_15M = 8   # 8 * 15m = 2 hours

# =========================
# Telegram env (both naming styles)
# =========================
BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID   = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()

# Scheduled alerts (Europe/Berlin)
ALERT_US_OPENING_TIME = "15:15"
ALERT_US_CLOSING_TIME = "21:45"
DELETE_BOT_MESSAGES_AT = "00:00"

# Runtime
LOOP_SLEEP_SECONDS = 25
RATE_LIMIT_BACKOFF_SECONDS = 60

DEBUG_LOGS = (os.getenv("DEBUG_LOGS") or "0").strip() in ("1", "true", "True")

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


# =========================
# DATA STRUCTURES
# =========================

@dataclass
class Bias:
    direction: str  # LONG/SHORT/NEUTRAL

@dataclass
class Zone:
    low: float
    high: float
    kind: str  # SUP/RES


# =========================
# TELEGRAM
# =========================

def telegram_enabled() -> bool:
    return bool(BOT_TOKEN and CHAT_ID)

def _tg_post(method: str, payload: Dict) -> Dict:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    try:
        return json.loads(body)
    except Exception:
        return {"ok": False, "raw": body}

def send_telegram_html(text: str) -> Optional[int]:
    if not telegram_enabled():
        print("⚠️ Telegram OFF (BOT_TOKEN/CHAT_ID missing)", flush=True)
        return None
    try:
        payload = {
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        res = _tg_post("sendMessage", payload)
        if not res.get("ok"):
            print(f"⚠️ Telegram send failed: {str(res)[:800]}", flush=True)
            return None
        return int(res["result"]["message_id"])
    except Exception as e:
        print(f"⚠️ Telegram error: {type(e).__name__}: {e}", flush=True)
        return None

def delete_telegram_message(message_id: int) -> bool:
    if not telegram_enabled():
        return False
    try:
        res = _tg_post("deleteMessage", {"chat_id": CHAT_ID, "message_id": int(message_id)})
        ok = bool(res.get("ok"))
        if not ok:
            print(f"⚠️ deleteMessage failed (id={message_id}): {str(res)[:300]}", flush=True)
        return ok
    except Exception as e:
        print(f"⚠️ deleteMessage error: {type(e).__name__}: {e}", flush=True)
        return False


# =========================
# TIME HELPERS
# =========================

def local_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ZoneInfo(LOCAL_TZ))

def today_key(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")

def to_local_str(ts_utc: pd.Timestamp) -> str:
    return ts_utc.to_pydatetime().astimezone(ZoneInfo(LOCAL_TZ)).strftime("%Y-%m-%d %H:%M:%S %Z")

def maybe_send_scheduled_alerts(state: Dict[str, str], sent_ids: List[int]):
    now = local_now()
    hhmm = now.strftime("%H:%M")
    day = today_key(now)

    if hhmm == ALERT_US_OPENING_TIME and state.get("us_opening") != day:
        mid = send_telegram_html("⚠️ <b>Achtung:</b> US Opening in 15 Minuten.\nDer Markt kann volatil werden.")
        if mid:
            sent_ids.append(mid)
        state["us_opening"] = day

    if hhmm == ALERT_US_CLOSING_TIME and state.get("us_closing") != day:
        mid = send_telegram_html(
            "⚠️ <b>Achtung:</b> US Closing in 15 Minuten.\n"
            "Danach kein Trade empfohlen, da Market Maker den Preis stark beeinflussen können."
        )
        if mid:
            sent_ids.append(mid)
        state["us_closing"] = day

def maybe_midnight_cleanup(state: Dict[str, str], sent_ids: List[int]):
    now = local_now()
    hhmm = now.strftime("%H:%M")
    day = today_key(now)
    if hhmm != DELETE_BOT_MESSAGES_AT:
        return
    if state.get("midnight_cleanup") == day:
        return

    remaining = []
    deleted = 0
    for mid in list(sent_ids):
        if delete_telegram_message(mid):
            deleted += 1
        else:
            remaining.append(mid)

    sent_ids.clear()
    sent_ids.extend(remaining)
    print(f"🧹 Midnight cleanup: deleted={deleted}, remaining={len(remaining)}", flush=True)
    state["midnight_cleanup"] = day


# =========================
# INDICATORS
# =========================

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def ema_slope_down(e: pd.Series, n: int = 3) -> bool:
    if len(e) < n + 1:
        return False
    return float(e.iloc[-1]) < float(e.iloc[-n])

def ema_slope_up(e: pd.Series, n: int = 3) -> bool:
    if len(e) < n + 1:
        return False
    return float(e.iloc[-1]) > float(e.iloc[-n])


# =========================
# ZONES
# =========================

def _pivots(series: pd.Series, left: int, right: int, mode: str) -> List[Tuple[pd.Timestamp, float]]:
    piv = []
    vals = series.values
    idx = series.index
    n = len(series)
    for i in range(left, n - right):
        window = vals[i-left:i+right+1]
        if mode == "high":
            if vals[i] == np.max(window):
                piv.append((idx[i], float(vals[i])))
        else:
            if vals[i] == np.min(window):
                piv.append((idx[i], float(vals[i])))
    return piv

def best_pivot_zones(df: pd.DataFrame, current_price: float, lookback: int, pad_pct: float) -> Tuple[Optional["Zone"], Optional["Zone"]]:
    if len(df) < lookback + 10:
        return None, None

    d = df.tail(lookback)
    piv_hi = _pivots(d["high"], PIV_LEFT, PIV_RIGHT, "high")
    piv_lo = _pivots(d["low"], PIV_LEFT, PIV_RIGHT, "low")

    res_price = None
    sup_price = None

    for _, ph in piv_hi:
        if ph > current_price:
            if res_price is None or ph < res_price:
                res_price = ph

    for _, pl in piv_lo:
        if pl < current_price:
            if sup_price is None or pl > sup_price:
                sup_price = pl

    pad = current_price * pad_pct
    sup = Zone(sup_price - pad, sup_price + pad, "SUP") if sup_price is not None else None
    res = Zone(res_price - pad, res_price + pad, "RES") if res_price is not None else None
    return sup, res

def fmt_zone(z: Optional[Zone]) -> str:
    if not z:
        return "—"
    if z.high >= 1000:
        return f"{z.low:.2f}–{z.high:.2f}"
    return f"{z.low:.6f}–{z.high:.6f}"


# =========================
# EXCHANGE / FETCH (with caching)
# =========================

def to_bybit_linear(sym: str) -> str:
    return sym if ":" in sym else f"{sym}:USDT"

def make_exchange() -> ccxt.Exchange:
    return ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

def fetch_df(exchange: ccxt.Exchange, symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df

@dataclass
class CacheItem:
    df: pd.DataFrame
    last_fetch: float

def get_df_cached(
    ex: ccxt.Exchange,
    cache: Dict[Tuple[str, str], CacheItem],
    symbol: str,
    tf: str,
    limit: int,
    refresh_seconds: int
) -> pd.DataFrame:
    key = (symbol, tf)
    now = time.time()
    item = cache.get(key)
    if item is None or (now - item.last_fetch) >= refresh_seconds:
        df = fetch_df(ex, symbol, tf, limit=limit)
        cache[key] = CacheItem(df=df, last_fetch=now)
        return df
    return item.df


# =========================
# STRATEGY CORE
# =========================

def compute_bias(df: pd.DataFrame) -> Bias:
    close = df["close"]
    e20 = float(ema(close, EMA_FAST).iloc[-1])
    e50 = float(ema(close, EMA_SLOW).iloc[-1])
    if e20 > e50:
        return Bias("LONG")
    if e20 < e50:
        return Bias("SHORT")
    return Bias("NEUTRAL")

def is_break_event(side: str, close: float, sup: Optional["Zone"], res: Optional["Zone"]) -> Tuple[bool, str, Optional[float]]:
    if side == "LONG":
        if not res:
            return False, "No 1h resistance zone", None
        lvl = float(res.high)
        ok = close > lvl
        if ok:
            return True, f"1h Breakout ✅ (close {close:.6f} über {lvl:.6f})", lvl
        return False, f"1h Breakout warten (close {close:.6f} noch nicht über {lvl:.6f})", lvl

    if side == "SHORT":
        if not sup:
            return False, "No 1h support zone", None
        lvl = float(sup.low)
        ok = close < lvl
        if ok:
            return True, f"1h Breakdown ✅ (close {close:.6f} unter {lvl:.6f})", lvl
        return False, f"1h Breakdown warten (close {close:.6f} noch nicht unter {lvl:.6f})", lvl

    return False, "Invalid side", None

def candle_strength_ok(row: pd.Series, atr_val: float) -> Tuple[bool, str]:
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])

    rng = max(h - l, 1e-12)
    body = abs(c - o)
    body_ratio = body / rng

    if body_ratio < MIN_BODY_TO_RANGE:
        return False, f"Body too small ({body_ratio:.2f} < {MIN_BODY_TO_RANGE})"

    if atr_val and atr_val > 0:
        if rng < atr_val * MIN_RANGE_ATR_MULT:
            return False, "Range too small vs ATR"

    return True, "OK"

def wick_metrics(o: float, h: float, l: float, c: float) -> Tuple[float, float, float, float]:
    rng = max(h - l, 1e-12)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body = abs(c - o)
    return (upper_wick / rng, lower_wick / rng, body / rng, rng)

def retest_event(
    side: str,
    row: pd.Series,
    sup: Optional[Zone],
    res: Optional[Zone],
    last_close: float
) -> Tuple[bool, str, Optional[Tuple[float, float]]]:
    """
    RELAXED Retest (still quality):
      SHORT: candle touches or comes near 1h resistance, closes NOT above zone.high, shows rejection (upper wick)
      LONG : candle touches or comes near 1h support, closes NOT below zone.low, shows rejection (lower wick)
    """
    o = float(row["open"]); h = float(row["high"]); l = float(row["low"]); c = float(row["close"])
    up_w, lo_w, body_r, _ = wick_metrics(o, h, l, c)

    prox = max(last_close * RETEST_PROX_PCT, 1e-12)

    if side == "SHORT":
        if not res:
            return False, "No 1h resistance zone", None

        # Touch OR near zone
        near_or_touch = (h >= res.low) or (abs(res.low - h) <= prox) or (abs(res.low - c) <= prox)
        # Must not close above zone high (avoid real breakout)
        not_breaking = c <= res.high
        # Rejection quality
        wick_ok = up_w >= RETEST_MIN_WICK_RATIO
        body_ok = body_r <= RETEST_MAX_BODY_RATIO
        bearish_or_small = (c <= o) or (body_r <= 0.35)

        if near_or_touch and not_breaking and wick_ok and body_ok and bearish_or_small:
            return True, f"1h Retest SHORT ✅ (Rejection an Resistance {fmt_zone(res)})", (res.low, res.high)

        return False, "1h Retest SHORT warten", (res.low, res.high)

    if side == "LONG":
        if not sup:
            return False, "No 1h support zone", None

        near_or_touch = (l <= sup.high) or (abs(l - sup.high) <= prox) or (abs(c - sup.high) <= prox)
        not_breaking = c >= sup.low
        wick_ok = lo_w >= RETEST_MIN_WICK_RATIO
        body_ok = body_r <= RETEST_MAX_BODY_RATIO
        bullish_or_small = (c >= o) or (body_r <= 0.35)

        if near_or_touch and not_breaking and wick_ok and body_ok and bullish_or_small:
            return True, f"1h Retest LONG ✅ (Bounce an Support {fmt_zone(sup)})", (sup.low, sup.high)

        return False, "1h Retest LONG warten", (sup.low, sup.high)

    return False, "Invalid side", None

def entry_filters_quality(
    side: str,
    c: float,
    e20v: float,
    e50v: float,
    e20_series: pd.Series,
    r15: float,
    r15_prev: float,
    v: float,
    vma: float,
    trigger_is_break: bool
) -> Tuple[bool, str]:
    vol_ratio = (v / vma) if (vma and vma > 0) else 0.0
    if vol_ratio < VOL_RATIO_MIN_SETUP:
        return False, f"Volume too low (v<{VOL_RATIO_MIN_SETUP:.2f}x vma)"

    tol = EMA_ALIGN_TOL_PCT

    if side == "SHORT":
        aligned = (e20v < e50v) or (e20v <= e50v * (1.0 + tol))
        micro = False
        if MICRO_TREND_ENABLED:
            micro = (c < e20v) and ema_slope_down(e20_series, EMA_SLOPE_LEN)

        if not (aligned or micro):
            return False, "15m not bearish yet (waiting pullback to finish)"

        if r15 < RSI_SHORT_MIN:
            if not trigger_is_break:
                return False, f"RSI too low for SHORT ({r15:.2f} < {RSI_SHORT_MIN})"
            if vol_ratio < VOL_RATIO_IF_RSI_EXCEPTION:
                return False, f"RSI low; need stronger volume ({vol_ratio:.2f}x < {VOL_RATIO_IF_RSI_EXCEPTION:.2f}x)"

        return True, "OK"

    if side == "LONG":
        aligned = (e20v > e50v) or (e20v >= e50v * (1.0 - tol))
        micro = False
        if MICRO_TREND_ENABLED:
            micro = (c > e20v) and ema_slope_up(e20_series, EMA_SLOPE_LEN)

        if not (aligned or micro):
            return False, "15m not bullish yet (waiting dip to finish)"

        if QUALITY_LONG_NEEDS_RSI50_CROSS:
            if not (r15_prev <= 50.0 and r15 > 50.0):
                return False, f"RSI momentum not confirmed (need cross >50, prev={r15_prev:.2f}, now={r15:.2f})"
        else:
            if r15 > RSI_LONG_MAX:
                return False, f"RSI too high for LONG ({r15:.2f} > {RSI_LONG_MAX})"

        return True, "OK"

    return False, "Invalid side"


# =========================
# RISK / MESSAGE
# =========================

def entry_range_from_price(px: float) -> Tuple[float, float]:
    low = px * (1 - ENTRY_PAD_PCT)
    high = px * (1 + ENTRY_PAD_PCT)
    return float(low), float(high)

def pullback_zone(level: float) -> Tuple[float, float]:
    pad = level * PULLBACK_PAD_PCT
    return float(level - pad), float(level + pad)

def build_plan_by_riskdist(side: str, entry: float, risk_dist: float) -> Dict:
    cap_dist = entry * MAX_SL_PCT
    risk_dist = min(risk_dist, cap_dist)
    if side == "SHORT":
        sl = entry + risk_dist
        tps = [entry - risk_dist * rr for rr in RR_TARGETS]
    else:
        sl = entry - risk_dist
        tps = [entry + risk_dist * rr for rr in RR_TARGETS]
    return {
        "sl": float(sl),
        "tp1": float(tps[0]),
        "tp2": float(tps[1]),
        "tp3": float(tps[2]),
        "risk_pct": float((risk_dist / entry) * 100.0),
        "crv": "1:2",
    }

def build_plan(side: str, entry_price: float, atr_val: float) -> Dict:
    atr_dist = atr_val * ATR_MULT if atr_val and atr_val > 0 else entry_price * 0.005
    return build_plan_by_riskdist(side, entry_price, atr_dist)

def build_retest_plan(side: str, entry_price: float, zone_low: float, zone_high: float, atr_val: float) -> Dict:
    buffer = (zone_high - zone_low) * 0.25
    if buffer <= 0:
        buffer = entry_price * 0.0015

    if side == "SHORT":
        desired_dist = max((zone_high + buffer) - entry_price, atr_val * 0.9 if atr_val else entry_price * 0.004)
    else:
        desired_dist = max(entry_price - (zone_low - buffer), atr_val * 0.9 if atr_val else entry_price * 0.004)

    return build_plan_by_riskdist(side, entry_price, desired_dist)

def build_setup_message(
    symbol: str,
    side: str,
    ts_close: pd.Timestamp,
    trigger_text: str,
    trigger_kind: str,   # "BREAK" or "RETEST"
    break_level: Optional[float],
    safe_zone: Optional[Tuple[float, float]],
    breakout_row: pd.Series,
    sup_1h: Optional[Zone], res_1h: Optional[Zone],
    sup_4h: Optional[Zone], res_4h: Optional[Zone],
    r15: float, r4h: float, r1d: float,
    vol_ratio: float,
    atrv: float
) -> str:
    side_u = side.upper()
    head = "🟢 <b>LONG</b>" if side_u == "LONG" else "🔴 <b>SHORT</b>"

    c = float(breakout_row["close"])
    pair = symbol.replace(":USDT", "").replace("/", "")
    t_local = to_local_str(ts_close)

    aggressive_entry = c
    ag_low, ag_high = entry_range_from_price(aggressive_entry)

    if trigger_kind == "BREAK" and break_level is not None:
        pb_low, pb_high = pullback_zone(break_level)
        safe_label = f"{pb_low:.6f} – {pb_high:.6f}"
        safe_mid = (pb_low + pb_high) / 2.0
        safe_plan = build_plan(side_u, safe_mid, atrv)
    else:
        if safe_zone is None:
            safe_zone = (aggressive_entry * 0.999, aggressive_entry * 1.001)
        z_low, z_high = safe_zone
        safe_label = f"{z_low:.6f} – {z_high:.6f}"
        safe_mid = (z_low + z_high) / 2.0
        safe_plan = build_retest_plan(side_u, safe_mid, z_low, z_high, atrv)

    ag_plan = build_plan(side_u, aggressive_entry, atrv)

    return (
        f"{head} <b>SETUP</b>\n\n"
        f"📊 <b>Pair:</b> {pair}\n"
        f"🕒 <b>Zeit (15m Close):</b> {t_local}\n"
        f"📌 <b>Trigger:</b> {trigger_text}\n\n"
        f"🚀 <b>Aggressiver Entry (sofort):</b> {aggressive_entry:.6f}\n"
        f"   Range: {ag_low:.6f} – {ag_high:.6f}\n"
        f"   🛑 SL: {ag_plan['sl']:.6f} (Risk {ag_plan['risk_pct']:.2f}%, max {MAX_SL_PCT*100:.0f}%)\n"
        f"   ✅ TP1: {ag_plan['tp1']:.6f}\n"
        f"   ✅ TP2: {ag_plan['tp2']:.6f}  (<b>CRV 1:2</b>)\n"
        f"   ✅ TP3: {ag_plan['tp3']:.6f}\n\n"
        f"🧲 <b>Sicherer Entry (Zone):</b> {safe_label}\n"
        f"   Gültig für die nächsten <b>{PULLBACK_VALID_CANDLES}</b> Kerzen (~{PULLBACK_VALID_CANDLES*15} Min)\n"
        f"   🛑 SL: {safe_plan['sl']:.6f} (Risk {safe_plan['risk_pct']:.2f}%, max {MAX_SL_PCT*100:.0f}%)\n"
        f"   ✅ TP1: {safe_plan['tp1']:.6f}\n"
        f"   ✅ TP2: {safe_plan['tp2']:.6f}  (<b>CRV 1:2</b>)\n"
        f"   ✅ TP3: {safe_plan['tp3']:.6f}\n\n"
        f"📍 <b>RSI:</b> 15m={r15:.2f} | 4h={r4h:.2f} | 1d={r1d:.2f}\n"
        f"📦 <b>Vol:</b> Ratio={vol_ratio:.2f}x (vs VolMA{VOL_MA_LEN})\n\n"
        f"🧱 <b>1h Zones (frozen):</b>\n"
        f"Support: {fmt_zone(sup_1h)}\n"
        f"Resistance: {fmt_zone(res_1h)}\n\n"
        f"🗺️ <b>4h Zones:</b>\n"
        f"Support: {fmt_zone(sup_4h)}\n"
        f"Resistance: {fmt_zone(res_4h)}\n\n"
        f"<b>©️ Copyright by crypto_mistik.</b>\n"
        f"⚠️ Kein Financial Advice"
    )


# =========================
# ZONE FREEZE STATE
# =========================

@dataclass
class FrozenZones:
    sup_1h: Optional[Zone]
    res_1h: Optional[Zone]
    remaining_15m: int

def maybe_update_frozen_zones(
    frozen: Dict[str, FrozenZones],
    symbol: str,
    df1h: pd.DataFrame,
    current_price: float
) -> FrozenZones:
    if not ZONE_FREEZE_ENABLED:
        sup, res = best_pivot_zones(df1h, current_price=current_price, lookback=LOOKBACK_1H, pad_pct=ZONE_PAD_PCT_1H)
        return FrozenZones(sup, res, remaining_15m=0)

    f = frozen.get(symbol)
    if f is None or f.remaining_15m <= 0:
        sup, res = best_pivot_zones(df1h, current_price=current_price, lookback=LOOKBACK_1H, pad_pct=ZONE_PAD_PCT_1H)
        f = FrozenZones(sup, res, remaining_15m=ZONE_FREEZE_CANDLES_15M)
        frozen[symbol] = f
        return f

    return f

def decrement_freeze(frozen: Dict[str, FrozenZones], symbol: str):
    if not ZONE_FREEZE_ENABLED:
        return
    f = frozen.get(symbol)
    if not f:
        return
    f.remaining_15m -= 1
    if f.remaining_15m < 0:
        f.remaining_15m = 0


# =========================
# MAIN
# =========================

def main():
    ex = make_exchange()

    symbols = [to_bybit_linear(s) for s in SYMBOLS]
    markets = ex.load_markets()
    symbols = [s for s in symbols if s in markets and markets[s].get("active", True)]

    print(
        f"✅ BOT AKTIV – ZoneFreeze={'ON' if ZONE_FREEZE_ENABLED else 'OFF'}({ZONE_FREEZE_CANDLES_15M}x15m) | "
        f"Bias=4h | Trigger=1hBreak OR 1hRetest | MicroTrend={'ON' if MICRO_TREND_ENABLED else 'OFF'} | "
        f"Telegram={'ON' if telegram_enabled() else 'OFF'}",
        flush=True
    )
    print(f"✅ Symbols: {symbols}", flush=True)

    last_processed_close: Dict[str, pd.Timestamp] = {}
    alert_state: Dict[str, str] = {}
    sent_bot_message_ids: List[int] = []

    df_cache: Dict[Tuple[str, str], CacheItem] = {}
    frozen_zones: Dict[str, FrozenZones] = {}

    refresh_15m = 35
    refresh_1h  = 180
    refresh_4h  = 240
    refresh_1d  = 600

    while True:
        try:
            maybe_send_scheduled_alerts(alert_state, sent_bot_message_ids)
            maybe_midnight_cleanup(alert_state, sent_bot_message_ids)

            for symbol in symbols:
                df15_raw = get_df_cached(ex, df_cache, symbol, TF_ENTRY, limit=360, refresh_seconds=refresh_15m)
                if len(df15_raw) < 160:
                    continue

                ts_close = df15_raw.index[-2]
                if last_processed_close.get(symbol) == ts_close:
                    continue
                last_processed_close[symbol] = ts_close

                df15_closed = df15_raw.iloc[:-1].copy()
                last_close = float(df15_closed["close"].iloc[-1])

                df1h = get_df_cached(ex, df_cache, symbol, TF_ZONES, limit=300, refresh_seconds=refresh_1h)
                df4h = get_df_cached(ex, df_cache, symbol, TF_BIAS,  limit=280, refresh_seconds=refresh_4h)
                df1d = get_df_cached(ex, df_cache, symbol, TF_DAILY, limit=240, refresh_seconds=refresh_1d)

                if len(df1h) < 160 or len(df4h) < 160 or len(df1d) < 120:
                    decrement_freeze(frozen_zones, symbol)
                    continue

                close_series = df15_closed["close"]
                rsi15_series = rsi(close_series, RSI_LEN)
                r15 = float(rsi15_series.iloc[-1])
                r15_prev = float(rsi15_series.iloc[-2])

                r4h = float(rsi(df4h["close"], RSI_LEN).iloc[-1])
                r1d = float(rsi(df1d["close"], RSI_LEN).iloc[-1])

                e20_series = ema(close_series, EMA_FAST)
                e50_series = ema(close_series, EMA_SLOW)
                e20v = float(e20_series.iloc[-1])
                e50v = float(e50_series.iloc[-1])

                volma_series = df15_closed["volume"].rolling(VOL_MA_LEN).mean()
                v = float(df15_closed["volume"].iloc[-1])
                vma = float(volma_series.iloc[-1]) if not np.isnan(volma_series.iloc[-1]) else float(df15_closed["volume"].tail(VOL_MA_LEN).mean())
                vol_ratio = (v / vma) if (vma and vma > 0) else 0.0

                atr_series = atr(df15_closed, ATR_LEN)
                atrv = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0

                bias_4h = compute_bias(df4h)
                if bias_4h.direction not in ("LONG", "SHORT"):
                    print(f"[{symbol}] close={ts_close} bias=NEUTRAL (4h neutral)", flush=True)
                    decrement_freeze(frozen_zones, symbol)
                    continue

                side = bias_4h.direction

                fz = maybe_update_frozen_zones(frozen_zones, symbol, df1h, current_price=last_close)
                sup_1h, res_1h = fz.sup_1h, fz.res_1h
                sup_4h, res_4h = best_pivot_zones(df4h, current_price=last_close, lookback=LOOKBACK_4H, pad_pct=ZONE_PAD_PCT_4H)

                trigger_row = df15_raw.iloc[-2]

                break_ok, break_reason, break_level = is_break_event(side, last_close, sup_1h, res_1h)

                ret_ok, ret_reason, ret_safe_zone = retest_event(side, trigger_row, sup_1h, res_1h, last_close)

                if not break_ok and not ret_ok:
                    print(
                        f"[{symbol}] close={ts_close} bias={side} trigger=NO "
                        f"(break: {break_reason} | retest: {ret_reason}) rsi15={r15:.2f}",
                        flush=True
                    )
                    decrement_freeze(frozen_zones, symbol)
                    continue

                trigger_kind = "BREAK" if break_ok else "RETEST"
                trigger_text = break_reason if break_ok else ret_reason
                safe_zone = None if break_ok else ret_safe_zone

                ok, why = entry_filters_quality(
                    side, last_close,
                    e20v, e50v,
                    e20_series,
                    r15, r15_prev,
                    v, vma,
                    trigger_is_break=bool(break_ok)
                )
                if not ok:
                    print(f"[{symbol}] close={ts_close} bias={side} setup=NO ({why}) rsi15={r15:.2f}", flush=True)
                    decrement_freeze(frozen_zones, symbol)
                    continue

                strong_ok, strong_why = candle_strength_ok(trigger_row, atrv)
                if not strong_ok:
                    print(f"[{symbol}] close={ts_close} bias={side} setup=NO ({strong_why}) rsi15={r15:.2f}", flush=True)
                    decrement_freeze(frozen_zones, symbol)
                    continue

                msg = build_setup_message(
                    symbol=symbol,
                    side=side,
                    ts_close=ts_close,
                    trigger_text=trigger_text,
                    trigger_kind=trigger_kind,
                    break_level=break_level if break_ok else None,
                    safe_zone=safe_zone,
                    breakout_row=trigger_row,
                    sup_1h=sup_1h, res_1h=res_1h,
                    sup_4h=sup_4h, res_4h=res_4h,
                    r15=r15, r4h=r4h, r1d=r1d,
                    vol_ratio=vol_ratio,
                    atrv=atrv
                )

                print("\n" + msg + "\n", flush=True)
                mid = send_telegram_html(msg)
                if mid:
                    sent_bot_message_ids.append(mid)

                decrement_freeze(frozen_zones, symbol)
                time.sleep(0.12)

            time.sleep(LOOP_SLEEP_SECONDS)

        except ccxt.RateLimitExceeded as e:
            print(f"⚠️ BOT ERROR: RateLimitExceeded: {e}", flush=True)
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
        except Exception as e:
            print(f"⚠️ BOT ERROR: {type(e).__name__}: {e}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
