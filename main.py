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
TF_SETUP = "1h"
TF_BIAS = "4h"
TF_DAILY = "1d"

EMA_FAST = 20
EMA_SLOW = 50
RSI_LEN = 14

# QUALITY
QUALITY_LONG_NEEDS_RSI50_CROSS = True  # 15m RSI must cross above 50 for LONG

# RSI filters
RSI_SHORT_MIN = 42.0     # blocks late shorts (oversold protection)
RSI_LONG_MAX = 72.0

# “Resume” thresholds (used only in strict resume)
RESUME_SHORT_MAX_RSI = 60.0
RESUME_LONG_MIN_RSI = 40.0

# NEW: Micro-trend mode (fixes your logs)
MICRO_TREND_ENABLED = True
# SHORT micro trend = close < EMA20 and EMA20 slope down
# LONG  micro trend = close > EMA20 and EMA20 slope up

# Reversal mode (optional watch)
REVERSAL_MODE = True
REV_RSI_15M_MAX = 30.0
REV_RSI_4H_MAX  = 35.0
REV_RSI_1D_MAX  = 45.0

# Volume
VOL_MA_LEN = 20
VOL_RATIO_BASE = 1.00
VOL_RATIO_BREAK_OVERRIDE = 1.20

# Risk
ATR_LEN = 14
ATR_MULT = 1.5
MAX_SL_PCT = 0.02
ENTRY_PAD_PCT = 0.0006
RR_TARGETS = (1, 2, 3)

# Fakeout rules (3 rules)
FAKEOUT_ENABLED = True

# Pivot zones settings
PIV_LEFT = 2
PIV_RIGHT = 2

LOOKBACK_1H = 180
ZONE_PAD_PCT_1H = 0.0010

LOOKBACK_4H = 140
ZONE_PAD_PCT_4H = 0.0012

EMA_SLOPE_LEN = 5

# Telegram env (supports both naming styles)
BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()

# Scheduled alerts (Europe/Berlin)
ALERT_US_OPENING_TIME = "15:15"
ALERT_US_CLOSING_TIME = "21:45"

# Midnight cleanup (delete bot messages)
DELETE_BOT_MESSAGES_AT = "00:00"

# Rate limit protection
LOOP_SLEEP_SECONDS = 25
RATE_LIMIT_BACKOFF_SECONDS = 60

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

@dataclass
class Pending:
    symbol: str
    side: str                 # LONG/SHORT
    stage: str                # WAIT_CONFIRM -> WAIT_PULLBACK -> WAIT_ENTRY
    tag: str                  # TREND/REVERSAL

    breakout_ts: pd.Timestamp
    breakout_high: float
    breakout_low: float
    breakout_close: float
    breakout_vol: float

    # zones
    sup_1h: Optional[Zone]
    res_1h: Optional[Zone]
    sup_4h: Optional[Zone]
    res_4h: Optional[Zone]

    confirm_ts: pd.Timestamp
    pullback_ts: Optional[pd.Timestamp] = None
    entry_open_ts: Optional[pd.Timestamp] = None

    rsi_15m: float = 0.0
    rsi_4h: float = 0.0
    rsi_1d: float = 0.0

    ema20: float = 0.0
    ema50: float = 0.0
    volma: float = 0.0
    atr: float = 0.0

    bias_1h: str = ""
    bias_4h: str = ""


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

    if not sent_ids:
        state["midnight_cleanup"] = day
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

def ema_slope_down(e: pd.Series, n: int = 5) -> bool:
    if len(e) < n + 1:
        return False
    return float(e.iloc[-1]) < float(e.iloc[-n])

def ema_slope_up(e: pd.Series, n: int = 5) -> bool:
    if len(e) < n + 1:
        return False
    return float(e.iloc[-1]) > float(e.iloc[-n])


# =========================
# PIVOT ZONES
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

def best_pivot_zones(df: pd.DataFrame, current_price: float, lookback: int, pad_pct: float) -> Tuple[Optional[Zone], Optional[Zone]]:
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

def is_break_event(side: str, close: float, sup: Optional[Zone], res: Optional[Zone]) -> Tuple[bool, str]:
    # break against 1h zones (setup levels)
    if side == "LONG":
        if not res:
            return False, "No 1h resistance zone"
        return (close > res.high), f"1h Breakout (close>{res.high:.6f})"
    else:
        if not sup:
            return False, "No 1h support zone"
        return (close < sup.low), f"1h Breakdown (close<{sup.low:.6f})"


def entry_filters_quality(
    side: str,
    c: float,
    e20v: float,
    e50v: float,
    e20_series: pd.Series,
    e50_series: pd.Series,
    r15: float,
    r15_prev: float,
    v: float,
    vma: float,
    break_event: bool
) -> Tuple[bool, str]:
    """
    Fix for your logs:
    - If 4h bias is SHORT but 15m EMAs not aligned, we still allow SHORT when "micro-trend" is bearish:
      close < EMA20 and EMA20 slope down.
    - Symmetric for LONG.
    - Still requires 1h break + confirm + pullback later (so quality stays high).
    """
    vol_ok = (vma > 0) and (v >= vma * VOL_RATIO_BASE)

    if side == "SHORT":
        aligned = (e20v < e50v)

        micro_bearish = False
        if MICRO_TREND_ENABLED:
            micro_bearish = (c < e20v) and ema_slope_down(e20_series, EMA_SLOPE_LEN)

        resume_short_strict = (
            (not aligned)
            and break_event
            and (c < min(e20v, e50v))
            and ema_slope_down(e20_series, EMA_SLOPE_LEN)
            and ema_slope_down(e50_series, EMA_SLOPE_LEN)
            and (r15 < RESUME_SHORT_MAX_RSI)
        )

        if not (aligned or micro_bearish or resume_short_strict):
            return False, "15m not bearish yet (waiting pullback to finish)"

        # oversold protection (avoid late shorts)
        if r15 < RSI_SHORT_MIN:
            if not break_event:
                return False, f"RSI too low for SHORT ({r15:.2f} < {RSI_SHORT_MIN})"
            if not (vma > 0 and v >= vma * VOL_RATIO_BREAK_OVERRIDE):
                return False, f"RSI low; need strong vol (v<{VOL_RATIO_BREAK_OVERRIDE}x vma)"

        if not vol_ok:
            return False, f"Volume too low (v<{VOL_RATIO_BASE}x vma)"
        return True, "OK"

    if side == "LONG":
        aligned = (e20v > e50v)

        micro_bullish = False
        if MICRO_TREND_ENABLED:
            micro_bullish = (c > e20v) and ema_slope_up(e20_series, EMA_SLOPE_LEN)

        resume_long_strict = (
            (not aligned)
            and break_event
            and (c > max(e20v, e50v))
            and ema_slope_up(e20_series, EMA_SLOPE_LEN)
            and ema_slope_up(e50_series, EMA_SLOPE_LEN)
            and (r15 > RESUME_LONG_MIN_RSI)
        )

        if not (aligned or micro_bullish or resume_long_strict):
            return False, "15m not bullish yet (waiting dip to finish)"

        if QUALITY_LONG_NEEDS_RSI50_CROSS:
            rsi_cross_up = (r15_prev <= 50.0 and r15 > 50.0)
            if not rsi_cross_up:
                return False, f"RSI momentum not confirmed (need cross >50, prev={r15_prev:.2f}, now={r15:.2f})"
        else:
            if r15 > RSI_LONG_MAX:
                return False, f"RSI too high for LONG ({r15:.2f} > {RSI_LONG_MAX})"

        if not vol_ok:
            return False, f"Volume too low (v<{VOL_RATIO_BASE}x vma)"
        return True, "OK"

    return False, "Invalid side"


# =========================
# FAKEOUT RULES (3 rules)
# =========================

def confirm_rule2(side: str, breakout_high: float, breakout_low: float, confirm_close: float) -> bool:
    if side == "LONG":
        return confirm_close > breakout_high
    return confirm_close < breakout_low

def is_pullback(side: str, breakout_high: float, breakout_low: float, pull_low: float, pull_high: float) -> bool:
    if side == "LONG":
        return pull_low <= breakout_high
    return pull_high >= breakout_low

def pullback_rule1_volume(pull_vol: float, breakout_vol: float) -> bool:
    return pull_vol <= breakout_vol


# =========================
# RISK / MESSAGE
# =========================

def build_plan(side: str, entry_price: float, atr_val: float) -> Dict:
    atr_dist = atr_val * ATR_MULT if atr_val and atr_val > 0 else entry_price * 0.005
    cap_dist = entry_price * MAX_SL_PCT
    risk_dist = min(atr_dist, cap_dist)

    if side.upper() == "SHORT":
        sl = entry_price + risk_dist
        tps = [entry_price - risk_dist * rr for rr in RR_TARGETS]
    else:
        sl = entry_price - risk_dist
        tps = [entry_price + risk_dist * rr for rr in RR_TARGETS]

    return {
        "sl": float(sl),
        "tp1": float(tps[0]),
        "tp2": float(tps[1]),
        "tp3": float(tps[2]),
        "risk_pct": float((risk_dist / entry_price) * 100.0),
        "crv": "1:2",
    }

def entry_range_from_price(px: float) -> Tuple[float, float]:
    low = px * (1 - ENTRY_PAD_PCT)
    high = px * (1 + ENTRY_PAD_PCT)
    return float(low), float(high)

def fmt_zone(z: Optional[Zone]) -> str:
    if not z:
        return "—"
    return f"{z.low:.6f}–{z.high:.6f}"

def to_local(ts: pd.Timestamp) -> str:
    return ts.to_pydatetime().astimezone(ZoneInfo(LOCAL_TZ)).strftime("%Y-%m-%d %H:%M:%S %Z")

def build_entry_html(p: Pending, entry_price: float, entry_open_ts: pd.Timestamp) -> str:
    side = p.side.upper()
    head = "🟢 <b>LONG</b>" if side == "LONG" else "🔴 <b>SHORT</b>"
    plan = build_plan(side, entry_price, p.atr)
    entry_low, entry_high = entry_range_from_price(p.breakout_close)

    pair = p.symbol.replace(":USDT", "").replace("/", "")
    t0 = to_local(p.breakout_ts)
    t1 = to_local(p.confirm_ts)
    t2 = to_local(p.pullback_ts) if p.pullback_ts else "—"
    t3 = to_local(entry_open_ts)

    return (
        f"{head} <b>ENTRY</b> (TREND | 1h-Setup + 15m-Trigger ✅)\n\n"
        f"📊 <b>Pair:</b> {pair}\n"
        f"🕒 <b>t0 Break Close (15m):</b> {t0}\n"
        f"✅ <b>t1 Confirm Close:</b> {t1}\n"
        f"✅ <b>t2 Pullback Close:</b> {t2}\n"
        f"🚀 <b>t3 Entry Open:</b> {t3}\n\n"
        f"🎯 <b>Entry Range:</b> {entry_low:.6f} – {entry_high:.6f}\n"
        f"✅ <b>Entry:</b> {entry_price:.6f}\n"
        f"🛑 <b>SL:</b> {plan['sl']:.6f} (Risk {plan['risk_pct']:.2f}%, max {MAX_SL_PCT*100:.0f}%)\n"
        f"✅ <b>TP1:</b> {plan['tp1']:.6f}\n"
        f"✅ <b>TP2:</b> {plan['tp2']:.6f}\n"
        f"✅ <b>TP3:</b> {plan['tp3']:.6f}\n"
        f"📌 <b>CRV (TP2):</b> {plan['crv']}\n\n"
        f"📍 <b>RSI:</b> 15m={p.rsi_15m:.2f} | 4h={p.rsi_4h:.2f} | 1d={p.rsi_1d:.2f}\n"
        f"📈 <b>Bias:</b> 1h={p.bias_1h} | 4h={p.bias_4h} (master)\n\n"
        f"🧱 <b>1h Zones (Trigger):</b>\n"
        f"Support: {fmt_zone(p.sup_1h)}\n"
        f"Resistance: {fmt_zone(p.res_1h)}\n\n"
        f"🗺️ <b>4h Zones (Map):</b>\n"
        f"Support: {fmt_zone(p.sup_4h)}\n"
        f"Resistance: {fmt_zone(p.res_4h)}\n\n"
        f"<b>©️ Copyright by crypto_mistik.</b>\n"
        f"⚠️ Kein Financial Advice"
    )


# =========================
# MAIN
# =========================

def main():
    ex = make_exchange()

    symbols = [to_bybit_linear(s) for s in SYMBOLS]
    markets = ex.load_markets()
    symbols = [s for s in symbols if s in markets and markets[s].get("active", True)]

    print(
        f"✅ BOT AKTIV – Fakeout={'ON' if FAKEOUT_ENABLED else 'OFF'} | "
        f"Bias=4h | Trigger=1hZones+15mBreak | MicroTrend={'ON' if MICRO_TREND_ENABLED else 'OFF'} | "
        f"Telegram={'ON' if telegram_enabled() else 'OFF'}",
        flush=True
    )
    print(f"✅ Symbols: {symbols}", flush=True)

    last_processed_close: Dict[str, pd.Timestamp] = {}
    pending: Dict[str, Pending] = {}
    alert_state: Dict[str, str] = {}
    sent_bot_message_ids: List[int] = []

    while True:
        try:
            maybe_send_scheduled_alerts(alert_state, sent_bot_message_ids)
            maybe_midnight_cleanup(alert_state, sent_bot_message_ids)

            for symbol in symbols:
                df15_raw = fetch_df(ex, symbol, TF_ENTRY, limit=320)
                df1h = fetch_df(ex, symbol, TF_SETUP, limit=260)
                df4h = fetch_df(ex, symbol, TF_BIAS, limit=260)
                df1d = fetch_df(ex, symbol, TF_DAILY, limit=220)

                if len(df15_raw) < 120 or len(df1h) < 140 or len(df4h) < 140:
                    continue

                ts_open = df15_raw.index[-1]
                ts_close = df15_raw.index[-2]

                # =========================
                # pending pipeline
                # =========================
                if symbol in pending:
                    p = pending[symbol]

                    if p.stage == "WAIT_CONFIRM" and ts_close == p.confirm_ts:
                        confirm_close = float(df15_raw["close"].iloc[-2])
                        ok_confirm = (not FAKEOUT_ENABLED) or confirm_rule2(p.side, p.breakout_high, p.breakout_low, confirm_close)
                        if not ok_confirm:
                            print(f"[{symbol}] Rule2 FAIL (confirm)", flush=True)
                            del pending[symbol]
                        else:
                            p.stage = "WAIT_PULLBACK"
                            p.pullback_ts = ts_open

                    if symbol in pending and pending[symbol].stage == "WAIT_PULLBACK":
                        p2 = pending[symbol]
                        if p2.pullback_ts is not None and ts_close == p2.pullback_ts:
                            pull_row = df15_raw.iloc[-2]
                            pull_low = float(pull_row["low"])
                            pull_high = float(pull_row["high"])
                            pull_vol = float(pull_row["volume"])

                            ok_pull = is_pullback(p2.side, p2.breakout_high, p2.breakout_low, pull_low, pull_high)
                            ok_vol = (not FAKEOUT_ENABLED) or pullback_rule1_volume(pull_vol, p2.breakout_vol)

                            if not ok_pull:
                                print(f"[{symbol}] Pullback FAIL", flush=True)
                                del pending[symbol]
                            elif not ok_vol:
                                print(f"[{symbol}] Rule1 FAIL (pullback vol)", flush=True)
                                del pending[symbol]
                            else:
                                p2.stage = "WAIT_ENTRY"
                                p2.entry_open_ts = ts_open

                    if symbol in pending and pending[symbol].stage == "WAIT_ENTRY":
                        p3 = pending[symbol]
                        if p3.entry_open_ts is not None and ts_open == p3.entry_open_ts:
                            open_price = float(df15_raw["open"].iloc[-1])
                            entry_low, entry_high = entry_range_from_price(p3.breakout_close)

                            in_range = entry_low <= open_price <= entry_high
                            plan_tmp = build_plan(p3.side, (entry_low + entry_high) / 2.0, p3.atr)
                            sl = plan_tmp["sl"]
                            invalid = (open_price >= sl) if p3.side.upper() == "SHORT" else (open_price <= sl)

                            if invalid:
                                print(f"[{symbol}] Entry invalid at open", flush=True)
                                del pending[symbol]
                            elif not in_range:
                                print(f"[{symbol}] Entry open not in range (open={open_price:.6f})", flush=True)
                                del pending[symbol]
                            else:
                                msg = build_entry_html(p3, entry_price=open_price, entry_open_ts=ts_open)
                                print("\n" + msg + "\n", flush=True)
                                mid = send_telegram_html(msg)
                                if mid:
                                    sent_bot_message_ids.append(mid)
                                del pending[symbol]

                # =========================
                # new setup only once per 15m close
                # =========================
                if last_processed_close.get(symbol) == ts_close:
                    continue
                last_processed_close[symbol] = ts_close

                df15_closed = df15_raw.iloc[:-1].copy()
                close_series = df15_closed["close"]

                c = float(close_series.iloc[-1])
                v = float(df15_closed["volume"].iloc[-1])

                volma_series = df15_closed["volume"].rolling(VOL_MA_LEN).mean()
                vma = float(volma_series.iloc[-1]) if not np.isnan(volma_series.iloc[-1]) else float(df15_closed["volume"].tail(VOL_MA_LEN).mean())

                rsi15_series = rsi(close_series, RSI_LEN)
                r15 = float(rsi15_series.iloc[-1])
                r15_prev = float(rsi15_series.iloc[-2])

                r4h = float(rsi(df4h["close"], RSI_LEN).iloc[-1])
                r1d = float(rsi(df1d["close"], RSI_LEN).iloc[-1])

                e20_series = ema(close_series, EMA_FAST)
                e50_series = ema(close_series, EMA_SLOW)
                e20v = float(e20_series.iloc[-1])
                e50v = float(e50_series.iloc[-1])

                atr_series = atr(df15_closed, ATR_LEN)
                atrv = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0

                bias_1h = compute_bias(df1h)
                bias_4h = compute_bias(df4h)

                if bias_4h.direction not in ("LONG", "SHORT"):
                    print(f"[{symbol}] close={ts_close} bias=NEUTRAL (4h neutral)", flush=True)
                    continue

                side = bias_4h.direction

                # 1h trigger zones
                sup_1h, res_1h = best_pivot_zones(df1h, current_price=c, lookback=LOOKBACK_1H, pad_pct=ZONE_PAD_PCT_1H)
                # 4h map zones
                sup_4h, res_4h = best_pivot_zones(df4h, current_price=c, lookback=LOOKBACK_4H, pad_pct=ZONE_PAD_PCT_4H)

                break_ok, break_reason = is_break_event(side, c, sup_1h, res_1h)

                ok, why = entry_filters_quality(
                    side, c,
                    e20v, e50v,
                    e20_series, e50_series,
                    r15, r15_prev,
                    v, vma,
                    break_ok
                )

                if not ok:
                    print(f"[{symbol}] close={ts_close} bias={side} setup=NO ({why}) rsi15={r15:.2f}", flush=True)
                    continue

                if not break_ok:
                    print(f"[{symbol}] close={ts_close} bias={side} break=NO ({break_reason}) rsi15={r15:.2f}", flush=True)
                    continue

                if symbol not in pending:
                    row = df15_raw.iloc[-2]
                    pending[symbol] = Pending(
                        symbol=symbol,
                        side=side,
                        stage="WAIT_CONFIRM",
                        tag="TREND",
                        breakout_ts=ts_close,
                        breakout_high=float(row["high"]),
                        breakout_low=float(row["low"]),
                        breakout_close=float(row["close"]),
                        breakout_vol=float(row["volume"]),
                        sup_1h=sup_1h,
                        res_1h=res_1h,
                        sup_4h=sup_4h,
                        res_4h=res_4h,
                        confirm_ts=ts_open,
                        rsi_15m=r15,
                        rsi_4h=r4h,
                        rsi_1d=r1d,
                        ema20=e20v,
                        ema50=e50v,
                        volma=vma,
                        atr=atrv,
                        bias_1h=bias_1h.direction,
                        bias_4h=bias_4h.direction
                    )
                    print(f"[{symbol}] ✅ Pending TREND: {side} | {break_reason} | wait CONFIRM at {ts_open}", flush=True)

                time.sleep(0.2)

            time.sleep(LOOP_SLEEP_SECONDS)

        except ccxt.RateLimitExceeded as e:
            print(f"⚠️ BOT ERROR: RateLimitExceeded: {e}", flush=True)
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
        except Exception as e:
            print(f"⚠️ BOT ERROR: {type(e).__name__}: {e}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
