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

TIMEFRAME_ENTRY = "15m"
TIMEFRAME_BIAS_1H = "1h"
TIMEFRAME_BIAS_4H = "4h"
LOCAL_TZ = "Europe/Berlin"

EMA_FAST = 20
EMA_SLOW = 50
RSI_LEN = 14

# RSI tuning (more realistic)
RSI_SHORT_MIN = 42.0
RSI_LONG_MAX = 62.0  # slightly relaxed

# Volume tuning
VOL_MA_LEN = 20
VOL_RATIO_BASE = 0.10           # mild baseline
VOL_RATIO_BREAK_OVERRIDE = 1.0  # if RSI is low, require strong vol on breakdown

ATR_LEN = 14
ATR_MULT = 1.5
MAX_SL_PCT = 0.02  # max 2%

ENTRY_PAD_PCT = 0.0006
RR_TARGETS = (1, 2, 3)

# Fakeout rules
FAKEOUT_ENABLED = True

# Pivot zone building (better than min/max)
PIVOT_LOOKBACK_4H = 140
PIVOT_LEFT = 2
PIVOT_RIGHT = 2
ZONE_PAD_PCT = 0.0012  # zone thickness around pivot

# Telegram ENV (supports both naming styles)
BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()

# Scheduled alerts (Europe/Berlin)
ALERT_US_OPENING_TIME = "15:15"
ALERT_US_CLOSING_TIME = "21:45"

# Midnight cleanup (delete bot messages)
DELETE_BOT_MESSAGES_AT = "00:00"

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
    direction: str  # "LONG" / "SHORT" / "NEUTRAL"

@dataclass
class Zone:
    low: float
    high: float
    kind: str  # "SUP" or "RES"

@dataclass
class Pending:
    symbol: str
    side: str                 # "LONG" / "SHORT"
    stage: str                # "WAIT_CONFIRM" -> "WAIT_PULLBACK" -> "WAIT_ENTRY"

    breakout_ts: pd.Timestamp
    breakout_high: float
    breakout_low: float
    breakout_close: float
    breakout_vol: float

    sup: Optional[Zone]
    res: Optional[Zone]

    confirm_ts: pd.Timestamp
    pullback_ts: Optional[pd.Timestamp] = None
    entry_open_ts: Optional[pd.Timestamp] = None

    rsi: float = 0.0
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
    """
    Sends message and returns message_id if successful.
    """
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
        msg_id = res["result"]["message_id"]
        return int(msg_id)
    except Exception as e:
        print(f"⚠️ Telegram error: {type(e).__name__}: {e}", flush=True)
        return None

def delete_telegram_message(message_id: int) -> bool:
    """
    Deletes one message. Only bot messages will be in our list anyway.
    Might require admin rights in groups/channels.
    """
    if not telegram_enabled():
        return False
    try:
        res = _tg_post("deleteMessage", {"chat_id": CHAT_ID, "message_id": int(message_id)})
        ok = bool(res.get("ok"))
        if not ok:
            # common if bot has no rights; keep log short
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
    """
    At 00:00 local time, delete all bot-sent messages we tracked.
    This will NOT touch user posts, because we only store bot message_ids.
    Runs once per day.
    """
    now = local_now()
    hhmm = now.strftime("%H:%M")
    day = today_key(now)

    if hhmm != DELETE_BOT_MESSAGES_AT:
        return

    if state.get("midnight_cleanup") == day:
        return  # already done today

    if not sent_ids:
        state["midnight_cleanup"] = day
        return

    # try delete all; keep list of those not deleted (e.g., no permissions)
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
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


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

def best_pivot_zones_4h(df4h: pd.DataFrame, current_price: float) -> Tuple[Optional[Zone], Optional[Zone]]:
    if len(df4h) < PIVOT_LOOKBACK_4H + 10:
        return None, None

    d = df4h.tail(PIVOT_LOOKBACK_4H)
    piv_hi = _pivots(d["high"], PIVOT_LEFT, PIVOT_RIGHT, "high")
    piv_lo = _pivots(d["low"], PIVOT_LEFT, PIVOT_RIGHT, "low")

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

    pad = current_price * ZONE_PAD_PCT
    sup = Zone(sup_price - pad, sup_price + pad, "SUP") if sup_price is not None else None
    res = Zone(res_price - pad, res_price + pad, "RES") if res_price is not None else None
    return sup, res


# =========================
# STRATEGY
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

def combine_bias(b1h: Bias, b4h: Bias) -> Bias:
    return Bias(b4h.direction if b4h.direction in ("LONG", "SHORT") else "NEUTRAL")

def is_break_event(side: str, close: float, sup: Optional[Zone], res: Optional[Zone]) -> Tuple[bool, str]:
    if side == "LONG":
        if not res:
            return False, "No resistance zone"
        return (close > res.high), f"Breakout (close>{res.high:.4f})"
    else:
        if not sup:
            return False, "No support zone"
        return (close < sup.low), f"Breakdown (close<{sup.low:.4f})"

def entry_filters(side: str, c: float, e20v: float, e50v: float, r: float, v: float, vma: float,
                  break_event: bool) -> Tuple[bool, str]:
    vol_ok = (vma > 0) and (v >= vma * VOL_RATIO_BASE)

    if side == "SHORT":
        if not (e20v < e50v):
            return False, "EMA trend not SHORT (e20>=e50)"
        in_pullback_band = (min(e20v, e50v) <= c <= max(e20v, e50v))
        below_fast = c < e20v
        if not (below_fast or in_pullback_band):
            return False, "Price not in SHORT zone (below e20 or between e20/e50)"

        if r < RSI_SHORT_MIN:
            if not break_event:
                return False, f"RSI too low for SHORT ({r:.2f} < {RSI_SHORT_MIN})"
            if not (vma > 0 and v >= vma * VOL_RATIO_BREAK_OVERRIDE):
                return False, f"RSI low; need strong vol for breakdown (vol<{VOL_RATIO_BREAK_OVERRIDE}x vma)"

        if not vol_ok:
            return False, "Volume too low"
        return True, "OK"

    if side == "LONG":
        if not (e20v > e50v):
            return False, "EMA trend not LONG (e20<=e50)"
        in_pullback_band = (min(e20v, e50v) <= c <= max(e20v, e50v))
        above_fast = c > e20v
        if not (above_fast or in_pullback_band):
            return False, "Price not in LONG zone (above e20 or between e20/e50)"
        if r > RSI_LONG_MAX:
            return False, f"RSI too high for LONG ({r:.2f} > {RSI_LONG_MAX})"
        if not vol_ok:
            return False, "Volume too low"
        return True, "OK"

    return False, "Invalid side"

def entry_range_from_price(px: float) -> Tuple[float, float]:
    low = px * (1 - ENTRY_PAD_PCT)
    high = px * (1 + ENTRY_PAD_PCT)
    return float(low), float(high)

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
# EXCHANGE / DATA
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


# =========================
# MESSAGE
# =========================

def fmt_zone(z: Optional[Zone]) -> str:
    if not z:
        return "—"
    return f"{z.low:.4f}–{z.high:.4f}"

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
        f"{head} <b>ENTRY</b> (Fakeout-Filter ✅)\n\n"
        f"📊 <b>Pair:</b> {pair}\n"
        f"🕒 <b>t0 Break Close:</b> {t0}\n"
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
        f"📍 <b>RSI({RSI_LEN}):</b> {p.rsi:.2f}\n"
        f"📈 <b>Bias:</b> 1h={p.bias_1h} | 4h={p.bias_4h} (master)\n\n"
        f"🧱 <b>4h Zones:</b>\n"
        f"Support: {fmt_zone(p.sup)}\n"
        f"Resistance: {fmt_zone(p.res)}\n\n"
        f"<b>©️ Copyright by crypto_mistik.</b>\n"
        f"⚠️ Kein Financial Advice"
    )


# =========================
# MAIN LOOP
# =========================

def main():
    ex = make_exchange()

    symbols = [to_bybit_linear(s) for s in SYMBOLS]
    markets = ex.load_markets()
    symbols = [s for s in symbols if s in markets and markets[s].get("active", True)]

    print(f"✅ BOT AKTIV – Fakeout-Regeln ON | Telegram={'ON' if telegram_enabled() else 'OFF'}", flush=True)
    print(f"✅ Symbols: {symbols}", flush=True)

    last_processed_close: Dict[str, pd.Timestamp] = {}
    pending: Dict[str, Pending] = {}
    alert_state: Dict[str, str] = {}

    # store only bot-sent message_ids (signals + alerts)
    sent_bot_message_ids: List[int] = []

    while True:
        try:
            # scheduled alerts + midnight delete
            maybe_send_scheduled_alerts(alert_state, sent_bot_message_ids)
            maybe_midnight_cleanup(alert_state, sent_bot_message_ids)

            for symbol in symbols:
                df15_raw = fetch_df(ex, symbol, TIMEFRAME_ENTRY, limit=300)
                df1h = fetch_df(ex, symbol, TIMEFRAME_BIAS_1H, limit=300)
                df4h = fetch_df(ex, symbol, TIMEFRAME_BIAS_4H, limit=300)

                if len(df15_raw) < 50 or len(df4h) < 80:
                    continue

                ts_open = df15_raw.index[-1]
                ts_close = df15_raw.index[-2]

                # ---- handle pending ----
                if symbol in pending:
                    p = pending[symbol]

                    if p.stage == "WAIT_CONFIRM" and ts_close == p.confirm_ts:
                        confirm_close = float(df15_raw["close"].iloc[-2])
                        ok_confirm = (not FAKEOUT_ENABLED) or confirm_rule2(p.side, p.breakout_high, p.breakout_low, confirm_close)
                        if not ok_confirm:
                            print(f"[{symbol}] Rule2 FAIL confirm_close={confirm_close:.6f}", flush=True)
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
                                print(f"[{symbol}] Pullback FAIL low/high={pull_low:.6f}/{pull_high:.6f}", flush=True)
                                del pending[symbol]
                            elif not ok_vol:
                                print(f"[{symbol}] Rule1 FAIL pull_vol>{p2.breakout_vol:.2f}", flush=True)
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
                                print(f"[{symbol}] Entry invalid at open (open={open_price:.6f}, sl={sl:.6f})", flush=True)
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

                # ---- create new setup once per close ----
                if last_processed_close.get(symbol) == ts_close:
                    continue
                last_processed_close[symbol] = ts_close

                df15_closed = df15_raw.iloc[:-1].copy()

                c = float(df15_closed["close"].iloc[-1])
                v = float(df15_closed["volume"].iloc[-1])
                vma = float(df15_closed["volume"].rolling(VOL_MA_LEN).mean().iloc[-1])
                r = float(rsi(df15_closed["close"], RSI_LEN).iloc[-1])

                e20v = float(ema(df15_closed["close"], EMA_FAST).iloc[-1])
                e50v = float(ema(df15_closed["close"], EMA_SLOW).iloc[-1])

                atr_series = atr(df15_closed, ATR_LEN)
                atrv = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0

                b1h = compute_bias(df1h)
                b4h = compute_bias(df4h)
                bias = combine_bias(b1h, b4h)

                if bias.direction == "NEUTRAL":
                    print(f"[{symbol}] close={ts_close} bias=NEUTRAL (1h={b1h.direction},4h={b4h.direction})", flush=True)
                    continue

                sup, res = best_pivot_zones_4h(df4h, current_price=c)
                break_ok, break_reason = is_break_event(bias.direction, c, sup, res)

                ok, why = entry_filters(bias.direction, c, e20v, e50v, r, v, vma, break_event=break_ok)
                if not ok:
                    print(f"[{symbol}] close={ts_close} bias={bias.direction} setup=NO ({why}) rsi={r:.2f}", flush=True)
                    continue

                if not break_ok:
                    print(f"[{symbol}] close={ts_close} bias={bias.direction} break=NO ({break_reason})", flush=True)
                    continue

                row = df15_raw.iloc[-2]  # breakout candle
                b_high = float(row["high"])
                b_low = float(row["low"])
                b_close = float(row["close"])
                b_vol = float(row["volume"])

                if symbol not in pending:
                    pending[symbol] = Pending(
                        symbol=symbol,
                        side=bias.direction,
                        stage="WAIT_CONFIRM",
                        breakout_ts=ts_close,
                        breakout_high=b_high,
                        breakout_low=b_low,
                        breakout_close=b_close,
                        breakout_vol=b_vol,
                        sup=sup,
                        res=res,
                        confirm_ts=ts_open,
                        rsi=r,
                        ema20=e20v,
                        ema50=e50v,
                        volma=vma,
                        atr=atrv,
                        bias_1h=b1h.direction,
                        bias_4h=b4h.direction,
                    )
                    print(f"[{symbol}] ✅ Pending created: {bias.direction} | {break_reason} | wait CONFIRM at {ts_open}", flush=True)

            time.sleep(5)

        except Exception as e:
            print(f"⚠️ BOT ERROR: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
