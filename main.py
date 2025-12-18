import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

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

# Prevent "shorting into oversold" / "longing into overbought"
RSI_SHORT_MIN = 48.0
RSI_LONG_MAX = 60.0

VOL_MA_LEN = 20
VOL_RATIO = 0.10  # mild; fakeout rules filter quality

ATR_LEN = 14
ATR_MULT = 1.5
MAX_SL_PCT = 0.02  # max 2%

ENTRY_PAD_PCT = 0.0006
RR_TARGETS = (1, 2, 3)

# Zones (simple stable fallback)
SR_LOOKBACK_4H = 60
SR_PAD_PCT = 0.0015

# Fakeout rules
FAKEOUT_ENABLED = True

# Telegram ENV (supports both naming styles)
BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()

# Scheduled alerts (Europe/Berlin)
ALERT_US_OPENING_TIME = "15:15"   # "US Opening in 15 Minuten"
ALERT_US_CLOSING_TIME = "21:45"   # "US Closing in 15 Minuten"

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

@dataclass
class Pending:
    symbol: str
    side: str                 # "LONG" / "SHORT"
    stage: str                # "WAIT_CONFIRM" -> "WAIT_PULLBACK" -> "WAIT_ENTRY"

    # breakout candle (t0)
    breakout_ts: pd.Timestamp
    breakout_high: float
    breakout_low: float
    breakout_close: float
    breakout_vol: float

    # zones at breakout time
    sup: Optional[Zone]
    res: Optional[Zone]

    # confirm candle (t1) is the next candle after breakout
    confirm_ts: pd.Timestamp

    # pullback candle (t2) is next candle after confirm
    pullback_ts: Optional[pd.Timestamp] = None

    # entry open candle (t3) is next candle after pullback close
    entry_open_ts: Optional[pd.Timestamp] = None

    # indicators snapshot (for message)
    rsi: float = 0.0
    ema20: float = 0.0
    ema50: float = 0.0
    volma: float = 0.0
    atr: float = 0.0

    # debug info
    bias_1h: str = ""
    bias_4h: str = ""


# =========================
# TELEGRAM
# =========================

def telegram_enabled() -> bool:
    return bool(BOT_TOKEN and CHAT_ID)

def send_telegram_html(text: str) -> bool:
    if not telegram_enabled():
        print("⚠️ Telegram OFF (BOT_TOKEN/CHAT_ID missing)", flush=True)
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        ok = '"ok":true' in body.replace(" ", "").lower()
        if not ok:
            print(f"⚠️ Telegram send failed: {body[:500]}", flush=True)
        return ok
    except Exception as e:
        print(f"⚠️ Telegram error: {type(e).__name__}: {e}", flush=True)
        return False


# =========================
# TIME HELPERS (scheduled alerts)
# =========================

def local_now():
    return pd.Timestamp.now(tz=ZoneInfo(LOCAL_TZ))

def today_key(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")

def maybe_send_scheduled_alerts(state: Dict[str, str]):
    """
    Sends daily alerts exactly once per day at specific local times.
    state dict keeps last sent date per alert key.
    """
    now = local_now()
    hhmm = now.strftime("%H:%M")
    day = today_key(now)

    if hhmm == ALERT_US_OPENING_TIME:
        if state.get("us_opening") != day:
            msg = (
                "⚠️ <b>Achtung:</b> US Opening in 15 Minuten.\n"
                "Der Markt kann volatil werden."
            )
            send_telegram_html(msg)
            state["us_opening"] = day

    if hhmm == ALERT_US_CLOSING_TIME:
        if state.get("us_closing") != day:
            msg = (
                "⚠️ <b>Achtung:</b> US Closing in 15 Minuten.\n"
                "Danach kein Trade empfohlen, da Market Maker den Preis stark beeinflussen können."
            )
            send_telegram_html(msg)
            state["us_closing"] = day


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
# ZONES (simple fallback)
# =========================

def simple_sr_zones_4h(df4h: pd.DataFrame, lookback: int = SR_LOOKBACK_4H, pad_pct: float = SR_PAD_PCT) -> Tuple[Optional[Zone], Optional[Zone]]:
    if len(df4h) < lookback + 5:
        return None, None
    recent = df4h.tail(lookback)
    lo = float(recent["low"].min())
    hi = float(recent["high"].max())
    mid = float(recent["close"].iloc[-1])
    pad = mid * pad_pct
    sup = Zone(low=lo - pad, high=lo + pad)
    res = Zone(low=hi - pad, high=hi + pad)
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
    """
    ✅ Feintuning: 4h is MASTER.
    1h is informative only; it will not block trades anymore.
    """
    if b4h.direction in ("LONG", "SHORT"):
        return Bias(b4h.direction)
    return Bias("NEUTRAL")

def entry_filters(side: str, close: float, e20v: float, e50v: float, r: float, v: float, vma: float) -> Tuple[bool, str]:
    """
    ✅ Feintuning EMA logic:
    Instead of requiring close < e20 < e50 (too strict),
    we require trend alignment e20 < e50 and price below either e20 OR e50.
    Same mirrored for LONG.
    """
    vol_ok = (vma > 0) and (v >= vma * VOL_RATIO)

    if side == "SHORT":
        if not (e20v < e50v and (close < e20v or close < e50v)):
            return False, "15m EMA not aligned for SHORT"
        if r < RSI_SHORT_MIN:
            return False, f"RSI too low for SHORT ({r:.2f} < {RSI_SHORT_MIN})"
        if not vol_ok:
            return False, "Volume too low"
        return True, "OK"

    if side == "LONG":
        if not (e20v > e50v and (close > e20v or close > e50v)):
            return False, "15m EMA not aligned for LONG"
        if r > RSI_LONG_MAX:
            return False, f"RSI too high for LONG ({r:.2f} > {RSI_LONG_MAX})"
        if not vol_ok:
            return False, "Volume too low"
        return True, "OK"

    return False, "Invalid side"

def is_break_event(side: str, close: float, sup: Optional[Zone], res: Optional[Zone]) -> Tuple[bool, str]:
    """
    Require breakout/breakdown relative to 4h zones to start fakeout pipeline.
    LONG: close > res.high
    SHORT: close < sup.low
    """
    if side == "LONG":
        if not res:
            return False, "No resistance zone"
        return (close > res.high), "Breakout above resistance"
    else:
        if not sup:
            return False, "No support zone"
        return (close < sup.low), "Breakdown below support"

def entry_range_from_price(px: float) -> Tuple[float, float]:
    low = px * (1 - ENTRY_PAD_PCT)
    high = px * (1 + ENTRY_PAD_PCT)
    return float(low), float(high)

def build_plan(side: str, entry_price: float, atr_val: float) -> Dict:
    side = side.upper()
    atr_dist = atr_val * ATR_MULT if atr_val and atr_val > 0 else entry_price * 0.005
    cap_dist = entry_price * MAX_SL_PCT
    risk_dist = min(atr_dist, cap_dist)

    if side == "SHORT":
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
        "crv": f"1:{int(RR_TARGETS[1])}" if len(RR_TARGETS) > 1 else f"1:{int(RR_TARGETS[0])}",
    }


# =========================
# FAKEOUT RULES (3 rules)
# =========================
# Rule 2: confirm candle closes beyond breakout extreme
# Rule 3: pullback must be immediate (next candle after confirm)
# Rule 1: pullback volume <= breakout volume

def confirm_rule2(side: str, breakout_high: float, breakout_low: float, confirm_close: float) -> bool:
    if side == "LONG":
        return confirm_close > breakout_high
    return confirm_close < breakout_low

def is_pullback(side: str, breakout_high: float, breakout_low: float, pull_low: float, pull_high: float) -> bool:
    """
    Pullback definition:
    LONG: pullback candle trades back to breakout_high area (low <= breakout_high)
    SHORT: pullback candle trades back to breakout_low area (high >= breakout_low)
    """
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
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
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

    sup_txt = fmt_zone(p.sup)
    res_txt = fmt_zone(p.res)

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
        f"Support: {sup_txt}\n"
        f"Resistance: {res_txt}\n\n"
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

    # scheduled alert state
    alert_state: Dict[str, str] = {}

    while True:
        try:
            # send time-based alerts (once per day)
            maybe_send_scheduled_alerts(alert_state)

            for symbol in symbols:
                df15_raw = fetch_df(ex, symbol, TIMEFRAME_ENTRY, limit=300)
                df1h = fetch_df(ex, symbol, TIMEFRAME_BIAS_1H, limit=300)
                df4h = fetch_df(ex, symbol, TIMEFRAME_BIAS_4H, limit=300)

                if len(df15_raw) < 10:
                    continue

                ts_open = df15_raw.index[-1]
                ts_close = df15_raw.index[-2]

                # 1) pending state machine
                if symbol in pending:
                    p = pending[symbol]

                    # WAIT_CONFIRM: confirm candle closes => ts_close == p.confirm_ts
                    if p.stage == "WAIT_CONFIRM" and ts_close == p.confirm_ts:
                        confirm_close = float(df15_raw["close"].iloc[-2])

                        ok_confirm = (not FAKEOUT_ENABLED) or confirm_rule2(p.side, p.breakout_high, p.breakout_low, confirm_close)
                        if not ok_confirm:
                            print(f"[{symbol}] Rule2 FAIL: confirm_close={confirm_close:.6f} (breakH/L={p.breakout_high:.6f}/{p.breakout_low:.6f})", flush=True)
                            del pending[symbol]
                        else:
                            p.stage = "WAIT_PULLBACK"
                            p.pullback_ts = ts_open  # next candle after confirm

                    # WAIT_PULLBACK: pullback candle closes => ts_close == p.pullback_ts
                    if symbol in pending and pending[symbol].stage == "WAIT_PULLBACK":
                        p2 = pending[symbol]
                        if p2.pullback_ts is not None and ts_close == p2.pullback_ts:
                            pull_row = df15_raw.iloc[-2]
                            pull_low = float(pull_row["low"])
                            pull_high = float(pull_row["high"])
                            pull_vol = float(pull_row["volume"])

                            ok_pullback = is_pullback(p2.side, p2.breakout_high, p2.breakout_low, pull_low, pull_high)
                            ok_vol = (not FAKEOUT_ENABLED) or pullback_rule1_volume(pull_vol, p2.breakout_vol)

                            if not ok_pullback:
                                print(f"[{symbol}] Pullback FAIL: not a pullback (low/high={pull_low:.6f}/{pull_high:.6f})", flush=True)
                                del pending[symbol]
                            elif not ok_vol:
                                print(f"[{symbol}] Rule1 FAIL: pull_vol={pull_vol:.2f} > breakout_vol={p2.breakout_vol:.2f}", flush=True)
                                del pending[symbol]
                            else:
                                p2.stage = "WAIT_ENTRY"
                                p2.entry_open_ts = ts_open

                    # WAIT_ENTRY: trigger at entry open candle
                    if symbol in pending and pending[symbol].stage == "WAIT_ENTRY":
                        p3 = pending[symbol]
                        if p3.entry_open_ts is not None and ts_open == p3.entry_open_ts:
                            open_price = float(df15_raw["open"].iloc[-1])

                            entry_low, entry_high = entry_range_from_price(p3.breakout_close)
                            in_range = entry_low <= open_price <= entry_high

                            plan_tmp = build_plan(p3.side, (entry_low + entry_high) / 2.0, p3.atr)
                            sl = plan_tmp["sl"]
                            invalid = (open_price >= sl) if p3.side == "SHORT" else (open_price <= sl)

                            if invalid:
                                print(f"[{symbol}] Entry invalidated at open {open_price:.6f} (SL={sl:.6f})", flush=True)
                                del pending[symbol]
                            elif not in_range:
                                print(f"[{symbol}] Entry open not in range {entry_low:.6f}-{entry_high:.6f} (open={open_price:.6f})", flush=True)
                                del pending[symbol]
                            else:
                                msg = build_entry_html(p3, entry_price=open_price, entry_open_ts=ts_open)
                                print("\n" + msg + "\n", flush=True)
                                send_telegram_html(msg)
                                del pending[symbol]

                # 2) create new setup on closed candle
                if last_processed_close.get(symbol) == ts_close:
                    continue
                last_processed_close[symbol] = ts_close

                df15_closed = df15_raw.iloc[:-1].copy()
                close_series = df15_closed["close"]
                vol_series = df15_closed["volume"]

                c = float(close_series.iloc[-1])
                v = float(vol_series.iloc[-1])
                vma = float(vol_series.rolling(VOL_MA_LEN).mean().iloc[-1])
                r = float(rsi(close_series, RSI_LEN).iloc[-1])
                e20v = float(ema(close_series, EMA_FAST).iloc[-1])
                e50v = float(ema(close_series, EMA_SLOW).iloc[-1])

                atr_series = atr(df15_closed, ATR_LEN)
                atrv = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0

                b1h = compute_bias(df1h)
                b4h = compute_bias(df4h)
                bias = combine_bias(b1h, b4h)

                sup, res = simple_sr_zones_4h(df4h)

                if bias.direction == "NEUTRAL":
                    print(f"[{symbol}] close={ts_close} bias=NEUTRAL (1h={b1h.direction},4h={b4h.direction})", flush=True)
                    continue

                ok, why = entry_filters(bias.direction, c, e20v, e50v, r, v, vma)
                if not ok:
                    print(f"[{symbol}] close={ts_close} bias={bias.direction} setup=NO ({why}) rsi={r:.2f}", flush=True)
                    continue

                ok_break, why_break = is_break_event(bias.direction, c, sup, res)
                if not ok_break:
                    print(f"[{symbol}] close={ts_close} bias={bias.direction} break=NO ({why_break})", flush=True)
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
                        confirm_ts=ts_open,  # next candle
                        rsi=r,
                        ema20=e20v,
                        ema50=e50v,
                        volma=vma,
                        atr=atrv,
                        bias_1h=b1h.direction,
                        bias_4h=b4h.direction,
                    )
                    print(f"[{symbol}] ✅ Pending created: {bias.direction} | {why_break} | wait CONFIRM at {ts_open}", flush=True)

            time.sleep(5)

        except Exception as e:
            print(f"⚠️ BOT ERROR: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
