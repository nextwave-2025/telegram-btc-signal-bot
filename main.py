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
RSI_LONG_MAX  = 60.0

VOL_MA_LEN = 20
VOL_RATIO = 0.10  # mild, follow-through is the main quality filter

ATR_LEN = 14
ATR_MULT = 1.5
MAX_SL_PCT = 0.02  # max 2%

ENTRY_PAD_PCT = 0.0006
RR_TARGETS = (1, 2, 3)

# Follow-through settings
FOLLOW_THROUGH_ENABLED = True

# Telegram ENV (supports both naming styles)
BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()

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
class PendingSetup:
    symbol: str
    side: str                   # "LONG" / "SHORT"
    setup_ts: pd.Timestamp      # timestamp of setup CLOSED candle (15m)
    setup_high: float
    setup_low: float
    setup_close: float

    follow_ts: pd.Timestamp     # timestamp of the next candle (open time) that must close for follow-through
    stage: str                  # "WAIT_FOLLOW" or "WAIT_ENTRY"

    entry_open_ts: Optional[pd.Timestamp] = None  # candle open timestamp where we attempt entry (after follow)
    atr: float = 0.0
    rsi: float = 0.0
    vol: float = 0.0
    volma: float = 0.0
    ema20: float = 0.0
    ema50: float = 0.0
    sup: Optional[Zone] = None
    res: Optional[Zone] = None


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
# ZONES (simple / stable)
# =========================

def simple_sr_zones_4h(df4h: pd.DataFrame, lookback: int = 60, pad_pct: float = 0.0015) -> Tuple[Optional[Zone], Optional[Zone]]:
    if len(df4h) < lookback + 5:
        return None, None
    recent = df4h.tail(lookback)
    lo = float(recent["low"].min())
    hi = float(recent["high"].max())
    mid = float(recent["close"].iloc[-1])
    pad = mid * pad_pct
    return Zone(lo - pad, lo + pad), Zone(hi - pad, hi + pad)


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
    if b1h.direction == b4h.direction and b1h.direction in ("LONG", "SHORT"):
        return Bias(b1h.direction)
    return Bias("NEUTRAL")

def entry_filters(side: str, c: float, e20v: float, e50v: float, r: float, v: float, vma: float) -> Tuple[bool, str]:
    vol_ok = vma > 0 and (v >= vma * VOL_RATIO)

    if side == "SHORT":
        if not (c < e20v < e50v):
            return False, "15m EMA trend not SHORT"
        if r < RSI_SHORT_MIN:
            return False, f"RSI too low for SHORT ({r:.2f} < {RSI_SHORT_MIN})"
        if not vol_ok:
            return False, "Volume too low"
        return True, "OK"

    if side == "LONG":
        if not (c > e20v > e50v):
            return False, "15m EMA trend not LONG"
        if r > RSI_LONG_MAX:
            return False, f"RSI too high for LONG ({r:.2f} > {RSI_LONG_MAX})"
        if not vol_ok:
            return False, "Volume too low"
        return True, "OK"

    return False, "Invalid side"

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

def entry_range_from_close(close: float) -> Tuple[float, float]:
    low = close * (1 - ENTRY_PAD_PCT)
    high = close * (1 + ENTRY_PAD_PCT)
    return float(low), float(high)

def follow_through_pass(side: str, setup_high: float, setup_low: float, follow_close: float) -> bool:
    """
    Follow-through confirmation:
    LONG: follow_close > setup_high
    SHORT: follow_close < setup_low
    """
    side = side.upper()
    if side == "LONG":
        return follow_close > setup_high
    return follow_close < setup_low


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

def to_local(ts: pd.Timestamp) -> str:
    return ts.to_pydatetime().astimezone(ZoneInfo(LOCAL_TZ)).strftime("%Y-%m-%d %H:%M:%S %Z")


# =========================
# MESSAGE
# =========================

def fmt_zone(z: Optional[Zone]) -> str:
    if not z:
        return "—"
    return f"{z.low:.4f}–{z.high:.4f}"

def build_entry_html(p: PendingSetup, entry_price: float, trigger_open_ts: pd.Timestamp) -> str:
    side = p.side.upper()
    head = "🟢 <b>LONG</b>" if side == "LONG" else "🔴 <b>SHORT</b>"
    plan = build_plan(side, entry_price, p.atr)
    entry_low, entry_high = entry_range_from_close(p.setup_close)

    pair = p.symbol.replace(":USDT", "").replace("/", "")
    time_setup = to_local(p.setup_ts)
    time_follow = to_local(p.follow_ts)
    time_trigger = to_local(trigger_open_ts)

    sup_txt = fmt_zone(p.sup)
    res_txt = fmt_zone(p.res)

    return (
        f"{head} <b>ENTRY</b> (Follow-Through bestätigt)\n\n"
        f"📊 <b>Pair:</b> {pair}\n"
        f"🕒 <b>Setup Close:</b> {time_setup}\n"
        f"✅ <b>Follow Close:</b> {time_follow}\n"
        f"🚀 <b>Entry Open:</b> {time_trigger}\n\n"
        f"💰 <b>Setup Close:</b> {p.setup_close:.4f}\n"
        f"📍 <b>RSI({RSI_LEN}):</b> {p.rsi:.2f}\n"
        f"📈 <b>EMA20/EMA50:</b> {p.ema20:.4f} / {p.ema50:.4f}\n"
        f"📦 <b>Vol/VolMA:</b> {p.vol:.2f} / {p.volma:.2f}\n\n"
        f"🎯 <b>Entry Range:</b> {entry_low:.4f} – {entry_high:.4f}\n"
        f"✅ <b>Entry:</b> {entry_price:.4f}\n"
        f"🛑 <b>SL:</b> {plan['sl']:.4f} (Risk {plan['risk_pct']:.2f}%, max {MAX_SL_PCT*100:.0f}%)\n"
        f"✅ <b>TP1:</b> {plan['tp1']:.4f}\n"
        f"✅ <b>TP2:</b> {plan['tp2']:.4f}\n"
        f"✅ <b>TP3:</b> {plan['tp3']:.4f}\n"
        f"📌 <b>CRV (TP2):</b> {plan['crv']}\n\n"
        f"🧱 <b>4h Zones:</b>\n"
        f"Support: {sup_txt}\n"
        f"Resistance: {res_txt}\n\n"
        f"<b>©️ Copyright by crypto_mistik.</b>\n"
        f"⚠️ Kein Financial Advice"
    )


# =========================
# MAIN LOOP
# Setup Close -> Follow Close confirm -> Entry at next Open
# =========================

def main():
    ex = make_exchange()

    symbols = [to_bybit_linear(s) for s in SYMBOLS]
    markets = ex.load_markets()
    symbols = [s for s in symbols if s in markets and markets[s].get("active", True)]

    print(f"✅ BOT AKTIV – 15m Setup -> Follow-Through -> Entry@Open | Telegram={'ON' if telegram_enabled() else 'OFF'}", flush=True)
    print(f"✅ Symbols: {symbols}", flush=True)

    last_processed_setup_close: Dict[str, pd.Timestamp] = {}
    pending: Dict[str, PendingSetup] = {}

    while True:
        try:
            for symbol in symbols:
                df15_raw = fetch_df(ex, symbol, TIMEFRAME_ENTRY, limit=300)
                df1h = fetch_df(ex, symbol, TIMEFRAME_BIAS_1H, limit=300)
                df4h = fetch_df(ex, symbol, TIMEFRAME_BIAS_4H, limit=300)

                if len(df15_raw) < 6:
                    continue

                # In ccxt OHLCV, index is candle open time.
                ts_open = df15_raw.index[-1]    # current open candle
                ts_close = df15_raw.index[-2]   # last closed candle (open time of that candle)

                # =========================
                # 1) Handle pending setups
                # =========================
                if symbol in pending:
                    p = pending[symbol]

                    # A) WAIT_FOLLOW: when the follow candle closes, evaluate confirmation
                    if p.stage == "WAIT_FOLLOW":
                        # follow candle closes when it becomes the last closed candle:
                        # i.e. ts_close == p.follow_ts
                        if ts_close == p.follow_ts:
                            follow_close = float(df15_raw["close"].iloc[-2])

                            if (not FOLLOW_THROUGH_ENABLED) or follow_through_pass(p.side, p.setup_high, p.setup_low, follow_close):
                                # confirmed -> next candle open is ts_open, attempt entry there
                                p.stage = "WAIT_ENTRY"
                                p.entry_open_ts = ts_open  # candle after follow
                                # fall through to entry attempt in same loop
                            else:
                                print(f"[{symbol}] Follow-Through FAILED ({p.side}) | follow_close={follow_close:.6f} setupH/L={p.setup_high:.6f}/{p.setup_low:.6f}", flush=True)
                                del pending[symbol]

                    # B) WAIT_ENTRY: only attempt entry at the open candle right after follow confirm
                    if symbol in pending and pending[symbol].stage == "WAIT_ENTRY":
                        p2 = pending[symbol]
                        if p2.entry_open_ts is not None and ts_open == p2.entry_open_ts:
                            open_price = float(df15_raw["open"].iloc[-1])

                            entry_low, entry_high = entry_range_from_close(p2.setup_close)
                            in_range = entry_low <= open_price <= entry_high

                            # SL invalidation at open
                            plan_tmp = build_plan(p2.side, (entry_low + entry_high) / 2.0, p2.atr)
                            sl = plan_tmp["sl"]
                            invalid = (open_price >= sl) if p2.side == "SHORT" else (open_price <= sl)

                            if invalid:
                                print(f"[{symbol}] Entry invalidated at open {open_price:.6f} (SL={sl:.6f})", flush=True)
                                del pending[symbol]
                            elif not in_range:
                                print(f"[{symbol}] Entry open not in range {entry_low:.6f}-{entry_high:.6f} (open={open_price:.6f})", flush=True)
                                del pending[symbol]
                            else:
                                msg = build_entry_html(p2, entry_price=open_price, trigger_open_ts=ts_open)
                                print("\n" + msg + "\n", flush=True)
                                send_telegram_html(msg)
                                del pending[symbol]

                # =========================
                # 2) Create new setup (once per closed candle)
                # =========================
                if last_processed_setup_close.get(symbol) == ts_close:
                    continue
                last_processed_setup_close[symbol] = ts_close

                # Setup candle is last closed candle => row -2
                setup_row = df15_raw.iloc[-2]
                setup_close = float(setup_row["close"])
                setup_high = float(setup_row["high"])
                setup_low = float(setup_row["low"])

                # Indicators calculated on closed candles only
                df15_closed = df15_raw.iloc[:-1].copy()

                b1h = compute_bias(df1h)
                b4h = compute_bias(df4h)
                bias = combine_bias(b1h, b4h)

                if bias.direction == "NEUTRAL":
                    print(f"[{symbol}] setup_close={ts_close} bias=NEUTRAL (1h={b1h.direction},4h={b4h.direction})", flush=True)
                    continue

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

                sup, res = simple_sr_zones_4h(df4h)

                ok, why = entry_filters(bias.direction, c, e20v, e50v, r, v, vma)

                print(
                    f"[{symbol}] setup_close={ts_close} bias={bias.direction} ok_setup={ok} "
                    f"rsi={r:.2f} ema20={e20v:.6f} ema50={e50v:.6f} vol={v:.2f} volma={vma:.2f} why={why}",
                    flush=True
                )

                if not ok:
                    continue

                # Create pending setup: follow candle is the current open candle at ts_open
                pending[symbol] = PendingSetup(
                    symbol=symbol,
                    side=bias.direction,
                    setup_ts=ts_close,
                    setup_high=setup_high,
                    setup_low=setup_low,
                    setup_close=setup_close,
                    follow_ts=ts_open,
                    stage="WAIT_FOLLOW",
                    atr=atrv,
                    rsi=r,
                    vol=v,
                    volma=vma,
                    ema20=e20v,
                    ema50=e50v,
                    sup=sup,
                    res=res,
                )

            time.sleep(5)

        except Exception as e:
            print(f"⚠️ BOT ERROR: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
