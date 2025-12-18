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

# RSI gates to prevent "shorting into oversold" / "longing into overbought"
RSI_SHORT_MIN = 48.0   # 👈 wichtig für deinen SUI-Fall (RSI 43 -> short wird blockiert)
RSI_LONG_MAX  = 60.0

VOL_MA_LEN = 20
VOL_RATIO = 0.10       # eher mild; wir arbeiten ohnehin mit Next-Open Trigger

ATR_LEN = 14
ATR_MULT = 1.5
MAX_SL_PCT = 0.02       # max 2%

ENTRY_PAD_PCT = 0.0006  # Entry-Range +-0.06%
RR_TARGETS = (1, 2, 3)

# "Next candle open" trigger validity:
MAX_WAIT_CANDLES = 1    # nur die nächste Candle; danach verfällt Setup

# Telegram ENV (supports both naming styles)
BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()

# Bybit symbols (you can write "SUI/USDT" etc.)
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
    side: str                 # "LONG" / "SHORT"
    setup_ts: pd.Timestamp    # timestamp of the CLOSED candle that created setup
    trigger_ts: pd.Timestamp  # timestamp of the NEXT candle open (df15_raw.index[-1])
    close: float
    atr: float
    rsi: float
    vol: float
    volma: float
    ema20: float
    ema50: float
    sup: Optional[Zone]
    res: Optional[Zone]
    candle_index: int         # how many 15m candles waited since setup (0 at creation)


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
# ZONES (simple, stable)
# =========================

def simple_sr_zones_4h(df4h: pd.DataFrame, lookback: int = 60, pad_pct: float = 0.0015) -> Tuple[Optional[Zone], Optional[Zone]]:
    """
    Very simple SR:
    - Support from recent swing low area
    - Resistance from recent swing high area
    Zone width = pad_pct of price.
    """
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
    # strict: both must match, else neutral
    if b1h.direction == b4h.direction and b1h.direction in ("LONG", "SHORT"):
        return Bias(b1h.direction)
    return Bias("NEUTRAL")

def entry_filters(side: str, c: float, e20v: float, e50v: float, r: float, v: float, vma: float) -> Tuple[bool, str]:
    vol_ok = vma > 0 and (v >= vma * VOL_RATIO)

    if side == "SHORT":
        # trend down on 15m + avoid shorting into oversold
        if not (c < e20v < e50v):
            return False, "15m trend not SHORT (EMA)"
        if r < RSI_SHORT_MIN:
            return False, f"RSI too low for SHORT ({r:.2f} < {RSI_SHORT_MIN})"
        if not vol_ok:
            return False, "Volume too low"
        return True, "OK"

    if side == "LONG":
        # trend up on 15m + avoid longing into overbought
        if not (c > e20v > e50v):
            return False, "15m trend not LONG (EMA)"
        if r > RSI_LONG_MAX:
            return False, f"RSI too high for LONG ({r:.2f} > {RSI_LONG_MAX})"
        if not vol_ok:
            return False, "Volume too low"
        return True, "OK"

    return False, "Invalid side"

def build_plan(side: str, entry_price: float, atr_val: float) -> Dict:
    """
    SL = min(ATR*mult, 2% cap)
    TP1/2/3 = 1R/2R/3R
    """
    side = side.upper()

    # ATR-based risk distance
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

def build_signal_html(setup: PendingSetup, entry_price: float) -> str:
    side = setup.side.upper()
    head = "🟢 <b>LONG</b>" if side == "LONG" else "🔴 <b>SHORT</b>"

    plan = build_plan(side, entry_price, setup.atr)

    entry_low, entry_high = entry_range_from_close(setup.close)

    pair = setup.symbol.replace(":USDT", "").replace("/", "")
    time_setup = to_local(setup.setup_ts)
    time_trigger = to_local(setup.trigger_ts)

    sup_txt = fmt_zone(setup.sup)
    res_txt = fmt_zone(setup.res)

    return (
        f"{head} <b>ENTRY TRIGGERED</b>\n\n"
        f"📊 <b>Pair:</b> {pair}\n"
        f"🕒 <b>Setup Close:</b> {time_setup}\n"
        f"🕒 <b>Trigger (Next Open):</b> {time_trigger}\n\n"
        f"💰 <b>Close:</b> {setup.close:.4f}\n"
        f"📍 <b>RSI({RSI_LEN}):</b> {setup.rsi:.2f}\n"
        f"📈 <b>EMA20/EMA50:</b> {setup.ema20:.4f} / {setup.ema50:.4f}\n"
        f"📦 <b>Vol/VolMA:</b> {setup.vol:.2f} / {setup.volma:.2f}\n\n"
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
# MAIN LOOP (Close -> Next Open Trigger)
# =========================

def main():
    ex = make_exchange()

    symbols = [to_bybit_linear(s) for s in SYMBOLS]
    markets = ex.load_markets()
    symbols = [s for s in symbols if s in markets and markets[s].get("active", True)]

    print(f"✅ BOT AKTIV – 15m Close -> Next Open Trigger | Telegram={'ON' if telegram_enabled() else 'OFF'}", flush=True)
    print(f"✅ Symbols: {symbols}", flush=True)

    last_processed_close: Dict[str, pd.Timestamp] = {}
    pending: Dict[str, PendingSetup] = {}

    while True:
        try:
            for symbol in symbols:
                df15_raw = fetch_df(ex, symbol, TIMEFRAME_ENTRY, limit=300)
                df1h = fetch_df(ex, symbol, TIMEFRAME_BIAS_1H, limit=300)
                df4h = fetch_df(ex, symbol, TIMEFRAME_BIAS_4H, limit=300)

                if len(df15_raw) < 5:
                    continue

                # df15_raw last row is CURRENT open candle. Previous (-2) is last CLOSED candle.
                ts_open = df15_raw.index[-1]    # current candle open timestamp
                ts_close = df15_raw.index[-2]   # last closed candle timestamp

                # 1) If we have pending setup, try to trigger at CURRENT candle open
                if symbol in pending:
                    p = pending[symbol]

                    # advance candle wait count if we moved forward
                    if ts_open > p.trigger_ts:
                        p.candle_index += 1
                        p.trigger_ts = ts_open

                    # expire if waited too long
                    if p.candle_index >= MAX_WAIT_CANDLES:
                        del pending[symbol]
                    else:
                        open_price = float(df15_raw["open"].iloc[-1])

                        # Entry range based on setup close
                        entry_low, entry_high = entry_range_from_close(p.close)

                        # invalidation: if open already beyond SL direction, cancel
                        plan_tmp = build_plan(p.side, (entry_low + entry_high) / 2.0, p.atr)
                        sl = plan_tmp["sl"]

                        if p.side == "SHORT" and open_price >= sl:
                            del pending[symbol]
                        elif p.side == "LONG" and open_price <= sl:
                            del pending[symbol]
                        else:
                            # trigger only if open is inside entry range
                            if entry_low <= open_price <= entry_high:
                                msg = build_signal_html(p, entry_price=open_price)
                                print("\n" + msg + "\n", flush=True)
                                send_telegram_html(msg)
                                del pending[symbol]

                # 2) Create new setup ONLY once per closed candle
                if last_processed_close.get(symbol) == ts_close:
                    continue
                last_processed_close[symbol] = ts_close

                # Use last CLOSED candle df15 = df15_raw[:-1]
                df15 = df15_raw.iloc[:-1].copy()

                # Compute bias (1h + 4h)
                b1h = compute_bias(df1h)
                b4h = compute_bias(df4h)
                bias = combine_bias(b1h, b4h)

                # If neutral, skip creating setups
                if bias.direction == "NEUTRAL":
                    print(f"[{symbol}] close={ts_close} bias=NEUTRAL (1h={b1h.direction},4h={b4h.direction})", flush=True)
                    continue

                close_series = df15["close"]
                vol_series = df15["volume"]

                c = float(close_series.iloc[-1])
                v = float(vol_series.iloc[-1])
                vma = float(vol_series.rolling(VOL_MA_LEN).mean().iloc[-1])
                r = float(rsi(close_series, RSI_LEN).iloc[-1])
                e20v = float(ema(close_series, EMA_FAST).iloc[-1])
                e50v = float(ema(close_series, EMA_SLOW).iloc[-1])
                atrv = float(atr(df15, ATR_LEN).iloc[-1]) if not np.isnan(atr(df15, ATR_LEN).iloc[-1]) else 0.0

                sup, res = simple_sr_zones_4h(df4h)

                ok, why = entry_filters(bias.direction, c, e20v, e50v, r, v, vma)

                print(
                    f"[{symbol}] close={ts_close} bias={bias.direction} ok_setup={ok} "
                    f"rsi={r:.2f} ema20={e20v:.4f} ema50={e50v:.4f} vol={v:.2f} volma={vma:.2f} why={why}",
                    flush=True
                )

                if not ok:
                    continue

                # Create pending setup; DO NOT SEND NOW.
                # It will only send if next candle open is inside entry range and not invalidated.
                pending[symbol] = PendingSetup(
                    symbol=symbol,
                    side=bias.direction,
                    setup_ts=ts_close,
                    trigger_ts=ts_open,   # next candle open timestamp
                    close=c,
                    atr=atrv,
                    rsi=r,
                    vol=v,
                    volma=vma,
                    ema20=e20v,
                    ema50=e50v,
                    sup=sup,
                    res=res,
                    candle_index=0,
                )

            time.sleep(5)

        except Exception as e:
            print(f"⚠️ BOT ERROR: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
