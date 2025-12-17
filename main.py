import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import ccxt
from zoneinfo import ZoneInfo


# =========================
# CONFIG
# =========================

DEBUG_LOGS = True

TIMEFRAME_ENTRY = "15m"
TIMEFRAME_BIAS = "4h"

# Indicators
EMA_FAST = 20
EMA_SLOW = 50
RSI_LOWER = 34
RSI_UPPER = 64

# Volume
VOL_MA_LEN = 20
VOL_RATIO = 0.12  # 🔧 0.08 = mehr Trades | 0.15 = strenger (für 15m realistisch)

# Risk / Plan
MAX_SL_PCT = 0.02          # max 2%
ENTRY_PAD_PCT = 0.0006     # Entry range ~0.06%
RR_TARGETS = (1.0, 2.0, 3.0)

# Liquidity (kept minimal in signal)
LIQ_RANGE_PCT = 0.008      # ±0.80%
LIQ_SNAPSHOTS = 3
LIQ_PERSIST_MIN = 2
LIQ_REQ_RATIO = 1.40       # asks/bids ratio for SHORT (example)

# Symbols (you can write plain "XXX/USDT" – we normalize to Bybit linear perp)
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

LOCAL_TZ = "Europe/Berlin"


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
    touches: int = 0
    rejections: int = 0
    strength: float = 0.0


# =========================
# HELPERS
# =========================

def to_bybit_linear(sym: str) -> str:
    """
    Normalize to ccxt Bybit linear perp symbol format:
    BTC/USDT:USDT, FARTCOIN/USDT:USDT, ...
    """
    if ":" in sym:
        return sym
    return f"{sym}:USDT"


def to_local_ts(ts) -> str:
    """
    Convert pandas Timestamp / datetime to Europe/Berlin string.
    """
    try:
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(ZoneInfo(LOCAL_TZ)).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(ts)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def fmt_zone(title: str, z: Optional[Zone]) -> str:
    if z is None:
        return f"{title}: —"
    return f"{title}: {z.low:.4f}–{z.high:.4f} (t={z.touches}, r={z.rejections}, s={z.strength:.1f})"


def compute_trade_plan(side: str, close: float,
                       max_sl_pct: float = MAX_SL_PCT,
                       entry_pad_pct: float = ENTRY_PAD_PCT,
                       rr_targets=RR_TARGETS) -> Dict:
    """
    Simple plan:
    - Entry range near close
    - SL capped at max 2%
    - TP1/TP2/TP3 at 1R/2R/3R
    """
    side = side.upper()

    if side == "SHORT":
        entry_low = close
        entry_high = close * (1 + entry_pad_pct)
    else:
        entry_low = close * (1 - entry_pad_pct)
        entry_high = close

    entry_mid = (entry_low + entry_high) / 2.0

    if side == "SHORT":
        sl = entry_mid * (1 + max_sl_pct)
        risk = sl - entry_mid
        tps = [entry_mid - risk * rr for rr in rr_targets]
    else:
        sl = entry_mid * (1 - max_sl_pct)
        risk = entry_mid - sl
        tps = [entry_mid + risk * rr for rr in rr_targets]

    rr_main = rr_targets[1] if len(rr_targets) > 1 else rr_targets[0]
    return {
        "entry_low": entry_low,
        "entry_high": entry_high,
        "entry_mid": entry_mid,
        "sl": sl,
        "tp1": tps[0],
        "tp2": tps[1] if len(tps) > 1 else None,
        "tp3": tps[2] if len(tps) > 2 else None,
        "crv": f"1:{rr_main:.0f}",
        "risk_pct": max_sl_pct * 100.0,
    }


# =========================
# STRATEGY CORE
# =========================

def check_entry(df_15m: pd.DataFrame, bias_dir: str) -> Tuple[bool, str, Dict]:
    close = df_15m["close"]
    vol = df_15m["volume"]

    e20_series = ema(close, EMA_FAST)
    e50_series = ema(close, EMA_SLOW)
    rsi_series = rsi(close, 14)
    volma_series = vol.rolling(VOL_MA_LEN).mean()

    c = float(close.iloc[-1])
    e20v = float(e20_series.iloc[-1])
    e50v = float(e50_series.iloc[-1])
    rv = float(rsi_series.iloc[-1])
    vv = float(vol.iloc[-1])

    vma_raw = volma_series.iloc[-1]
    vma = float(vma_raw) if not np.isnan(vma_raw) else float(np.mean(vol.tail(VOL_MA_LEN)))

    vol_ratio = (vv / vma) if vma > 0 else 0.0
    vol_ok = vv >= vma * VOL_RATIO

    info = {
        "close": c,
        "ema20": e20v,
        "ema50": e50v,
        "rsi": rv,
        "vol": vv,
        "volma": vma,
        "vol_ratio": vol_ratio,
    }

    bias_dir = (bias_dir or "").upper()

    if bias_dir == "LONG":
        # Trend up + RSI not overheated + volume ok
        rsi_ok = rv < RSI_UPPER
        ok = (c > e20v) and (e20v > e50v) and rsi_ok and vol_ok
        reason = "15m confirms LONG" if ok else f"LONG blocked (rsi_ok={rsi_ok}, vol_ratio={vol_ratio:.2f})"
        return ok, reason, info

    if bias_dir == "SHORT":
        # Trend down + RSI not oversold + volume ok
        rsi_ok = rv > RSI_LOWER
        ok = (c < e20v) and (e20v < e50v) and rsi_ok and vol_ok
        reason = "15m confirms SHORT" if ok else f"SHORT blocked (rsi_ok={rsi_ok}, vol_ratio={vol_ratio:.2f})"
        return ok, reason, info

    return False, "Bias neutral", info


def compute_bias(df_4h: pd.DataFrame) -> Bias:
    """
    Placeholder bias: EMA20 vs EMA50 on 4h.
    Replace with your real bias logic if needed.
    """
    close = df_4h["close"]
    e20 = ema(close, EMA_FAST).iloc[-1]
    e50 = ema(close, EMA_SLOW).iloc[-1]
    if e20 > e50:
        return Bias("LONG")
    if e20 < e50:
        return Bias("SHORT")
    return Bias("NEUTRAL")


def best_zones_4h(df_4h: pd.DataFrame, current_price: float) -> Tuple[Optional[Zone], Optional[Zone]]:
    """
    Placeholder: You already have a real zone finder.
    Keep your existing implementation and return Zone objects (low/high/...).
    """
    # --- Replace this with your own best_zones_4h ---
    return None, None


def check_breakout_breakdown(close: float, sup: Optional[Zone], res: Optional[Zone]) -> Tuple[Optional[str], str]:
    """
    Basic breakdown/breakout detection vs 4h zones.
    Returns (signal_side, label).
    """
    if sup and close < sup.low:
        return "SHORT", "BREAKDOWN (Support broken)"
    if res and close > res.high:
        return "LONG", "BREAKOUT (Resistance broken)"
    return None, "No 4h breakout/breakdown"


# =========================
# LIQUIDITY (minimal)
# =========================

def fetch_orderbook(exchange, symbol: str, limit: int = 50) -> Optional[Dict]:
    try:
        return exchange.fetch_order_book(symbol, limit=limit)
    except Exception:
        return None


def liquidity_snapshot(orderbook: Dict, mid: float, pct: float) -> Tuple[float, float]:
    """
    Sum bids below and asks above in +-pct band around mid.
    """
    if not orderbook:
        return 0.0, 0.0

    low = mid * (1 - pct)
    high = mid * (1 + pct)

    bids = orderbook.get("bids", []) or []
    asks = orderbook.get("asks", []) or []

    below = sum(a for p, a in bids if low <= p <= mid)
    above = sum(a for p, a in asks if mid <= p <= high)

    return float(below), float(above)


def liquidity_check_minimal(exchange, symbol: str, mid: float) -> Dict:
    """
    Minimal liquidity check: asks/bids ratio for SHORT (example).
    """
    below_list = []
    above_list = []

    for _ in range(LIQ_SNAPSHOTS):
        ob = fetch_orderbook(exchange, symbol)
        b, a = liquidity_snapshot(ob, mid, LIQ_RANGE_PCT)
        below_list.append(b)
        above_list.append(a)
        time.sleep(0.2)

    below = float(np.mean(below_list))
    above = float(np.mean(above_list))
    ratio = (above / below) if below > 0 else 999.0

    ok_short = ratio >= LIQ_REQ_RATIO
    ok_long = (1/ratio) >= LIQ_REQ_RATIO if ratio > 0 else False  # symmetric-ish

    return {
        "below": below,
        "above": above,
        "ratio": ratio,
        "ok_short": ok_short,
        "ok_long": ok_long,
    }


# =========================
# EXCHANGE / DATA
# =========================

def make_exchange() -> ccxt.Exchange:
    ex = ccxt.bybit({
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",  # linear perps
        }
    })

    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    if api_key and api_secret:
        ex.apiKey = api_key
        ex.secret = api_secret

    return ex


def fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    # ccxt ts is ms UTC
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df


# =========================
# SIGNAL MESSAGE
# =========================

def build_signal_message(symbol: str,
                         candle_ts,
                         side: str,
                         label: str,
                         bias_dir: str,
                         info: Dict,
                         sup: Optional[Zone],
                         res: Optional[Zone],
                         liq: Optional[Dict]) -> str:
    side = side.upper()
    head = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
    icon = "🟢" if side == "LONG" else "🔴"

    local_time = to_local_ts(candle_ts)

    plan = compute_trade_plan(
        side=side,
        close=float(info["close"]),
        max_sl_pct=MAX_SL_PCT,
        entry_pad_pct=ENTRY_PAD_PCT,
        rr_targets=RR_TARGETS
    )

    liq_line = "💧 Liquidity: —"
    if liq:
        if side == "SHORT":
            liq_ok = liq.get("ok_short", False)
        else:
            liq_ok = liq.get("ok_long", False)
        liq_line = f"💧 Liquidity: ratio={liq.get('ratio', 0):.2f}x | ok={liq_ok}"

    pair_compact = symbol.replace("/", "").replace(":USDT", "")

    msg = (
        f"{head} {icon}  —  {label}\n\n"
        f"📊 Pair: {pair_compact}\n"
        f"🕒 TF: 15m Close | 4h Bias: {bias_dir}\n"
        f"🕯️ Candle: {local_time}\n\n"
        f"💰 Close: {info['close']:.4f}\n"
        f"📈 EMA20: {info['ema20']:.4f} | 📉 EMA50: {info['ema50']:.4f}\n"
        f"📍 RSI(14): {info['rsi']:.2f} (Bands {RSI_LOWER}/{RSI_UPPER})\n"
        f"📦 Vol: {info['vol']:.2f} | VolMA({VOL_MA_LEN}): {info['volma']:.2f} | Ratio: {info.get('vol_ratio', 0):.2f}\n\n"
        f"🎯 Entry: {plan['entry_low']:.4f} – {plan['entry_high']:.4f}\n"
        f"🛑 SL (max {plan['risk_pct']:.0f}%): {plan['sl']:.4f}\n"
        f"✅ TP1: {plan['tp1']:.4f}\n"
        f"✅ TP2: {plan['tp2']:.4f}\n"
        f"✅ TP3: {plan['tp3']:.4f}\n"
        f"📌 CRV (TP2): {plan['crv']}\n\n"
        f"🧱 4h Zones\n"
        f"- {fmt_zone('Support', sup)}\n"
        f"- {fmt_zone('Resistance', res)}\n\n"
        f"{liq_line}\n"
        f"⚠️ Automatisches Signal (kein Financial Advice), Achtung Kryptowährungen können sehr volatil sein, bitte analysiere zusätzlich den Chart"
    )
    return msg


# =========================
# MAIN LOOP
# =========================

def main():
    exchange = make_exchange()

    # Normalize symbols to Bybit linear perp format
    symbols = [to_bybit_linear(s) for s in SYMBOLS]

    # Load markets and filter active
    markets = exchange.load_markets()
    symbols_ok = []
    for s in symbols:
        if s in markets and markets[s].get("active", True):
            symbols_ok.append(s)
        else:
            print(f"⚠️ Skipping invalid/inactive symbol: {s}", flush=True)

    if not symbols_ok:
        print("❌ No valid symbols after filtering. Exiting.", flush=True)
        return

    last_seen_candle: Dict[str, pd.Timestamp] = {}

    print(f"✅ Bot started. Symbols: {symbols_ok}", flush=True)

    while True:
        try:
            for symbol in symbols_ok:
                print(f"Checking {symbol.replace(':USDT','')}", flush=True)

                df15 = fetch_ohlcv_df(exchange, symbol, TIMEFRAME_ENTRY, limit=300)
                df4h = fetch_ohlcv_df(exchange, symbol, TIMEFRAME_BIAS, limit=300)

                candle_ts = df15.index[-1]  # UTC timestamp

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
                        f"vol={info['vol']:.2f} volma={info['volma']:.2f} vol_ratio={info.get('vol_ratio',0):.2f} "
                        f"sup={fmt_zone('sup', sup)} res={fmt_zone('res', res)}",
                        flush=True
                    )

                # Require 15m confirmation first
                if not ok_entry:
                    continue

                # Determine breakout/breakdown vs 4h zones
                side, label = check_breakout_breakdown(info["close"], sup, res)
                if side is None:
                    continue

                # Minimal liquidity check + gate
                liq = liquidity_check_minimal(exchange, symbol, mid=info["close"])
                if side == "SHORT" and not liq.get("ok_short", False):
                    continue
                if side == "LONG" and not liq.get("ok_long", False):
                    continue

                # Build and print signal
                msg = build_signal_message(
                    symbol=symbol,
                    candle_ts=candle_ts,
                    side=side,
                    label=label,
                    bias_dir=bias.direction,
                    info=info,
                    sup=sup,
                    res=res,
                    liq=liq
                )

                print("\n" + msg + "\n", flush=True)

            # Sleep a bit; candle-close logic handles dedupe
            time.sleep(5)

        except Exception as e:
            print(f"⚠️ Bot error: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
