import os
import time
import ccxt
import numpy as np
import pandas as pd
import urllib.parse
import urllib.request

from typing import Dict, Tuple
from dataclasses import dataclass
from zoneinfo import ZoneInfo


# =========================================================
# CONFIG
# =========================================================

TIMEFRAME_ENTRY = "15m"
TIMEFRAME_BIAS = "1h"
TIMEFRAME_CTX_4H = "4h"
TIMEFRAME_CTX_D1 = "1d"

EMA_FAST = 20
EMA_SLOW = 50

RSI_LOWER = 34
RSI_UPPER = 64

VOL_MA_LEN = 20
VOL_RATIO = 0.12

MAX_SL_PCT = 0.02
ENTRY_PAD_PCT = 0.0006
RR_TARGETS = (1, 2, 3)

LOCAL_TZ = "Europe/Berlin"

# Cooldown to prevent spam (per symbol+side)
COOLDOWN_MINUTES = 45

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

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


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class Bias:
    direction: str  # LONG / SHORT / NEUTRAL


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ Telegram OFF (BOT_TOKEN / CHAT_ID missing)", flush=True)
        return

    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception as e:
        print(f"⚠️ Telegram error: {e}", flush=True)


# =========================================================
# HELPERS
# =========================================================

def to_bybit(sym: str) -> str:
    return sym if ":" in sym else f"{sym}:USDT"

def local_time(ts) -> str:
    return ts.to_pydatetime().astimezone(
        ZoneInfo(LOCAL_TZ)
    ).strftime("%Y-%m-%d %H:%M:%S %Z")

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)

def minutes_between(ts_a, ts_b) -> float:
    return abs((ts_a.to_pydatetime() - ts_b.to_pydatetime()).total_seconds()) / 60.0


# =========================================================
# STRATEGY
# =========================================================

def compute_bias(df: pd.DataFrame) -> Bias:
    e20 = ema(df["close"], EMA_FAST).iloc[-1]
    e50 = ema(df["close"], EMA_SLOW).iloc[-1]
    if e20 > e50:
        return Bias("LONG")
    if e20 < e50:
        return Bias("SHORT")
    return Bias("NEUTRAL")


def compute_vol_ok(df15: pd.DataFrame) -> Tuple[bool, float, float, float]:
    vol = df15["volume"]
    v = float(vol.iloc[-1])
    vma = float(vol.rolling(VOL_MA_LEN).mean().iloc[-1])
    vol_ratio = (v / vma) if vma > 0 else 0.0
    return (vol_ratio >= VOL_RATIO), v, vma, vol_ratio


def entry_signal_15m(df15: pd.DataFrame, side: str) -> Tuple[bool, str, Dict]:
    """
    Two entries:
      - AGGRESSIVE: shift/cross relative to EMA20 (avoid spam)
      - SAFE_PULLBACK: pullback above/below EMA20 then reclaim/reject
    """
    close = df15["close"]
    c0 = float(close.iloc[-1])   # last CLOSED
    c1 = float(close.iloc[-2])   # prev candle

    e20 = ema(close, EMA_FAST)
    e50 = ema(close, EMA_SLOW)
    r = rsi(close, 14)

    e20_0 = float(e20.iloc[-1])
    e50_0 = float(e50.iloc[-1])
    r0 = float(r.iloc[-1])

    vol_ok, v, vma, vol_ratio = compute_vol_ok(df15)

    info = {
        "close": c0,
        "prev_close": c1,
        "ema20": e20_0,
        "ema50": e50_0,
        "rsi": r0,
        "vol": v,
        "volma": vma,
        "vol_ratio": vol_ratio,
    }

    if side == "SHORT":
        rsi_ok = r0 > RSI_LOWER
        if not (rsi_ok and vol_ok):
            return False, "blocked(rsi/vol)", info

        # SAFE_PULLBACK: was above EMA20, now closes back below (rejection)
        safe = (c1 > e20_0) and (c0 < e20_0)

        # AGGRESSIVE: first push under EMA20 while EMA20 already below EMA50 OR flattening
        aggressive = (c1 >= e20_0) and (c0 < e20_0) and (e20_0 <= e50_0 or (e20_0 - e50_0) / c0 < 0.001)

        if safe:
            return True, "SAFE_PULLBACK", info
        if aggressive:
            return True, "AGGRESSIVE", info

        return False, "no_trigger", info

    if side == "LONG":
        rsi_ok = r0 < RSI_UPPER
        if not (rsi_ok and vol_ok):
            return False, "blocked(rsi/vol)", info

        # SAFE_PULLBACK: was below EMA20, now closes back above (reclaim)
        safe = (c1 < e20_0) and (c0 > e20_0)

        # AGGRESSIVE: first push over EMA20 while EMA20 already above EMA50 OR flattening
        aggressive = (c1 <= e20_0) and (c0 > e20_0) and (e20_0 >= e50_0 or (e50_0 - e20_0) / c0 < 0.001)

        if safe:
            return True, "SAFE_PULLBACK", info
        if aggressive:
            return True, "AGGRESSIVE", info

        return False, "no_trigger", info

    return False, "neutral", info


# =========================================================
# TRADE PLAN
# =========================================================

def trade_plan(side: str, price: float):
    if side == "SHORT":
        entry_lo = price
        entry_hi = price * (1 + ENTRY_PAD_PCT)
        entry_mid = (entry_lo + entry_hi) / 2
        sl = entry_mid * (1 + MAX_SL_PCT)
        risk = sl - entry_mid
        tps = [entry_mid - risk * r for r in RR_TARGETS]
    else:
        entry_lo = price * (1 - ENTRY_PAD_PCT)
        entry_hi = price
        entry_mid = (entry_lo + entry_hi) / 2
        sl = entry_mid * (1 - MAX_SL_PCT)
        risk = entry_mid - sl
        tps = [entry_mid + risk * r for r in RR_TARGETS]
    return entry_lo, entry_hi, sl, tps


# =========================================================
# EXCHANGE
# =========================================================

def fetch_df(ex, sym, tf, limit=300):
    data = ex.fetch_ohlcv(sym, tf, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df


# =========================================================
# MAIN
# =========================================================

def main():
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})

    symbols = [to_bybit(s) for s in SYMBOLS]
    markets = ex.load_markets()
    symbols = [s for s in symbols if s in markets and markets[s].get("active", True)]

    print("✅ BOT AKTIV – 1h Bias / 15m Entry (SAFE+AGGRESSIVE) + Cooldown", flush=True)
    print(f"✅ Symbols: {symbols}", flush=True)

    last_seen_candle: Dict[str, pd.Timestamp] = {}
    last_signal_time: Dict[str, pd.Timestamp] = {}  # key = f"{symbol}|{side}"

    while True:
        try:
            for sym in symbols:
                # Fetch entry tf, use last CLOSED candle
                df15_raw = fetch_df(ex, sym, TIMEFRAME_ENTRY)
                if len(df15_raw) < 5:
                    continue
                df15 = df15_raw.iloc[:-1].copy()
                candle_ts = df15.index[-1]

                if last_seen_candle.get(sym) == candle_ts:
                    continue
                last_seen_candle[sym] = candle_ts

                # Bias / context
                df1h = fetch_df(ex, sym, TIMEFRAME_BIAS)
                df4h = fetch_df(ex, sym, TIMEFRAME_CTX_4H)
                dfd = fetch_df(ex, sym, TIMEFRAME_CTX_D1)

                bias_1h = compute_bias(df1h)
                bias_4h = compute_bias(df4h)
                bias_d1 = compute_bias(dfd)

                side = bias_1h.direction
                if side not in ("LONG", "SHORT"):
                    continue

                ok, entry_type, info = entry_signal_15m(df15, side)

                print(
                    f"[{sym}] candle={candle_ts} 1h={bias_1h.direction} "
                    f"ok={ok} type={entry_type} rsi={info.get('rsi',0):.2f} vol_ratio={info.get('vol_ratio',0):.2f}",
                    flush=True
                )

                if not ok:
                    continue

                # Cooldown per symbol+side
                key = f"{sym}|{side}"
                if key in last_signal_time:
                    mins = minutes_between(candle_ts, last_signal_time[key])
                    if mins < COOLDOWN_MINUTES:
                        continue
                last_signal_time[key] = candle_ts

                entry_lo, entry_hi, sl, tps = trade_plan(side, info["close"])

                warn = ""
                if side != bias_4h.direction:
                    warn += "⚠️ Gegen 4h-Trend\n"
                if side != bias_d1.direction:
                    warn += "⚠️ Gegen Tages-Trend\n"

                head = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
                pair = sym.replace(":USDT", "").replace("/", "")

                msg = (
                    f"{head} SETUP ({entry_type})\n\n"
                    f"📊 Pair: {pair}\n"
                    f"🕒 15m Close: {local_time(candle_ts)}\n\n"
                    f"🧭 Bias: 1h={bias_1h.direction} | 4h={bias_4h.direction} | D1={bias_d1.direction}\n"
                    f"{warn}\n"
                    f"💰 Close: {info['close']:.4f} | RSI: {info['rsi']:.2f}\n"
                    f"📈 EMA20: {info['ema20']:.4f} | 📉 EMA50: {info['ema50']:.4f}\n"
                    f"📦 VolRatio: {info['vol_ratio']:.2f} (req ≥ {VOL_RATIO})\n\n"
                    f"🎯 Entry: {entry_lo:.4f} – {entry_hi:.4f}\n"
                    f"🛑 SL: {sl:.4f} (≤2%)\n"
                    f"✅ TP1: {tps[0]:.4f}\n"
                    f"✅ TP2: {tps[1]:.4f}\n"
                    f"✅ TP3: {tps[2]:.4f}\n\n"
                    FOOTER_TEXT = (
    "<b>©️ Copyright by crypto_mistik</b>\n\n"
    "⚠️ <b>Kein Financial Advice</b>\n"
    "Dieses Signal dient ausschließlich zu Informations- und Bildungszwecken. "
    "Es stellt keine Anlageberatung, Kauf- oder Verkaufsempfehlung dar.\n\n"
    "💥 <b>Hohe Risiken</b>\n"
    "Der Handel mit Kryptowährungen ist mit hohen Risiken verbunden. "
    "Krypto-Märkte sind extrem volatil und können zu schnellen und erheblichen Verlusten führen. "
    "Jeder handelt auf eigenes Risiko.\n\n"
    "🔒 <b>Urheberrecht & Nutzung</b>\n"
    "Die Nutzung dieses AI-Trading-Bots, von Teilen davon oder das Kopieren relevanter Code-Bestandteile "
    "ist ohne einen gültigen <b>Premium-Zugang</b> sowie ohne vorherige schriftliche Zustimmung des "
    "Channel-Betreibers ausdrücklich untersagt.\n"
    "Zuwiderhandlungen können zivil- und strafrechtlich verfolgt werden."
)
                )

                print("\n" + msg + "\n", flush=True)
                send_telegram(msg)

            time.sleep(5)

        except Exception as e:
            print(f"⚠️ ERROR: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
