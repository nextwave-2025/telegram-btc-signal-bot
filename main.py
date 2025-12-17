import os
import time
import ccxt
import numpy as np
import pandas as pd
import urllib.parse
import urllib.request

from dataclasses import dataclass
from typing import Dict, Tuple
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

COOLDOWN_MINUTES = 45
LOCAL_TZ = "Europe/Berlin"

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
# FOOTER / DISCLAIMER
# =========================================================

FOOTER_TEXT = (
    "<b>©️ Copyright by crypto_mistik</b>\n\n"
    "⚠️ <b>Kein Financial Advice</b>\n"
    "Dieses Signal dient ausschließlich zu Informations- und Bildungszwecken. "
    "Es stellt keine Anlageberatung, Kauf- oder Verkaufsempfehlung dar.\n\n"
    "💥 <b>Hohe Risiken</b>\n"
    "Der Handel mit Kryptowährungen ist hochriskant. "
    "Krypto-Märkte sind extrem volatil und können zu schnellen und erheblichen Verlusten führen. "
    "Jeder handelt auf eigenes Risiko.\n\n"
    "🔒 <b>Urheberrecht & Nutzung</b>\n"
    "Die Nutzung dieses AI-Trading-Bots oder das Kopieren relevanter Code-Bestandteile "
    "ist ohne gültigen <b>Premium-Zugang</b> sowie ohne schriftliche Zustimmung des "
    "Channel-Betreibers untersagt. "
    "Zuwiderhandlungen können zivil- und strafrechtlich verfolgt werden."
)


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
        payload = {
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        data = urllib.parse.urlencode(payload).encode()
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
    return (100 - (100 / (1 + rs))).fillna(50)

def minutes_between(a, b) -> float:
    return abs((a.to_pydatetime() - b.to_pydatetime()).total_seconds()) / 60.0


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


def entry_signal_15m(df15: pd.DataFrame, side: str) -> Tuple[bool, str, Dict]:
    close = df15["close"]
    vol = df15["volume"]

    c0 = float(close.iloc[-1])
    c1 = float(close.iloc[-2])

    e20 = ema(close, EMA_FAST).iloc[-1]
    e50 = ema(close, EMA_SLOW).iloc[-1]
    r = rsi(close).iloc[-1]

    v = float(vol.iloc[-1])
    vma = float(vol.rolling(VOL_MA_LEN).mean().iloc[-1])
    vol_ratio = v / vma if vma > 0 else 0.0

    info = {
        "close": c0,
        "ema20": float(e20),
        "ema50": float(e50),
        "rsi": float(r),
        "vol_ratio": float(vol_ratio),
    }

    if vol_ratio < VOL_RATIO:
        return False, "LOW_VOLUME", info

    if side == "SHORT":
        if r <= RSI_LOWER:
            return False, "RSI_BLOCK", info

        safe = c1 > e20 and c0 < e20
        aggressive = c1 >= e20 and c0 < e20 and e20 <= e50

        if safe:
            return True, "SAFE_PULLBACK", info
        if aggressive:
            return True, "AGGRESSIVE", info

    if side == "LONG":
        if r >= RSI_UPPER:
            return False, "RSI_BLOCK", info

        safe = c1 < e20 and c0 > e20
        aggressive = c1 <= e20 and c0 > e20 and e20 >= e50

        if safe:
            return True, "SAFE_PULLBACK", info
        if aggressive:
            return True, "AGGRESSIVE", info

    return False, "NO_TRIGGER", info


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
    ex = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })

    symbols = [to_bybit(s) for s in SYMBOLS]
    markets = ex.load_markets()
    symbols = [s for s in symbols if s in markets]

    print("✅ BOT AKTIV – 1h Bias | 15m Entry | Pullback + Cooldown", flush=True)

    last_seen_candle: Dict[str, pd.Timestamp] = {}
    last_signal_time: Dict[str, pd.Timestamp] = {}

    while True:
        try:
            for sym in symbols:
                df15_raw = fetch_df(ex, sym, TIMEFRAME_ENTRY)
                if len(df15_raw) < 5:
                    continue

                df15 = df15_raw.iloc[:-1]
                candle_ts = df15.index[-1]

                if last_seen_candle.get(sym) == candle_ts:
                    continue
                last_seen_candle[sym] = candle_ts

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
                    f"[{sym}] {candle_ts} 1h={side} ok={ok} type={entry_type}",
                    flush=True
                )

                if not ok:
                    continue

                key = f"{sym}|{side}"
                if key in last_signal_time:
                    if minutes_between(candle_ts, last_signal_time[key]) < COOLDOWN_MINUTES:
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
                    f"{FOOTER_TEXT}"
                )

                print("\n" + msg + "\n", flush=True)
                send_telegram(msg)

            time.sleep(5)

        except Exception as e:
            print(f"⚠️ ERROR: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
