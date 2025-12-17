import os
import time
import json
import urllib.parse
import urllib.request
import ccxt
import numpy as np
import pandas as pd

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from zoneinfo import ZoneInfo


# =========================================================
# CONFIG
# =========================================================

TIMEFRAME_ENTRY = "15m"
TIMEFRAME_BIAS = "4h"

EMA_FAST = 20
EMA_SLOW = 50

RSI_LOWER = 34
RSI_UPPER = 64

VOL_MA_LEN = 20
VOL_RATIO = 0.12        # realistisch für 15m

MAX_SL_PCT = 0.02       # max 2%
ENTRY_PAD_PCT = 0.0006
RR_TARGETS = (1, 2, 3)

LOCAL_TZ = "Europe/Berlin"

# Telegram ENV:
# TELEGRAM_BOT_TOKEN=123:ABC...
# TELEGRAM_CHAT_ID=123456789
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID", "")

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
    direction: str   # LONG / SHORT / NEUTRAL


# Optional placeholder if you later re-add zones
@dataclass
class Zone:
    low: float
    high: float
    touches: int = 0
    rejections: int = 0
    strength: float = 0.0


# =========================================================
# TELEGRAM
# =========================================================

def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

def send_telegram(text: str) -> bool:
    """
    Sends a Telegram message via Bot API.
    Uses urllib (no extra dependencies).
    """
    if not telegram_enabled():
        # Keep this visible so you immediately notice missing env vars
        print("⚠️ Telegram disabled: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set", flush=True)
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
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


# =========================================================
# HELPERS
# =========================================================

def to_bybit_linear(sym: str) -> str:
    return sym if ":" in sym else f"{sym}:USDT"

def to_local_time(ts) -> str:
    dt = ts.to_pydatetime()
    return dt.astimezone(ZoneInfo(LOCAL_TZ)).strftime("%Y-%m-%d %H:%M:%S %Z")

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


# =========================================================
# STRATEGY
# =========================================================

def compute_bias(df4h: pd.DataFrame) -> Bias:
    e20 = ema(df4h["close"], EMA_FAST).iloc[-1]
    e50 = ema(df4h["close"], EMA_SLOW).iloc[-1]
    if e20 > e50:
        return Bias("LONG")
    if e20 < e50:
        return Bias("SHORT")
    return Bias("NEUTRAL")

def check_entry(df15: pd.DataFrame, bias_dir: str) -> Tuple[bool, Dict]:
    close = df15["close"]
    vol = df15["volume"]

    e20 = ema(close, EMA_FAST).iloc[-1]
    e50 = ema(close, EMA_SLOW).iloc[-1]
    r = rsi(close, 14).iloc[-1]

    v = float(vol.iloc[-1])
    vma = float(vol.rolling(VOL_MA_LEN).mean().iloc[-1])

    vol_ratio = (v / vma) if vma > 0 else 0.0
    vol_ok = v >= vma * VOL_RATIO

    info = {
        "close": float(close.iloc[-1]),
        "ema20": float(e20),
        "ema50": float(e50),
        "rsi": float(r),
        "vol": float(v),
        "volma": float(vma),
        "vol_ratio": float(vol_ratio),
    }

    bias_dir = (bias_dir or "").upper()

    if bias_dir == "SHORT":
        ok = (close.iloc[-1] < e20) and (e20 < e50) and (r > RSI_LOWER) and vol_ok
        return bool(ok), info

    if bias_dir == "LONG":
        ok = (close.iloc[-1] > e20) and (e20 > e50) and (r < RSI_UPPER) and vol_ok
        return bool(ok), info

    return False, info


# =========================================================
# TRADE PLAN
# =========================================================

def trade_plan(side: str, close: float) -> Dict:
    side = side.upper()

    if side == "SHORT":
        entry_low = close
        entry_high = close * (1 + ENTRY_PAD_PCT)
        entry_mid = (entry_low + entry_high) / 2
        sl = entry_mid * (1 + MAX_SL_PCT)
        risk = sl - entry_mid
        tps = [entry_mid - risk * rr for rr in RR_TARGETS]
    else:
        entry_low = close * (1 - ENTRY_PAD_PCT)
        entry_high = close
        entry_mid = (entry_low + entry_high) / 2
        sl = entry_mid * (1 - MAX_SL_PCT)
        risk = entry_mid - sl
        tps = [entry_mid + risk * rr for rr in RR_TARGETS]

    crv = f"1:{int(RR_TARGETS[1])}" if len(RR_TARGETS) > 1 else f"1:{int(RR_TARGETS[0])}"

    return {
        "entry_low": float(entry_low),
        "entry_high": float(entry_high),
        "entry_mid": float(entry_mid),
        "sl": float(sl),
        "tp1": float(tps[0]),
        "tp2": float(tps[1]),
        "tp3": float(tps[2]),
        "crv": crv,
        "risk_pct": float(MAX_SL_PCT * 100.0),
    }


def build_signal_message(symbol: str, candle_ts, side: str, info: Dict) -> str:
    side = side.upper()
    head = "🟢 LONG" if side == "LONG" else "🔴 SHORT"

    plan = trade_plan(side, info["close"])

    pair = symbol.replace(":USDT", "").replace("/", "")
    ts_local = to_local_time(candle_ts)

    msg = (
        f"{head} SIGNAL\n\n"
        f"📊 Pair: {pair}\n"
        f"🕒 Candle Close: {ts_local}\n\n"
        f"💰 Close: {info['close']:.4f}\n"
        f"📈 EMA20: {info['ema20']:.4f} | 📉 EMA50: {info['ema50']:.4f}\n"
        f"📍 RSI(14): {info['rsi']:.2f} (Bands {RSI_LOWER}/{RSI_UPPER})\n"
        f"📦 Vol: {info['vol']:.2f} | VolMA({VOL_MA_LEN}): {info['volma']:.2f} | Ratio: {info['vol_ratio']:.2f}\n\n"
        f"🎯 Entry: {plan['entry_low']:.4f} – {plan['entry_high']:.4f}\n"
        f"🛑 SL (max {plan['risk_pct']:.0f}%): {plan['sl']:.4f}\n"
        f"✅ TP1: {plan['tp1']:.4f}\n"
        f"✅ TP2: {plan['tp2']:.4f}\n"
        f"✅ TP3: {plan['tp3']:.4f}\n"
        f"📌 CRV (TP2): {plan['crv']}\n\n"
        f"⚠️ Automatisches Signal (kein Financial Advice) - Kryptowährungen können volatil sein! Bitte überprüfe zusätzlich den jeweiligen Chart bevor du tradest."
        f"©️ Copyright by **crypto_mistik.** Der Einsatz dieses Bots oder von Teilen davon ohne vorherige Zustimmung des Channel-Betreibers ist untersagt. Zuwiderhandlungen können zivil- und strafrechtlich verfolgt werden."
    )
    return msg


# =========================================================
# EXCHANGE
# =========================================================

def make_exchange():
    return ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

def fetch_df(exchange, symbol, tf, limit=300):
    data = exchange.fetch_ohlcv(symbol, tf, limit=limit)
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df


# =========================================================
# MAIN
# =========================================================

def main():
    exchange = make_exchange()
    symbols = [to_bybit_linear(s) for s in SYMBOLS]

    markets = exchange.load_markets()
    symbols = [s for s in symbols if s in markets and markets[s].get("active", True)]

    print(f"✅ BOT AKTIV – 15m Candle Close | Telegram={'ON' if telegram_enabled() else 'OFF'}", flush=True)
    print(f"✅ Symbols: {symbols}", flush=True)

    last_seen: Dict[str, pd.Timestamp] = {}

    while True:
        try:
            for symbol in symbols:
                df15_raw = fetch_df(exchange, symbol, TIMEFRAME_ENTRY)
                df4h = fetch_df(exchange, symbol, TIMEFRAME_BIAS)

                if len(df15_raw) < 3:
                    continue

                # ✅ Use the last CLOSED candle
                df15 = df15_raw.iloc[:-1]
                candle_ts = df15.index[-1]

                if last_seen.get(symbol) == candle_ts:
                    continue
                last_seen[symbol] = candle_ts

                bias = compute_bias(df4h)
                ok, info = check_entry(df15, bias.direction)

                print(
                    f"[{symbol}] {candle_ts} bias={bias.direction} ok={ok} "
                    f"rsi={info['rsi']:.2f} vol_ratio={info['vol_ratio']:.2f}",
                    flush=True
                )

                if not ok:
                    continue

                side = bias.direction  # LONG/SHORT
                msg = build_signal_message(symbol, candle_ts, side, info)

                # Print to logs
                print("\n" + msg + "\n", flush=True)

                # Send to Telegram
                sent = send_telegram(msg)
                if sent:
                    print("✅ Telegram sent", flush=True)

            time.sleep(5)

        except Exception as e:
            print(f"⚠️ BOT ERROR: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()

