import os
import requests
import pandas as pd

from fastapi import FastAPI
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
from openai import OpenAI

load_dotenv()

app = FastAPI(title="SignalPro AI")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def get_mexc_klines(symbol: str = "BTC_USDT", interval: str = "Min15"):
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {"interval": interval, "limit": 200}

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    if not data.get("success"):
        raise Exception("Erreur MEXC: données indisponibles")

    k = data["data"]

    df = pd.DataFrame({
        "time": k["time"],
        "open": k["open"],
        "high": k["high"],
        "low": k["low"],
        "close": k["close"],
        "volume": k["vol"],
    })

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna()


def calculate_indicators(df: pd.DataFrame):
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    df["ema20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()

    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    df["atr"] = atr.average_true_range()

    return df.dropna()


def generate_signal(df: pd.DataFrame):
    last = df.iloc[-1]

    price = float(last["close"])
    rsi = float(last["rsi"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    macd = float(last["macd"])
    macd_signal = float(last["macd_signal"])
    atr = float(last["atr"])

    score = 0
    reasons = []

    if price > ema20 > ema50:
        score += 30
        reasons.append("Le prix est au-dessus des EMA 20 et 50.")
    elif price < ema20 < ema50:
        score -= 30
        reasons.append("Le prix est sous les EMA 20 et 50.")
    else:
        reasons.append("Les moyennes mobiles ne donnent pas encore une tendance claire.")

    if 45 <= rsi <= 65:
        score += 15
        reasons.append("Le RSI est dans une zone saine.")
    elif rsi < 35:
        score += 10
        reasons.append("Le RSI indique une zone survendue.")
    elif rsi > 70:
        score -= 20
        reasons.append("Le RSI indique une zone surachetée.")

    if macd > macd_signal:
        score += 25
        reasons.append("Le MACD montre un momentum positif.")
    else:
        score -= 25
        reasons.append("Le MACD montre un momentum négatif.")

    if score >= 40:
        signal = "BUY"
        confidence = min(90, 50 + score)
        stop_loss = price - atr * 1.5
        take_profit_1 = price + atr * 2
        take_profit_2 = price + atr * 3
    elif score <= -40:
        signal = "SELL"
        confidence = min(90, 50 + abs(score))
        stop_loss = price + atr * 1.5
        take_profit_1 = price - atr * 2
        take_profit_2 = price - atr * 3
    else:
        signal = "WAIT"
        confidence = 50
        stop_loss = None
        take_profit_1 = None
        take_profit_2 = None

    return {
        "price": round(price, 6),
        "rsi": round(rsi, 2),
        "ema20": round(ema20, 6),
        "ema50": round(ema50, 6),
        "macd": round(macd, 6),
        "macd_signal": round(macd_signal, 6),
        "signal": signal,
        "confidence": round(confidence, 2),
        "stop_loss": round(stop_loss, 6) if stop_loss else None,
        "take_profit_1": round(take_profit_1, 6) if take_profit_1 else None,
        "take_profit_2": round(take_profit_2, 6) if take_profit_2 else None,
        "reasons": reasons,
    }


def ai_explanation(symbol: str, interval: str, result: dict):
    if not OPENAI_API_KEY:
        return "Analyse IA indisponible : clé OpenAI non configurée."

    prompt = f"""
Explique ce signal crypto en français simple et prudent.

Paire: {symbol}
Timeframe: {interval}
Prix: {result['price']}
Signal: {result['signal']}
Confiance: {result['confidence']}%
RSI: {result['rsi']}
EMA20: {result['ema20']}
EMA50: {result['ema50']}
MACD: {result['macd']}
MACD Signal: {result['macd_signal']}
Stop-loss: {result['stop_loss']}
Take-profit 1: {result['take_profit_1']}
Take-profit 2: {result['take_profit_2']}
Raisons: {result['reasons']}

Réponds avec :
1. Pourquoi ce signal
2. Le risque principal
3. Ce qu'il faut surveiller avant d'entrer
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Tu donnes des analyses trading prudentes, sans promettre de gains.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return completion.choices[0].message.content


@app.get("/")
def home():
    return {
        "app": "SignalPro AI",
        "status": "online",
        "example": "/analyze?symbol=BTC_USDT&interval=Min15",
        "pairs": ["BTC_USDT", "ETH_USDT", "SOL_USDT", "RSR_USDT"],
        "intervals": ["Min5", "Min15", "Min60"],
    }


@app.get("/analyze")
def analyze(symbol: str = "BTC_USDT", interval: str = "Min15"):
    df = get_mexc_klines(symbol=symbol.upper(), interval=interval)
    df = calculate_indicators(df)
    result = generate_signal(df)

    try:
        explanation = ai_explanation(symbol.upper(), interval, result)
    except Exception:
        explanation = "Analyse IA indisponible pour le moment."

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "analysis": result,
        "ai_explanation": explanation,
        "warning": "Analyse informative uniquement. Ce n’est pas un conseil financier.",
      }
