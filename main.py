import os
import requests
import pandas as pd

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
from openai import OpenAI

load_dotenv()

app = FastAPI(title="NAMU SignalPro AI")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SCAN_SYMBOLS = [
    "BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT", "DOGE_USDT",
    "BNB_USDT", "ADA_USDT", "LINK_USDT", "AVAX_USDT", "SUI_USDT",
    "ENA_USDT", "RSR_USDT", "PEPE_USDT", "WIF_USDT", "TON_USDT"
]


def get_mexc_klines(symbol: str = "BTC_USDT", interval: str = "Min15"):
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {"interval": interval, "limit": 200}

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    if not data.get("success"):
        raise Exception(f"Erreur MEXC : données indisponibles pour {symbol}")

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

    df["volume_avg20"] = df["volume"].rolling(20).mean()
    return df.dropna()


def support_resistance(df: pd.DataFrame):
    recent = df.tail(50)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    price = float(df.iloc[-1]["close"])

    distance_support = ((price - support) / price) * 100 if price else 0
    distance_resistance = ((resistance - price) / price) * 100 if price else 0

    return {
        "support": round(support, 6),
        "resistance": round(resistance, 6),
        "distance_support_pct": round(distance_support, 2),
        "distance_resistance_pct": round(distance_resistance, 2),
    }


def volume_analysis(df: pd.DataFrame):
    last = df.iloc[-1]
    volume = float(last["volume"])
    avg = float(last["volume_avg20"]) if last["volume_avg20"] else 0
    ratio = volume / avg if avg > 0 else 0

    if ratio >= 1.5:
        status = "fort"
    elif ratio >= 0.8:
        status = "normal"
    else:
        status = "faible"

    return {
        "volume": round(volume, 4),
        "volume_avg20": round(avg, 4),
        "volume_ratio": round(ratio, 2),
        "volume_status": status,
    }


def trend_status(df: pd.DataFrame):
    last = df.iloc[-1]
    price = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])

    if price > ema20 > ema50:
        return "haussière"
    if price < ema20 < ema50:
        return "baissière"
    return "neutre"


def generate_signal(df: pd.DataFrame):
    last = df.iloc[-1]

    price = float(last["close"])
    rsi = float(last["rsi"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    macd = float(last["macd"])
    macd_signal = float(last["macd_signal"])
    atr = float(last["atr"])

    sr = support_resistance(df)
    vol = volume_analysis(df)
    trend = trend_status(df)

    score = 0
    reasons = []
    warnings = []

    if price > ema20 > ema50:
        score += 30
        reasons.append("Tendance haussière : prix au-dessus des EMA 20 et 50.")
    elif price < ema20 < ema50:
        score -= 30
        reasons.append("Tendance baissière : prix sous les EMA 20 et 50.")
    else:
        reasons.append("Tendance encore neutre ou mal alignée.")

    if 45 <= rsi <= 65:
        score += 12
        reasons.append("RSI dans une zone saine.")
    elif rsi < 35:
        score += 10
        reasons.append("RSI proche d'une zone survendue.")
    elif rsi > 70:
        score -= 18
        reasons.append("RSI en zone surachetée.")

    if macd > macd_signal:
        score += 22
        reasons.append("MACD positif.")
    else:
        score -= 22
        reasons.append("MACD négatif.")

    if vol["volume_status"] == "fort":
        if score > 0:
            score += 12
        elif score < 0:
            score -= 12
        reasons.append("Volume fort : le mouvement est mieux confirmé.")
    elif vol["volume_status"] == "faible":
        score = int(score * 0.65)
        warnings.append("Volume faible : risque de faux signal.")
        reasons.append("Le volume est faible, confirmation limitée.")

    # Anti faux signaux
    if sr["distance_resistance_pct"] < 0.8 and score > 0:
        score -= 25
        warnings.append("Prix très proche d'une résistance : attention au rejet.")
    if sr["distance_support_pct"] < 0.8 and score < 0:
        score += 25
        warnings.append("Prix très proche d'un support : attention au rebond.")

    if ema50 > price * 3 or ema50 < price / 3:
        warnings.append("Certaines données semblent anormales. Vérifie la paire MEXC avant décision.")

    if score >= 45:
        signal = "BUY"
        confidence = min(92, 50 + score)
        stop_loss = price - atr * 1.5
        take_profit_1 = price + atr * 2
        take_profit_2 = price + atr * 3
    elif score <= -45:
        signal = "SELL"
        confidence = min(92, 50 + abs(score))
        stop_loss = price + atr * 1.5
        take_profit_1 = price - atr * 2
        take_profit_2 = price - atr * 3
    else:
        signal = "WAIT"
        confidence = max(40, min(65, 50 + abs(score) / 3))
        stop_loss = None
        take_profit_1 = None
        take_profit_2 = None

    risk_score = 0
    if vol["volume_status"] == "faible":
        risk_score += 30
    if warnings:
        risk_score += 25
    if signal == "WAIT":
        risk_score += 20
    if confidence < 65:
        risk_score += 20
    if rsi > 70 or rsi < 30:
        risk_score += 15

    if risk_score >= 60:
        risk = "élevé"
    elif risk_score >= 30:
        risk = "moyen"
    else:
        risk = "faible"

    if signal == "BUY":
        confirmation = f"Attendre une clôture au-dessus de {round(price + atr * 0.25, 6)} avec volume au moins normal."
    elif signal == "SELL":
        confirmation = f"Attendre une clôture sous {round(price - atr * 0.25, 6)} avec volume au moins normal."
    else:
        confirmation = "Attendre un meilleur alignement EMA/MACD ou une cassure claire du support/résistance."

    return {
        "price": round(price, 6),
        "rsi": round(rsi, 2),
        "ema20": round(ema20, 6),
        "ema50": round(ema50, 6),
        "macd": round(macd, 6),
        "macd_signal": round(macd_signal, 6),
        "signal": signal,
        "confidence": round(confidence, 2),
        "score": score,
        "trend": trend,
        "risk": risk,
        "support": sr["support"],
        "resistance": sr["resistance"],
        "distance_support_pct": sr["distance_support_pct"],
        "distance_resistance_pct": sr["distance_resistance_pct"],
        "volume_status": vol["volume_status"],
        "volume_ratio": vol["volume_ratio"],
        "stop_loss": round(stop_loss, 6) if stop_loss else None,
        "take_profit_1": round(take_profit_1, 6) if take_profit_1 else None,
        "take_profit_2": round(take_profit_2, 6) if take_profit_2 else None,
        "confirmation": confirmation,
        "reasons": reasons,
        "warnings": warnings,
    }


def ai_explanation(symbol: str, interval: str, result: dict):
    if not client:
        return "Analyse IA indisponible : clé OpenAI non configurée."

    prompt = f"""
Tu es un assistant d'analyse crypto prudent.
Explique ce signal en français simple, court et utile.

Paire: {symbol}
Timeframe: {interval}
Prix: {result['price']}
Signal: {result['signal']}
Confiance: {result['confidence']}%
Tendance: {result['trend']}
Risque: {result['risk']}
RSI: {result['rsi']}
Support: {result['support']}
Résistance: {result['resistance']}
Volume: {result['volume_status']} ratio {result['volume_ratio']}
Confirmation: {result['confirmation']}
Raisons: {result['reasons']}
Avertissements: {result['warnings']}

Réponds avec :
1. Pourquoi ce signal
2. Risque principal
3. Confirmation à attendre
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
        temperature=0.25,
    )

    return completion.choices[0].message.content


def analyze_symbol(symbol: str, interval: str, with_ai: bool = False):
    df = get_mexc_klines(symbol=symbol.upper(), interval=interval)
    df = calculate_indicators(df)
    result = generate_signal(df)

    explanation = ""
    if with_ai:
        try:
            explanation = ai_explanation(symbol.upper(), interval, result)
        except Exception:
            explanation = "Analyse IA indisponible pour le moment."

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "analysis": result,
        "ai_explanation": explanation,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NAMU SignalPro AI</title>
    <style>
        :root {
            --bg: #050816;
            --card: rgba(255,255,255,0.08);
            --card2: rgba(255,255,255,0.12);
            --text: #f8fafc;
            --muted: #94a3b8;
            --gold: #f5c542;
            --blue: #38bdf8;
            --green: #22c55e;
            --red: #ef4444;
            --orange: #f97316;
            --line: rgba(255,255,255,0.12);
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Arial, Helvetica, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(56,189,248,0.24), transparent 35%),
                radial-gradient(circle at top right, rgba(245,197,66,0.18), transparent 30%),
                linear-gradient(135deg, #020617, #0f172a 55%, #020617);
        }

        .container {
            width: 100%;
            max-width: 1150px;
            margin: 0 auto;
            padding: 28px 16px 48px;
        }

        .header {
            padding: 24px;
            border: 1px solid var(--line);
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
            backdrop-filter: blur(12px);
            box-shadow: 0 25px 80px rgba(0,0,0,0.35);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 18px;
        }

        .logo {
            width: 52px;
            height: 52px;
            border-radius: 18px;
            background: linear-gradient(135deg, var(--gold), var(--blue));
            display: flex;
            align-items: center;
            justify-content: center;
            color: #020617;
            font-weight: 900;
            font-size: 20px;
            box-shadow: 0 10px 30px rgba(245,197,66,0.25);
        }

        h1 { margin: 0; font-size: 31px; letter-spacing: -0.5px; }
        .subtitle { margin: 6px 0 0; color: var(--muted); font-size: 14px; line-height: 1.5; }

        .tabs {
            display: flex;
            gap: 10px;
            margin: 18px 0;
        }

        .tab {
            border: 1px solid var(--line);
            background: rgba(15,23,42,0.8);
            color: var(--text);
            border-radius: 999px;
            padding: 12px 16px;
            font-weight: 800;
            cursor: pointer;
        }

        .tab.active {
            background: linear-gradient(135deg, var(--gold), #fde68a);
            color: #020617;
        }

        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr auto;
            gap: 12px;
            margin-top: 16px;
        }

        select, button {
            width: 100%;
            border: none;
            border-radius: 16px;
            padding: 15px 16px;
            font-size: 15px;
            outline: none;
        }

        select {
            background: rgba(15,23,42,0.9);
            color: var(--text);
            border: 1px solid var(--line);
        }

        button {
            cursor: pointer;
            font-weight: 900;
            color: #020617;
            background: linear-gradient(135deg, var(--gold), #fde68a);
            box-shadow: 0 12px 30px rgba(245,197,66,0.18);
        }

        button.secondary {
            background: linear-gradient(135deg, var(--blue), #bae6fd);
        }

        button:disabled { opacity: 0.65; cursor: not-allowed; }

        .grid {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 18px;
            margin-top: 18px;
        }

        .card {
            border-radius: 26px;
            padding: 22px;
            background: var(--card);
            border: 1px solid var(--line);
            box-shadow: 0 20px 60px rgba(0,0,0,0.25);
        }

        .signal-box {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 18px;
        }

        .signal { font-size: 42px; font-weight: 900; letter-spacing: -1px; }
        .badge {
            padding: 10px 14px;
            border-radius: 999px;
            font-weight: 900;
            font-size: 13px;
            background: rgba(255,255,255,0.1);
        }

        .buy { color: var(--green); }
        .sell { color: var(--red); }
        .wait { color: var(--orange); }

        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }

        .metric {
            padding: 14px;
            border-radius: 18px;
            background: var(--card2);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .label { color: var(--muted); font-size: 12px; margin-bottom: 7px; }
        .value { font-weight: 900; font-size: 18px; word-break: break-word; }

        .analysis {
            white-space: pre-line;
            color: #e2e8f0;
            line-height: 1.55;
            font-size: 14px;
        }

        .warning {
            margin-top: 14px;
            padding: 13px;
            border-radius: 16px;
            background: rgba(249,115,22,0.12);
            border: 1px solid rgba(249,115,22,0.28);
            color: #fed7aa;
            font-size: 13px;
            line-height: 1.5;
        }

        .loading, .error {
            display: none;
            margin-top: 18px;
            padding: 18px;
            border-radius: 20px;
        }

        .loading {
            background: rgba(56,189,248,0.10);
            border: 1px solid rgba(56,189,248,0.18);
            color: #bae6fd;
        }

        .error {
            background: rgba(239,68,68,0.12);
            border: 1px solid rgba(239,68,68,0.25);
            color: #fecaca;
        }

        .scanner-list {
            display: grid;
            gap: 12px;
            margin-top: 18px;
        }

        .scan-item {
            padding: 16px;
            border-radius: 20px;
            background: rgba(255,255,255,0.08);
            border: 1px solid var(--line);
        }

        .scan-top {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            align-items: center;
            margin-bottom: 12px;
        }

        .pair {
            font-size: 20px;
            font-weight: 900;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-top: 10px;
        }

        .mini {
            padding: 10px;
            border-radius: 14px;
            background: rgba(15,23,42,0.55);
        }

        .footer {
            margin-top: 18px;
            color: var(--muted);
            font-size: 12px;
            text-align: center;
            line-height: 1.5;
        }

        @media (max-width: 760px) {
            .controls { grid-template-columns: 1fr; }
            .grid { grid-template-columns: 1fr; }
            .mini-grid { grid-template-columns: 1fr; }
            h1 { font-size: 25px; }
            .signal { font-size: 36px; }
            .metrics { grid-template-columns: 1fr; }
            .header { padding: 20px; border-radius: 22px; }
            .tabs { flex-direction: column; }
        }
    </style>
</head>
<body>
    <main class="container">
        <section class="header">
            <div class="brand">
                <div class="logo">NS</div>
                <div>
                    <h1>NAMU SignalPro AI</h1>
                    <p class="subtitle">Dashboard intelligent MEXC : signaux, support/résistance, volume, risque et confirmation avant entrée.</p>
                </div>
            </div>

            <div class="tabs">
                <button class="tab active" id="tabSingle" onclick="setMode('single')">Analyse simple</button>
                <button class="tab" id="tabScanner" onclick="setMode('scanner')">Scanner MEXC</button>
            </div>

            <div id="singleControls" class="controls">
                <select id="symbol">
                    <option value="BTC_USDT">BTC / USDT</option>
                    <option value="ETH_USDT">ETH / USDT</option>
                    <option value="SOL_USDT">SOL / USDT</option>
                    <option value="XRP_USDT">XRP / USDT</option>
                    <option value="DOGE_USDT">DOGE / USDT</option>
                    <option value="SUI_USDT">SUI / USDT</option>
                    <option value="ENA_USDT">ENA / USDT</option>
                    <option value="RSR_USDT">RSR / USDT</option>
                    <option value="PEPE_USDT">PEPE / USDT</option>
                </select>

                <select id="interval">
                    <option value="Min5">5 minutes</option>
                    <option value="Min15" selected>15 minutes</option>
                    <option value="Min60">1 heure</option>
                </select>

                <button id="analyzeBtn" onclick="analyze()">Analyser</button>
            </div>

            <div id="scannerControls" class="controls" style="display:none;">
                <select id="scanInterval">
                    <option value="Min5">Scanner 5 minutes</option>
                    <option value="Min15" selected>Scanner 15 minutes</option>
                    <option value="Min60">Scanner 1 heure</option>
                </select>

                <select id="scanFilter">
                    <option value="ALL">Tous les signaux</option>
                    <option value="BUY">Top BUY</option>
                    <option value="SELL">Top SELL</option>
                    <option value="WAIT">WAIT</option>
                </select>

                <button class="secondary" id="scanBtn" onclick="scanMarket()">Scanner MEXC</button>
            </div>

            <div id="loading" class="loading">Analyse en cours... Le premier chargement peut prendre quelques secondes.</div>
            <div id="error" class="error"></div>
        </section>

        <section id="result" class="grid" style="display:none;">
            <div class="card">
                <div class="signal-box">
                    <div>
                        <div class="label">Signal détecté</div>
                        <div id="signal" class="signal">WAIT</div>
                    </div>
                    <div id="confidence" class="badge">Confiance : --%</div>
                </div>

                <div class="metrics">
                    <div class="metric"><div class="label">Prix actuel</div><div id="price" class="value">--</div></div>
                    <div class="metric"><div class="label">Risque</div><div id="risk" class="value">--</div></div>
                    <div class="metric"><div class="label">Tendance</div><div id="trend" class="value">--</div></div>
                    <div class="metric"><div class="label">RSI</div><div id="rsi" class="value">--</div></div>
                    <div class="metric"><div class="label">Support</div><div id="support" class="value">--</div></div>
                    <div class="metric"><div class="label">Résistance</div><div id="resistance" class="value">--</div></div>
                    <div class="metric"><div class="label">Volume</div><div id="volumeStatus" class="value">--</div></div>
                    <div class="metric"><div class="label">Stop-loss</div><div id="stopLoss" class="value">--</div></div>
                    <div class="metric"><div class="label">Take-profit 1 / 2</div><div id="takeProfit" class="value">--</div></div>
                    <div class="metric"><div class="label">Confirmation</div><div id="confirmation" class="value">--</div></div>
                </div>

                <div id="warningBox"></div>
            </div>

            <div class="card">
                <div class="label">Explication IA</div>
                <div id="aiExplanation" class="analysis">--</div>
                <div class="warning">
                    Analyse informative uniquement. Utilise toujours une gestion du risque stricte.
                </div>
            </div>
        </section>

        <section id="scannerResult" class="card" style="display:none; margin-top:18px;">
            <div class="signal-box">
                <div>
                    <div class="label">Scanner intelligent MEXC</div>
                    <div class="value">Top signaux du moment</div>
                </div>
                <div id="scanCount" class="badge">-- paires</div>
            </div>
            <div id="scannerList" class="scanner-list"></div>
        </section>

        <div class="footer">
            NAMU SignalPro AI • Analyse uniquement • Trading automatique désactivé • Mode démo à ajouter à l’étape suivante.
        </div>
    </main>

    <script>
        let mode = "single";

        function setMode(newMode) {
            mode = newMode;
            document.getElementById("tabSingle").classList.toggle("active", newMode === "single");
            document.getElementById("tabScanner").classList.toggle("active", newMode === "scanner");
            document.getElementById("singleControls").style.display = newMode === "single" ? "grid" : "none";
            document.getElementById("scannerControls").style.display = newMode === "scanner" ? "grid" : "none";
            document.getElementById("result").style.display = "none";
            document.getElementById("scannerResult").style.display = "none";
        }

        function formatValue(v) {
            if (v === null || v === undefined) return "--";
            if (typeof v === "number") return v.toLocaleString("en-US", { maximumFractionDigits: 6 });
            return v;
        }

        function signalClass(signal) {
            return signal.toLowerCase();
        }

        async function analyze() {
            const symbol = document.getElementById("symbol").value;
            const interval = document.getElementById("interval").value;
            const btn = document.getElementById("analyzeBtn");
            const loading = document.getElementById("loading");
            const error = document.getElementById("error");
            const result = document.getElementById("result");
            const scannerResult = document.getElementById("scannerResult");

            btn.disabled = true;
            loading.style.display = "block";
            error.style.display = "none";
            result.style.display = "none";
            scannerResult.style.display = "none";

            try {
                const response = await fetch(`/api/analyze?symbol=${symbol}&interval=${interval}`);
                if (!response.ok) throw new Error("Erreur serveur");

                const data = await response.json();
                const a = data.analysis;

                const signalEl = document.getElementById("signal");
                signalEl.textContent = a.signal;
                signalEl.className = "signal " + signalClass(a.signal);

                document.getElementById("confidence").textContent = `Confiance : ${a.confidence}%`;
                document.getElementById("price").textContent = formatValue(a.price);
                document.getElementById("risk").textContent = a.risk;
                document.getElementById("trend").textContent = a.trend;
                document.getElementById("rsi").textContent = formatValue(a.rsi);
                document.getElementById("support").textContent = formatValue(a.support);
                document.getElementById("resistance").textContent = formatValue(a.resistance);
                document.getElementById("volumeStatus").textContent = `${a.volume_status} x${a.volume_ratio}`;
                document.getElementById("stopLoss").textContent = formatValue(a.stop_loss);
                document.getElementById("takeProfit").textContent = `${formatValue(a.take_profit_1)} / ${formatValue(a.take_profit_2)}`;
                document.getElementById("confirmation").textContent = a.confirmation;
                document.getElementById("aiExplanation").textContent = data.ai_explanation || "Analyse IA indisponible.";

                const warningBox = document.getElementById("warningBox");
                if (a.warnings && a.warnings.length > 0) {
                    warningBox.innerHTML = `<div class="warning">${a.warnings.join("<br>")}</div>`;
                } else {
                    warningBox.innerHTML = "";
                }

                result.style.display = "grid";
            } catch (e) {
                error.textContent = "Impossible de terminer l’analyse. Réessaie dans quelques secondes.";
                error.style.display = "block";
            } finally {
                loading.style.display = "none";
                btn.disabled = false;
            }
        }

        async function scanMarket() {
            const interval = document.getElementById("scanInterval").value;
            const filter = document.getElementById("scanFilter").value;
            const btn = document.getElementById("scanBtn");
            const loading = document.getElementById("loading");
            const error = document.getElementById("error");
            const result = document.getElementById("result");
            const scannerResult = document.getElementById("scannerResult");
            const scannerList = document.getElementById("scannerList");

            btn.disabled = true;
            loading.style.display = "block";
            loading.textContent = "Scanner MEXC en cours... Analyse de plusieurs paires.";
            error.style.display = "none";
            result.style.display = "none";
            scannerResult.style.display = "none";
            scannerList.innerHTML = "";

            try {
                const response = await fetch(`/api/scan?interval=${interval}&filter_signal=${filter}`);
                if (!response.ok) throw new Error("Erreur scanner");

                const data = await response.json();
                document.getElementById("scanCount").textContent = `${data.results.length} paires`;

                scannerList.innerHTML = data.results.map(item => {
                    const a = item.analysis;
                    return `
                        <div class="scan-item">
                            <div class="scan-top">
                                <div>
                                    <div class="pair">${item.symbol}</div>
                                    <div class="label">${item.interval} • tendance ${a.trend}</div>
                                </div>
                                <div class="badge ${signalClass(a.signal)}">${a.signal} • ${a.confidence}%</div>
                            </div>
                            <div class="mini-grid">
                                <div class="mini"><div class="label">Prix</div><div class="value">${formatValue(a.price)}</div></div>
                                <div class="mini"><div class="label">Risque</div><div class="value">${a.risk}</div></div>
                                <div class="mini"><div class="label">RSI</div><div class="value">${formatValue(a.rsi)}</div></div>
                                <div class="mini"><div class="label">Support</div><div class="value">${formatValue(a.support)}</div></div>
                                <div class="mini"><div class="label">Résistance</div><div class="value">${formatValue(a.resistance)}</div></div>
                                <div class="mini"><div class="label">Volume</div><div class="value">${a.volume_status} x${a.volume_ratio}</div></div>
                            </div>
                            <div class="warning">${a.confirmation}</div>
                        </div>
                    `;
                }).join("");

                scannerResult.style.display = "block";
            } catch (e) {
                error.textContent = "Scanner indisponible pour le moment. Réessaie dans quelques secondes.";
                error.style.display = "block";
            } finally {
                loading.style.display = "none";
                loading.textContent = "Analyse en cours... Le premier chargement peut prendre quelques secondes.";
                btn.disabled = false;
            }
        }

        window.addEventListener("load", analyze);
    </script>
</body>
</html>
    """


@app.get("/api/analyze")
def api_analyze(
    symbol: str = Query(default="BTC_USDT"),
    interval: str = Query(default="Min15"),
):
    data = analyze_symbol(symbol, interval, with_ai=True)
    return {
        **data,
        "warning": "Analyse informative uniquement. Ce n’est pas un conseil financier.",
    }


@app.get("/api/scan")
def api_scan(
    interval: str = Query(default="Min15"),
    filter_signal: str = Query(default="ALL"),
):
    results = []

    for symbol in SCAN_SYMBOLS:
        try:
            item = analyze_symbol(symbol, interval, with_ai=False)
            if filter_signal != "ALL" and item["analysis"]["signal"] != filter_signal:
                continue
            results.append(item)
        except Exception:
            continue

    def sort_key(item):
        a = item["analysis"]
        priority = {"BUY": 3, "SELL": 2, "WAIT": 1}.get(a["signal"], 0)
        return (priority, a["confidence"], -abs(a["score"]))

    results = sorted(results, key=sort_key, reverse=True)

    return {
        "interval": interval,
        "count": len(results),
        "results": results[:20],
    }


@app.get("/analyze")
def analyze_legacy(
    symbol: str = Query(default="BTC_USDT"),
    interval: str = Query(default="Min15"),
):
    data = analyze_symbol(symbol, interval, with_ai=True)
    return {
        **data,
        "warning": "Analyse informative uniquement. Ce n’est pas un conseil financier.",
    }
