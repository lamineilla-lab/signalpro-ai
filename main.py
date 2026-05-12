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


def get_mexc_klines(symbol: str = "BTC_USDT", interval: str = "Min15"):
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {"interval": interval, "limit": 200}

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    if not data.get("success"):
        raise Exception("Erreur MEXC : données indisponibles")

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
        reasons.append("Les moyennes mobiles ne donnent pas encore une tendance parfaitement claire.")

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

    warnings = []
    if ema50 > price * 3 or ema50 < price / 3:
        warnings.append("Attention : certaines données semblent anormales. Vérifie la paire ou le contrat MEXC avant toute décision.")

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
        "warnings": warnings,
    }


def ai_explanation(symbol: str, interval: str, result: dict):
    if not client:
        return "Analyse IA indisponible : clé OpenAI non configurée."

    prompt = f"""
Tu es un assistant d'analyse crypto prudent.
Explique ce signal en français simple.

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
Avertissements: {result['warnings']}

Réponds de manière courte avec :
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
        }

        * {
            box-sizing: border-box;
        }

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
            max-width: 1050px;
            margin: 0 auto;
            padding: 28px 16px 48px;
        }

        .header {
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.12);
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

        h1 {
            margin: 0;
            font-size: 31px;
            letter-spacing: -0.5px;
        }

        .subtitle {
            margin: 6px 0 0;
            color: var(--muted);
            font-size: 14px;
        }

        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr auto;
            gap: 12px;
            margin-top: 20px;
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
            border: 1px solid rgba(255,255,255,0.12);
        }

        button {
            cursor: pointer;
            font-weight: 800;
            color: #020617;
            background: linear-gradient(135deg, var(--gold), #fde68a);
            box-shadow: 0 12px 30px rgba(245,197,66,0.18);
        }

        button:disabled {
            opacity: 0.65;
            cursor: not-allowed;
        }

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
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 20px 60px rgba(0,0,0,0.25);
        }

        .signal-box {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 18px;
        }

        .signal {
            font-size: 42px;
            font-weight: 900;
            letter-spacing: -1px;
        }

        .badge {
            padding: 10px 14px;
            border-radius: 999px;
            font-weight: 800;
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

        .label {
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 7px;
        }

        .value {
            font-weight: 800;
            font-size: 18px;
            word-break: break-word;
        }

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
        }

        .footer {
            margin-top: 18px;
            color: var(--muted);
            font-size: 12px;
            text-align: center;
            line-height: 1.5;
        }

        .loading {
            display: none;
            margin-top: 18px;
            padding: 18px;
            border-radius: 20px;
            background: rgba(56,189,248,0.10);
            border: 1px solid rgba(56,189,248,0.18);
            color: #bae6fd;
        }

        .error {
            display: none;
            margin-top: 18px;
            padding: 18px;
            border-radius: 20px;
            background: rgba(239,68,68,0.12);
            border: 1px solid rgba(239,68,68,0.25);
            color: #fecaca;
        }

        @media (max-width: 760px) {
            .controls {
                grid-template-columns: 1fr;
            }

            .grid {
                grid-template-columns: 1fr;
            }

            h1 {
                font-size: 25px;
            }

            .signal {
                font-size: 36px;
            }

            .metrics {
                grid-template-columns: 1fr;
            }

            .header {
                padding: 20px;
                border-radius: 22px;
            }
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
                    <p class="subtitle">Analyse crypto intelligente avec données MEXC, indicateurs techniques et explication IA.</p>
                </div>
            </div>

            <div class="controls">
                <select id="symbol">
                    <option value="BTC_USDT">BTC / USDT</option>
                    <option value="ETH_USDT">ETH / USDT</option>
                    <option value="SOL_USDT">SOL / USDT</option>
                    <option value="RSR_USDT">RSR / USDT</option>
                </select>

                <select id="interval">
                    <option value="Min5">5 minutes</option>
                    <option value="Min15" selected>15 minutes</option>
                    <option value="Min60">1 heure</option>
                </select>

                <button id="analyzeBtn" onclick="analyze()">Analyser</button>
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
                    <div class="metric">
                        <div class="label">Prix actuel</div>
                        <div id="price" class="value">--</div>
                    </div>
                    <div class="metric">
                        <div class="label">RSI</div>
                        <div id="rsi" class="value">--</div>
                    </div>
                    <div class="metric">
                        <div class="label">EMA 20</div>
                        <div id="ema20" class="value">--</div>
                    </div>
                    <div class="metric">
                        <div class="label">EMA 50</div>
                        <div id="ema50" class="value">--</div>
                    </div>
                    <div class="metric">
                        <div class="label">MACD</div>
                        <div id="macd" class="value">--</div>
                    </div>
                    <div class="metric">
                        <div class="label">MACD Signal</div>
                        <div id="macdSignal" class="value">--</div>
                    </div>
                    <div class="metric">
                        <div class="label">Stop-loss</div>
                        <div id="stopLoss" class="value">--</div>
                    </div>
                    <div class="metric">
                        <div class="label">Take-profit 1 / 2</div>
                        <div id="takeProfit" class="value">--</div>
                    </div>
                </div>

                <div id="warningBox"></div>
            </div>

            <div class="card">
                <div class="label">Explication IA</div>
                <div id="aiExplanation" class="analysis">--</div>

                <div class="warning">
                    Cette analyse est informative uniquement. Elle ne garantit aucun gain et ne remplace pas ta propre gestion du risque.
                </div>
            </div>
        </section>

        <div class="footer">
            NAMU SignalPro AI • Version analyse uniquement • Trading automatique désactivé pour la sécurité.
        </div>
    </main>

    <script>
        function formatValue(v) {
            if (v === null || v === undefined) return "--";
            if (typeof v === "number") return v.toLocaleString("en-US", { maximumFractionDigits: 6 });
            return v;
        }

        async function analyze() {
            const symbol = document.getElementById("symbol").value;
            const interval = document.getElementById("interval").value;
            const btn = document.getElementById("analyzeBtn");
            const loading = document.getElementById("loading");
            const error = document.getElementById("error");
            const result = document.getElementById("result");

            btn.disabled = true;
            loading.style.display = "block";
            error.style.display = "none";
            result.style.display = "none";

            try {
                const response = await fetch(`/api/analyze?symbol=${symbol}&interval=${interval}`);
                if (!response.ok) {
                    throw new Error("Erreur serveur");
                }

                const data = await response.json();
                const a = data.analysis;

                const signalEl = document.getElementById("signal");
                signalEl.textContent = a.signal;
                signalEl.className = "signal " + a.signal.toLowerCase();

                document.getElementById("confidence").textContent = `Confiance : ${a.confidence}%`;
                document.getElementById("price").textContent = formatValue(a.price);
                document.getElementById("rsi").textContent = formatValue(a.rsi);
                document.getElementById("ema20").textContent = formatValue(a.ema20);
                document.getElementById("ema50").textContent = formatValue(a.ema50);
                document.getElementById("macd").textContent = formatValue(a.macd);
                document.getElementById("macdSignal").textContent = formatValue(a.macd_signal);
                document.getElementById("stopLoss").textContent = formatValue(a.stop_loss);
                document.getElementById("takeProfit").textContent = `${formatValue(a.take_profit_1)} / ${formatValue(a.take_profit_2)}`;
                document.getElementById("aiExplanation").textContent = data.ai_explanation;

                const warningBox = document.getElementById("warningBox");
                if (a.warnings && a.warnings.length > 0) {
                    warningBox.innerHTML = `<div class="warning">${a.warnings.join("<br>")}</div>`;
                } else {
                    warningBox.innerHTML = "";
                }

                result.style.display = "grid";
            } catch (e) {
                error.textContent = "Impossible de terminer l’analyse. Vérifie la paire ou réessaie dans quelques secondes.";
                error.style.display = "block";
            } finally {
                loading.style.display = "none";
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


@app.get("/analyze")
def analyze_legacy(
    symbol: str = Query(default="BTC_USDT"),
    interval: str = Query(default="Min15"),
):
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
