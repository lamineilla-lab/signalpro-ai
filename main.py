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
    "ENA_USDT", "RSR_USDT", "PEPE_USDT", "WIF_USDT", "TON_USDT",
    "OP_USDT", "ARB_USDT", "APT_USDT", "NEAR_USDT", "INJ_USDT",
    "FET_USDT", "SEI_USDT", "TIA_USDT", "LTC_USDT", "BCH_USDT"
]


def get_mexc_klines(symbol: str = "BTC_USDT", interval: str = "Min15"):
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {"interval": interval, "limit": 200}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if not data.get("success"):
        raise Exception(f"Données MEXC indisponibles pour {symbol}")

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
    price = float(df.iloc[-1]["close"])
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    return {
        "support": round(support, 6),
        "resistance": round(resistance, 6),
        "distance_support_pct": round(((price - support) / price) * 100, 2) if price else 0,
        "distance_resistance_pct": round(((resistance - price) / price) * 100, 2) if price else 0,
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
        reasons.append("Tendance neutre ou mal alignée.")

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
        score += 12 if score > 0 else -12 if score < 0 else 0
        reasons.append("Volume fort : mouvement mieux confirmé.")
    elif vol["volume_status"] == "faible":
        score = int(score * 0.65)
        warnings.append("Volume faible : risque de faux signal.")
        reasons.append("Volume faible, confirmation limitée.")

    if sr["distance_resistance_pct"] < 0.8 and score > 0:
        score -= 25
        warnings.append("Prix très proche d'une résistance : attention au rejet.")
    if sr["distance_support_pct"] < 0.8 and score < 0:
        score += 25
        warnings.append("Prix très proche d'un support : attention au rebond.")

    if ema50 > price * 3 or ema50 < price / 3:
        warnings.append("Données possiblement anormales. Vérifie la paire MEXC avant décision.")

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

    risk = "élevé" if risk_score >= 60 else "moyen" if risk_score >= 30 else "faible"

    if signal == "BUY":
        confirmation = f"Attendre une clôture au-dessus de {round(price + atr * 0.25, 6)} avec volume au moins normal."
    elif signal == "SELL":
        confirmation = f"Attendre une clôture sous {round(price - atr * 0.25, 6)} avec volume au moins normal."
    else:
        confirmation = "Attendre un meilleur alignement EMA/MACD ou une cassure claire."

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


def analyze_symbol(symbol: str, interval: str):
    df = get_mexc_klines(symbol=symbol.upper(), interval=interval)
    df = calculate_indicators(df)
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "analysis": generate_signal(df),
    }


def multi_timeframe(symbol: str):
    frames = ["Min5", "Min15", "Min60"]
    data = []

    for tf in frames:
        try:
            data.append(analyze_symbol(symbol, tf))
        except Exception:
            pass

    buy_count = sum(1 for x in data if x["analysis"]["signal"] == "BUY")
    sell_count = sum(1 for x in data if x["analysis"]["signal"] == "SELL")
    wait_count = sum(1 for x in data if x["analysis"]["signal"] == "WAIT")

    if buy_count >= 2:
        alignment = "haussier"
    elif sell_count >= 2:
        alignment = "baissier"
    elif wait_count >= 2:
        alignment = "neutre"
    else:
        alignment = "mixte"

    return {
        "alignment": alignment,
        "frames": data,
    }


def ai_explanation(symbol: str, interval: str, result: dict, mtf: dict = None):
    if not client:
        return "Analyse IA indisponible : clé OpenAI non configurée."

    prompt = f"""
Explique ce signal crypto en français simple, court et utile.

Paire: {symbol}
Timeframe principal: {interval}
Signal: {result['signal']}
Confiance: {result['confidence']}%
Tendance: {result['trend']}
Risque: {result['risk']}
Prix: {result['price']}
RSI: {result['rsi']}
Support: {result['support']}
Résistance: {result['resistance']}
Volume: {result['volume_status']} x{result['volume_ratio']}
Confirmation: {result['confirmation']}
Multi-timeframe: {mtf['alignment'] if mtf else 'non disponible'}
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
            {"role": "system", "content": "Tu donnes des analyses trading prudentes, sans promettre de gains."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
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
:root{--bg:#050816;--card:rgba(255,255,255,.08);--card2:rgba(255,255,255,.12);--text:#f8fafc;--muted:#94a3b8;--gold:#f5c542;--blue:#38bdf8;--green:#22c55e;--red:#ef4444;--orange:#f97316;--line:rgba(255,255,255,.12)}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;font-family:Arial,Helvetica,sans-serif;color:var(--text);background:radial-gradient(circle at top left,rgba(56,189,248,.24),transparent 35%),radial-gradient(circle at top right,rgba(245,197,66,.18),transparent 30%),linear-gradient(135deg,#020617,#0f172a 55%,#020617)}
.container{width:100%;max-width:1180px;margin:0 auto;padding:28px 16px 48px}
.header,.card{border:1px solid var(--line);border-radius:28px;background:linear-gradient(135deg,rgba(255,255,255,.12),rgba(255,255,255,.05));box-shadow:0 25px 80px rgba(0,0,0,.35)}
.header{padding:24px}.card{padding:22px;margin-top:18px}
.brand{display:flex;align-items:center;gap:14px;margin-bottom:18px}
.logo{width:52px;height:52px;border-radius:18px;background:linear-gradient(135deg,var(--gold),var(--blue));display:flex;align-items:center;justify-content:center;color:#020617;font-weight:900;font-size:20px}
h1{margin:0;font-size:31px}.subtitle{margin:6px 0 0;color:var(--muted);font-size:14px;line-height:1.5}
.tabs{display:flex;gap:10px;margin:18px 0}.tab{border:1px solid var(--line);background:rgba(15,23,42,.8);color:var(--text);border-radius:999px;padding:12px 16px;font-weight:900;cursor:pointer}.tab.active{background:linear-gradient(135deg,var(--gold),#fde68a);color:#020617}
.controls{display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:12px;margin-top:16px}
select,button{width:100%;border:none;border-radius:16px;padding:15px 16px;font-size:15px;outline:none}
select{background:rgba(15,23,42,.9);color:var(--text);border:1px solid var(--line)}
button{cursor:pointer;font-weight:900;color:#020617;background:linear-gradient(135deg,var(--gold),#fde68a)}
button.secondary{background:linear-gradient(135deg,var(--blue),#bae6fd)}button:disabled{opacity:.65}
.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;margin-top:18px}
.signal-box,.scan-top,.demo-top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px}
.signal{font-size:42px;font-weight:900}.badge{padding:10px 14px;border-radius:999px;font-weight:900;font-size:13px;background:rgba(255,255,255,.1)}
.buy{color:var(--green)}.sell{color:var(--red)}.wait{color:var(--orange)}
.metrics,.mini-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.mini-grid{grid-template-columns:repeat(3,1fr)}
.metric,.mini,.scan-item,.trade-item{padding:14px;border-radius:18px;background:var(--card2);border:1px solid rgba(255,255,255,.08)}
.label{color:var(--muted);font-size:12px;margin-bottom:7px}.value{font-weight:900;font-size:18px;word-break:break-word}
.analysis{white-space:pre-line;color:#e2e8f0;line-height:1.55;font-size:14px}
.warning{margin-top:14px;padding:13px;border-radius:16px;background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.28);color:#fed7aa;font-size:13px;line-height:1.5}
.loading,.error{display:none;margin-top:18px;padding:18px;border-radius:20px}.loading{background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.18);color:#bae6fd}.error{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.25);color:#fecaca}
.scanner-list,.trade-list{display:grid;gap:12px;margin-top:18px}
.demo-balance{font-size:32px;font-weight:900;color:var(--gold)}
.footer{margin-top:18px;color:var(--muted);font-size:12px;text-align:center;line-height:1.5}
@media(max-width:760px){.controls,.grid,.metrics,.mini-grid{grid-template-columns:1fr}.tabs{flex-direction:column}h1{font-size:25px}.signal{font-size:36px}.header{padding:20px;border-radius:22px}}
</style>
</head>
<body>
<main class="container">
<section class="header">
<div class="brand"><div class="logo">NS</div><div><h1>NAMU SignalPro AI</h1><p class="subtitle">Scanner MEXC intelligent, multi-timeframe, gestion du risque et compte démo fictif.</p></div></div>

<div class="tabs">
<button class="tab active" id="tabSingle" onclick="setMode('single')">Analyse simple</button>
<button class="tab" id="tabScanner" onclick="setMode('scanner')">Scanner MEXC</button>
<button class="tab" id="tabDemo" onclick="setMode('demo')">Compte Démo</button>
</div>

<div id="singleControls" class="controls">
<select id="symbol">
<option value="BTC_USDT">BTC / USDT</option><option value="ETH_USDT">ETH / USDT</option><option value="SOL_USDT">SOL / USDT</option>
<option value="XRP_USDT">XRP / USDT</option><option value="DOGE_USDT">DOGE / USDT</option><option value="SUI_USDT">SUI / USDT</option>
<option value="ENA_USDT">ENA / USDT</option><option value="RSR_USDT">RSR / USDT</option><option value="PEPE_USDT">PEPE / USDT</option>
</select>
<select id="interval"><option value="Min5">5 minutes</option><option value="Min15" selected>15 minutes</option><option value="Min60">1 heure</option></select>
<select id="dummy"><option>Mode prudent</option></select>
<button id="analyzeBtn" onclick="analyze()">Analyser</button>
</div>

<div id="scannerControls" class="controls" style="display:none">
<select id="scanInterval"><option value="Min5">Scanner 5 minutes</option><option value="Min15" selected>Scanner 15 minutes</option><option value="Min60">Scanner 1 heure</option></select>
<select id="scanFilter"><option value="ALL">Tous</option><option value="BUY">Top BUY</option><option value="SELL">Top SELL</option><option value="WAIT">WAIT</option></select>
<select id="minConfidence"><option value="0">Confiance min 0%</option><option value="60">Confiance min 60%</option><option value="70" selected>Confiance min 70%</option><option value="80">Confiance min 80%</option></select>
<button class="secondary" id="scanBtn" onclick="scanMarket()">Scanner</button>
</div>

<div id="loading" class="loading">Analyse en cours...</div>
<div id="error" class="error"></div>
</section>

<section id="result" class="grid" style="display:none">
<div class="card">
<div class="signal-box"><div><div class="label">Signal détecté</div><div id="signal" class="signal">WAIT</div></div><div id="confidence" class="badge">--%</div></div>
<div class="metrics">
<div class="metric"><div class="label">Prix</div><div id="price" class="value">--</div></div>
<div class="metric"><div class="label">Risque</div><div id="risk" class="value">--</div></div>
<div class="metric"><div class="label">Tendance</div><div id="trend" class="value">--</div></div>
<div class="metric"><div class="label">Multi-timeframe</div><div id="mtf" class="value">--</div></div>
<div class="metric"><div class="label">RSI</div><div id="rsi" class="value">--</div></div>
<div class="metric"><div class="label">Volume</div><div id="volumeStatus" class="value">--</div></div>
<div class="metric"><div class="label">Support</div><div id="support" class="value">--</div></div>
<div class="metric"><div class="label">Résistance</div><div id="resistance" class="value">--</div></div>
<div class="metric"><div class="label">Stop-loss</div><div id="stopLoss" class="value">--</div></div>
<div class="metric"><div class="label">TP1 / TP2</div><div id="takeProfit" class="value">--</div></div>
</div>
<div class="warning" id="confirmation">--</div>
<div id="warningBox"></div>
<button style="margin-top:14px" onclick="openDemoFromCurrent()">Entrer en démo avec ce signal</button>
</div>
<div class="card"><div class="label">Explication IA</div><div id="aiExplanation" class="analysis">--</div><div class="warning">Analyse informative uniquement. Ne risque jamais plus que ce que tu peux perdre.</div></div>
</section>

<section id="scannerResult" class="card" style="display:none">
<div class="signal-box"><div><div class="label">Scanner intelligent MEXC</div><div class="value">Top signaux filtrés</div></div><div id="scanCount" class="badge">-- paires</div></div>
<div id="scannerList" class="scanner-list"></div>
</section>

<section id="demoPanel" class="card" style="display:none">
<div class="demo-top"><div><div class="label">Compte Démo NAMU</div><div id="demoBalance" class="demo-balance">1000 USDT</div></div><button onclick="resetDemo()">Réinitialiser</button></div>
<div class="metrics">
<div class="metric"><div class="label">Capital initial</div><div class="value">1000 USDT</div></div>
<div class="metric"><div class="label">Risque simulé</div><div class="value">2% par trade</div></div>
<div class="metric"><div class="label">Trades démo</div><div id="tradeCount" class="value">0</div></div>
<div class="metric"><div class="label">Performance</div><div id="demoPerf" class="value">0 USDT</div></div>
</div>
<div id="tradeList" class="trade-list"></div>
<div class="warning">Le compte démo est fictif et enregistré seulement dans ton navigateur. Il sert à t’entraîner à respecter les signaux, SL et TP.</div>
</section>

<div class="footer">NAMU SignalPro AI • V5 • Scanner, multi-timeframe, compte démo • Trading automatique désactivé.</div>
</main>

<script>
let currentSignal=null;
let demo=JSON.parse(localStorage.getItem("namu_demo")||'{"balance":1000,"trades":[]}');

function saveDemo(){localStorage.setItem("namu_demo",JSON.stringify(demo))}
function fmt(v){if(v===null||v===undefined)return"--";if(typeof v==="number")return v.toLocaleString("en-US",{maximumFractionDigits:6});return v}
function cls(s){return String(s||"WAIT").toLowerCase()}
function setMode(m){
["single","scanner","demo"].forEach(x=>document.getElementById("tab"+(x==="single"?"Single":x==="scanner"?"Scanner":"Demo")).classList.remove("active"));
document.getElementById("tab"+(m==="single"?"Single":m==="scanner"?"Scanner":"Demo")).classList.add("active");
document.getElementById("singleControls").style.display=m==="single"?"grid":"none";
document.getElementById("scannerControls").style.display=m==="scanner"?"grid":"none";
document.getElementById("result").style.display="none";document.getElementById("scannerResult").style.display="none";document.getElementById("demoPanel").style.display=m==="demo"?"block":"none";
if(m==="demo")renderDemo();
}
function showLoad(t){document.getElementById("loading").textContent=t;document.getElementById("loading").style.display="block";document.getElementById("error").style.display="none"}
function hideLoad(){document.getElementById("loading").style.display="none"}
function showErr(t){document.getElementById("error").textContent=t;document.getElementById("error").style.display="block"}

async function analyze(){
const symbol=document.getElementById("symbol").value, interval=document.getElementById("interval").value;
document.getElementById("analyzeBtn").disabled=true;showLoad("Analyse en cours...");
document.getElementById("result").style.display="none";document.getElementById("scannerResult").style.display="none";
try{
const r=await fetch(`/api/analyze?symbol=${symbol}&interval=${interval}`); if(!r.ok)throw new Error();
const data=await r.json(); const a=data.analysis; currentSignal=data;
document.getElementById("signal").textContent=a.signal;document.getElementById("signal").className="signal "+cls(a.signal);
document.getElementById("confidence").textContent=`Confiance : ${a.confidence}%`;
document.getElementById("price").textContent=fmt(a.price);document.getElementById("risk").textContent=a.risk;document.getElementById("trend").textContent=a.trend;
document.getElementById("mtf").textContent=data.mtf.alignment;document.getElementById("rsi").textContent=fmt(a.rsi);
document.getElementById("volumeStatus").textContent=`${a.volume_status} x${a.volume_ratio}`;
document.getElementById("support").textContent=fmt(a.support);document.getElementById("resistance").textContent=fmt(a.resistance);
document.getElementById("stopLoss").textContent=fmt(a.stop_loss);document.getElementById("takeProfit").textContent=`${fmt(a.take_profit_1)} / ${fmt(a.take_profit_2)}`;
document.getElementById("confirmation").textContent=a.confirmation;document.getElementById("aiExplanation").textContent=data.ai_explanation||"Analyse IA indisponible.";
document.getElementById("warningBox").innerHTML=(a.warnings&&a.warnings.length)?`<div class="warning">${a.warnings.join("<br>")}</div>`:"";
document.getElementById("result").style.display="grid";
}catch(e){showErr("Impossible de terminer l’analyse. Réessaie dans quelques secondes.")}
finally{hideLoad();document.getElementById("analyzeBtn").disabled=false}
}

async function scanMarket(){
const interval=document.getElementById("scanInterval").value, filter=document.getElementById("scanFilter").value, min=document.getElementById("minConfidence").value;
document.getElementById("scanBtn").disabled=true;showLoad("Scanner MEXC en cours... 25 paires analysées.");
document.getElementById("result").style.display="none";document.getElementById("scannerResult").style.display="none";
try{
const r=await fetch(`/api/scan?interval=${interval}&filter_signal=${filter}&min_confidence=${min}`); if(!r.ok)throw new Error();
const data=await r.json();document.getElementById("scanCount").textContent=`${data.results.length} paires`;
document.getElementById("scannerList").innerHTML=data.results.map((item,i)=>{const a=item.analysis;return`
<div class="scan-item">
<div class="scan-top"><div><div class="pair">${i+1}. ${item.symbol}</div><div class="label">${item.interval} • MTF ${item.mtf.alignment} • tendance ${a.trend}</div></div><div class="badge ${cls(a.signal)}">${a.signal} • ${a.confidence}%</div></div>
<div class="mini-grid">
<div class="mini"><div class="label">Prix</div><div class="value">${fmt(a.price)}</div></div>
<div class="mini"><div class="label">Risque</div><div class="value">${a.risk}</div></div>
<div class="mini"><div class="label">RSI</div><div class="value">${fmt(a.rsi)}</div></div>
<div class="mini"><div class="label">Support</div><div class="value">${fmt(a.support)}</div></div>
<div class="mini"><div class="label">Résistance</div><div class="value">${fmt(a.resistance)}</div></div>
<div class="mini"><div class="label">Volume</div><div class="value">${a.volume_status} x${a.volume_ratio}</div></div>
</div>
<div class="warning">${a.confirmation}</div>
<button onclick='demoEnter(${JSON.stringify(item).replaceAll("'","&apos;")})'>Entrer en démo</button>
</div>`}).join("");
document.getElementById("scannerResult").style.display="block";
}catch(e){showErr("Scanner indisponible. Réessaie dans quelques secondes.")}
finally{hideLoad();document.getElementById("scanBtn").disabled=false}
}

function openDemoFromCurrent(){if(!currentSignal)return;demoEnter(currentSignal)}
function demoEnter(item){
const a=item.analysis;if(a.signal==="WAIT"){alert("Signal WAIT : mieux vaut attendre un meilleur signal.");return}
const riskAmount=demo.balance*0.02;
const trade={id:Date.now(),symbol:item.symbol,side:a.signal,entry:a.price,sl:a.stop_loss,tp:a.take_profit_1,risk:riskAmount,status:"OPEN",pnl:0,time:new Date().toLocaleString()};
demo.trades.unshift(trade);saveDemo();setMode("demo");
}
function closeTrade(id,result){
const t=demo.trades.find(x=>x.id===id);if(!t||t.status!=="OPEN")return;
if(result==="TP"){t.status="TP";t.pnl=t.risk*2}
if(result==="SL"){t.status="SL";t.pnl=-t.risk}
if(result==="MANUAL"){t.status="MANUAL";t.pnl=0}
demo.balance+=t.pnl;saveDemo();renderDemo();
}
function resetDemo(){if(confirm("Réinitialiser le compte démo à 1000 USDT ?")){demo={balance:1000,trades:[]};saveDemo();renderDemo()}}
function renderDemo(){
document.getElementById("demoBalance").textContent=fmt(demo.balance)+" USDT";
document.getElementById("tradeCount").textContent=demo.trades.length;
document.getElementById("demoPerf").textContent=fmt(demo.balance-1000)+" USDT";
document.getElementById("tradeList").innerHTML=demo.trades.length?demo.trades.map(t=>`
<div class="trade-item">
<div class="scan-top"><div><div class="pair">${t.symbol} ${t.side}</div><div class="label">${t.time} • ${t.status}</div></div><div class="badge">${fmt(t.pnl)} USDT</div></div>
<div class="mini-grid"><div class="mini"><div class="label">Entrée</div><div class="value">${fmt(t.entry)}</div></div><div class="mini"><div class="label">SL</div><div class="value">${fmt(t.sl)}</div></div><div class="mini"><div class="label">TP</div><div class="value">${fmt(t.tp)}</div></div></div>
${t.status==="OPEN"?`<div class="controls" style="grid-template-columns:1fr 1fr 1fr;margin-top:12px"><button onclick="closeTrade(${t.id},'TP')">TP touché</button><button onclick="closeTrade(${t.id},'SL')">SL touché</button><button onclick="closeTrade(${t.id},'MANUAL')">Clôture manuelle</button></div>`:""}
</div>`).join(""):"<div class='warning'>Aucun trade démo pour le moment. Entre depuis un signal ou depuis le scanner.</div>";
}
window.addEventListener("load",analyze);
</script>
</body>
</html>
    """


@app.get("/api/analyze")
def api_analyze(symbol: str = Query(default="BTC_USDT"), interval: str = Query(default="Min15")):
    base = analyze_symbol(symbol, interval)
    mtf = multi_timeframe(symbol)
    try:
        explanation = ai_explanation(symbol.upper(), interval, base["analysis"], mtf)
    except Exception:
        explanation = "Analyse IA indisponible pour le moment."

    return {
        **base,
        "mtf": mtf,
        "ai_explanation": explanation,
        "warning": "Analyse informative uniquement. Ce n’est pas un conseil financier.",
    }


@app.get("/api/scan")
def api_scan(
    interval: str = Query(default="Min15"),
    filter_signal: str = Query(default="ALL"),
    min_confidence: int = Query(default=70),
):
    results = []

    for symbol in SCAN_SYMBOLS:
        try:
            item = analyze_symbol(symbol, interval)
            item["mtf"] = multi_timeframe(symbol)

            if filter_signal != "ALL" and item["analysis"]["signal"] != filter_signal:
                continue

            if item["analysis"]["confidence"] < min_confidence:
                continue

            results.append(item)
        except Exception:
            continue

    def sort_key(item):
        a = item["analysis"]
        mtf_bonus = 10 if item["mtf"]["alignment"] in ["haussier", "baissier"] else 0
        priority = {"BUY": 3, "SELL": 2, "WAIT": 1}.get(a["signal"], 0)
        return (priority, a["confidence"] + mtf_bonus, abs(a["score"]))

    results = sorted(results, key=sort_key, reverse=True)

    return {
        "interval": interval,
        "count": len(results),
        "results": results[:25],
    }


@app.get("/analyze")
def analyze_legacy(symbol: str = Query(default="BTC_USDT"), interval: str = Query(default="Min15")):
    return api_analyze(symbol, interval)
