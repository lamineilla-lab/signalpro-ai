import os

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from openai import OpenAI

from engine import analyze_symbol, scan_market

load_dotenv()

app = FastAPI(title="NAMU SignalPro AI V7")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def ai_explanation(symbol: str, interval: str, result: dict, mtf: dict):
    if not client:
        return "Analyse IA indisponible : clé OpenAI non configurée."

    prompt = f"""
Analyse crypto prudente en français.

Paire: {symbol}
Timeframe: {interval}
Signal: {result.get('signal')}
Confiance: {result.get('confidence')}%
Qualité: {result.get('quality')}
Décision: {result.get('decision')}
Risque: {result.get('risk')}
Score opportunité: {result.get('opportunity_score')}
Score danger: {result.get('danger_score')}
Tendance: {result.get('trend')}
Structure: {result.get('structure')}
Cassure/Rejet: {result.get('breakout')}
Multi-timeframe: {mtf.get('alignment')}
Prix: {result.get('price')}
RSI: {result.get('rsi')}
Support: {result.get('support')}
Résistance: {result.get('resistance')}
Volume: {result.get('volume_status')} x{result.get('volume_ratio')}
Zone d'entrée: {result.get('entry_zone')}
Stop-loss: {result.get('stop_loss')}
Take-profit 1: {result.get('take_profit_1')}
Take-profit 2: {result.get('take_profit_2')}
Risk/Reward 1: {result.get('risk_reward_1')}
Risk/Reward 2: {result.get('risk_reward_2')}
Confirmation: {result.get('confirmation')}
Invalidation: {result.get('invalidation')}
Raisons: {result.get('reasons')}
Avertissements: {result.get('warnings')}

Réponds en 5 points courts :
1. Lecture du signal
2. Opportunité
3. Risque principal
4. Confirmation à attendre
5. Décision prudente
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Tu donnes des analyses trading prudentes. Tu ne promets jamais de gains.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
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
<title>NAMU SignalPro AI V7</title>
<style>
:root{
--bg:#050816;--card:rgba(255,255,255,.08);--card2:rgba(255,255,255,.12);
--text:#f8fafc;--muted:#94a3b8;--gold:#f5c542;--blue:#38bdf8;
--green:#22c55e;--red:#ef4444;--orange:#f97316;--line:rgba(255,255,255,.12)
}
*{box-sizing:border-box}
body{
margin:0;min-height:100vh;font-family:Arial,Helvetica,sans-serif;color:var(--text);
background:radial-gradient(circle at top left,rgba(56,189,248,.24),transparent 35%),
radial-gradient(circle at top right,rgba(245,197,66,.18),transparent 30%),
linear-gradient(135deg,#020617,#0f172a 55%,#020617)
}
.container{width:100%;max-width:1240px;margin:0 auto;padding:28px 16px 48px}
.header,.card{
border:1px solid var(--line);border-radius:28px;
background:linear-gradient(135deg,rgba(255,255,255,.12),rgba(255,255,255,.05));
box-shadow:0 25px 80px rgba(0,0,0,.35)
}
.header{padding:24px}.card{padding:22px;margin-top:18px}
.brand{display:flex;align-items:center;gap:14px;margin-bottom:18px}
.logo{
width:52px;height:52px;border-radius:18px;background:linear-gradient(135deg,var(--gold),var(--blue));
display:flex;align-items:center;justify-content:center;color:#020617;font-weight:900;font-size:20px
}
h1{margin:0;font-size:31px}.subtitle{margin:6px 0 0;color:var(--muted);font-size:14px;line-height:1.5}
.tabs{display:flex;gap:10px;margin:18px 0}
.tab{
border:1px solid var(--line);background:rgba(15,23,42,.8);color:var(--text);
border-radius:999px;padding:12px 16px;font-weight:900;cursor:pointer
}
.tab.active{background:linear-gradient(135deg,var(--gold),#fde68a);color:#020617}
.controls{display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:12px;margin-top:16px}
select,button{width:100%;border:none;border-radius:16px;padding:15px 16px;font-size:15px;outline:none}
select{background:rgba(15,23,42,.9);color:var(--text);border:1px solid var(--line)}
button{cursor:pointer;font-weight:900;color:#020617;background:linear-gradient(135deg,var(--gold),#fde68a)}
button.secondary{background:linear-gradient(135deg,var(--blue),#bae6fd)}button:disabled{opacity:.65}
.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;margin-top:18px}
.signal-box,.scan-top,.demo-top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px}
.signal{font-size:42px;font-weight:900}.badge{padding:10px 14px;border-radius:999px;font-weight:900;font-size:13px;background:rgba(255,255,255,.1)}
.buy{color:var(--green)}.sell{color:var(--red)}.wait{color:var(--orange)}
.metrics,.mini-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.mini-grid{grid-template-columns:repeat(4,1fr)}
.metric,.mini,.scan-item,.trade-item{
padding:14px;border-radius:18px;background:var(--card2);border:1px solid rgba(255,255,255,.08)
}
.label{color:var(--muted);font-size:12px;margin-bottom:7px}.value{font-weight:900;font-size:18px;word-break:break-word}
.analysis{white-space:pre-line;color:#e2e8f0;line-height:1.55;font-size:14px}
.warning{
margin-top:14px;padding:13px;border-radius:16px;background:rgba(249,115,22,.12);
border:1px solid rgba(249,115,22,.28);color:#fed7aa;font-size:13px;line-height:1.5
}
.good{background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.25);color:#bbf7d0}
.bad{background:rgba(239,68,68,.12);border-color:rgba(239,68,68,.25);color:#fecaca}
.loading,.error{display:none;margin-top:18px;padding:18px;border-radius:20px}
.loading{background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.18);color:#bae6fd}
.error{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.25);color:#fecaca}
.scanner-list,.trade-list{display:grid;gap:12px;margin-top:18px}
.demo-balance{font-size:32px;font-weight:900;color:var(--gold)}
.footer{margin-top:18px;color:var(--muted);font-size:12px;text-align:center;line-height:1.5}
@media(max-width:760px){
.controls,.grid,.metrics,.mini-grid{grid-template-columns:1fr}.tabs{flex-direction:column}
h1{font-size:25px}.signal{font-size:36px}.header{padding:20px;border-radius:22px}
}
</style>
</head>
<body>
<main class="container">
<section class="header">
<div class="brand">
<div class="logo">NS</div>
<div>
<h1>NAMU SignalPro AI V7</h1>
<p class="subtitle">Moteur pro : scanner dynamique MEXC, score opportunité, danger, plan de trade, Risk/Reward et compte démo.</p>
</div>
</div>

<div class="tabs">
<button class="tab active" id="tabSingle" onclick="setMode('single')">Analyse simple</button>
<button class="tab" id="tabScanner" onclick="setMode('scanner')">Scanner MEXC</button>
<button class="tab" id="tabDemo" onclick="setMode('demo')">Compte Démo</button>
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
<option value="TON_USDT">TON / USDT</option>
</select>
<select id="interval">
<option value="Min5">5 minutes</option>
<option value="Min15" selected>15 minutes</option>
<option value="Min60">1 heure</option>
</select>
<select><option>Mode prudent V7</option></select>
<button id="analyzeBtn" onclick="analyze()">Analyser</button>
</div>

<div id="scannerControls" class="controls" style="display:none">
<select id="scanInterval">
<option value="Min5">Scanner 5 minutes</option>
<option value="Min15" selected>Scanner 15 minutes</option>
<option value="Min60">Scanner 1 heure</option>
</select>
<select id="scanFilter">
<option value="ALL">Tous</option>
<option value="BUY">Top BUY</option>
<option value="SELL">Top SELL</option>
<option value="WAIT">WAIT</option>
</select>
<select id="minConfidence">
<option value="0">Confiance min 0%</option>
<option value="60">Confiance min 60%</option>
<option value="70" selected>Confiance min 70%</option>
<option value="80">Confiance min 80%</option>
</select>
<button class="secondary" id="scanBtn" onclick="scanMarket()">Scanner V7</button>
</div>

<div id="loading" class="loading">Analyse en cours...</div>
<div id="error" class="error"></div>
</section>

<section id="result" class="grid" style="display:none">
<div class="card">
<div class="signal-box">
<div><div class="label">Signal détecté</div><div id="signal" class="signal">WAIT</div></div>
<div id="confidence" class="badge">--%</div>
</div>
<div class="metrics">
<div class="metric"><div class="label">Prix</div><div id="price" class="value">--</div></div>
<div class="metric"><div class="label">Décision</div><div id="decision" class="value">--</div></div>
<div class="metric"><div class="label">Qualité</div><div id="quality" class="value">--</div></div>
<div class="metric"><div class="label">Risque</div><div id="risk" class="value">--</div></div>
<div class="metric"><div class="label">Opportunité</div><div id="opportunity" class="value">--</div></div>
<div class="metric"><div class="label">Danger</div><div id="danger" class="value">--</div></div>
<div class="metric"><div class="label">Risk/Reward</div><div id="rr" class="value">--</div></div>
<div class="metric"><div class="label">Multi-timeframe</div><div id="mtf" class="value">--</div></div>
<div class="metric"><div class="label">Tendance</div><div id="trend" class="value">--</div></div>
<div class="metric"><div class="label">Cassure/Rejet</div><div id="breakout" class="value">--</div></div>
<div class="metric"><div class="label">Zone d’entrée</div><div id="entryZone" class="value">--</div></div>
<div class="metric"><div class="label">Stop-loss</div><div id="stopLoss" class="value">--</div></div>
<div class="metric"><div class="label">TP1 / TP2</div><div id="takeProfit" class="value">--</div></div>
<div class="metric"><div class="label">Support / Résistance</div><div id="sr" class="value">--</div></div>
</div>
<div class="warning" id="confirmation">--</div>
<div class="warning" id="invalidation">--</div>
<div id="warningBox"></div>
<button style="margin-top:14px" onclick="openDemoFromCurrent()">Entrer en démo avec ce signal</button>
</div>

<div class="card">
<div class="label">Explication IA V7</div>
<div id="aiExplanation" class="analysis">--</div>
<div class="warning">Analyse informative uniquement. Aucun signal ne garantit un gain.</div>
</div>
</section>

<section id="scannerResult" class="card" style="display:none">
<div class="signal-box">
<div><div class="label">Scanner intelligent MEXC V7</div><div class="value">Classement par opportunité, qualité et danger</div></div>
<div id="scanCount" class="badge">-- paires</div>
</div>
<div id="scannerList" class="scanner-list"></div>
</section>

<section id="demoPanel" class="card" style="display:none">
<div class="demo-top">
<div><div class="label">Compte Démo NAMU</div><div id="demoBalance" class="demo-balance">1000 USDT</div></div>
<button onclick="resetDemo()">Réinitialiser</button>
</div>
<div class="metrics">
<div class="metric"><div class="label">Capital initial</div><div class="value">1000 USDT</div></div>
<div class="metric"><div class="label">Risque simulé</div><div class="value">2% par trade</div></div>
<div class="metric"><div class="label">Trades</div><div id="tradeCount" class="value">0</div></div>
<div class="metric"><div class="label">Performance</div><div id="demoPerf" class="value">0 USDT</div></div>
</div>
<div id="tradeList" class="trade-list"></div>
<div class="warning">Compte fictif enregistré localement dans ton navigateur.</div>
</section>

<div class="footer">NAMU SignalPro AI • V7 • Scanner dynamique MEXC • Trading automatique désactivé.</div>
</main>

<script>
let currentSignal=null;
let demo=JSON.parse(localStorage.getItem("namu_demo")||'{"balance":1000,"trades":[]}');

function saveDemo(){localStorage.setItem("namu_demo",JSON.stringify(demo))}
function fmt(v){if(v===null||v===undefined)return"--";if(typeof v==="number")return v.toLocaleString("en-US",{maximumFractionDigits:8});return v}
function cls(s){return String(s||"WAIT").toLowerCase()}
function setMode(m){
["Single","Scanner","Demo"].forEach(x=>document.getElementById("tab"+x).classList.remove("active"));
document.getElementById("tab"+(m==="single"?"Single":m==="scanner"?"Scanner":"Demo")).classList.add("active");
document.getElementById("singleControls").style.display=m==="single"?"grid":"none";
document.getElementById("scannerControls").style.display=m==="scanner"?"grid":"none";
document.getElementById("result").style.display="none";
document.getElementById("scannerResult").style.display="none";
document.getElementById("demoPanel").style.display=m==="demo"?"block":"none";
if(m==="demo")renderDemo();
}
function showLoad(t){document.getElementById("loading").textContent=t;document.getElementById("loading").style.display="block";document.getElementById("error").style.display="none"}
function hideLoad(){document.getElementById("loading").style.display="none"}
function showErr(t){document.getElementById("error").textContent=t;document.getElementById("error").style.display="block"}

async function analyze(){
const symbol=document.getElementById("symbol").value;
const interval=document.getElementById("interval").value;
document.getElementById("analyzeBtn").disabled=true;
showLoad("Analyse V7 en cours...");
document.getElementById("result").style.display="none";
document.getElementById("scannerResult").style.display="none";

try{
const r=await fetch(`/api/analyze?symbol=${symbol}&interval=${interval}`);
if(!r.ok)throw new Error();
const data=await r.json();
const a=data.analysis;
currentSignal=data;

document.getElementById("signal").textContent=a.signal;
document.getElementById("signal").className="signal "+cls(a.signal);
document.getElementById("confidence").textContent=`Confiance : ${a.confidence}%`;
document.getElementById("price").textContent=fmt(a.price);
document.getElementById("decision").textContent=a.decision;
document.getElementById("quality").textContent=a.quality;
document.getElementById("risk").textContent=a.risk;
document.getElementById("opportunity").textContent=a.opportunity_score;
document.getElementById("danger").textContent=a.danger_score;
document.getElementById("rr").textContent=`${fmt(a.risk_reward_1)} / ${fmt(a.risk_reward_2)}`;
document.getElementById("mtf").textContent=data.mtf.alignment;
document.getElementById("trend").textContent=a.trend;
document.getElementById("breakout").textContent=a.breakout;
document.getElementById("entryZone").textContent=a.entry_zone ? `${fmt(a.entry_zone[0])} - ${fmt(a.entry_zone[1])}` : "--";
document.getElementById("stopLoss").textContent=fmt(a.stop_loss);
document.getElementById("takeProfit").textContent=`${fmt(a.take_profit_1)} / ${fmt(a.take_profit_2)}`;
document.getElementById("sr").textContent=`${fmt(a.support)} / ${fmt(a.resistance)}`;
document.getElementById("confirmation").textContent=a.confirmation;
document.getElementById("invalidation").textContent=a.invalidation;
document.getElementById("aiExplanation").textContent=data.ai_explanation||"Analyse IA indisponible.";
document.getElementById("warningBox").innerHTML=(a.warnings&&a.warnings.length)?`<div class="warning">${a.warnings.join("<br>")}</div>`:"";
document.getElementById("result").style.display="grid";
}catch(e){
showErr("Impossible de terminer l’analyse. Réessaie dans quelques secondes.");
}
finally{
hideLoad();
document.getElementById("analyzeBtn").disabled=false;
}
}

async function scanMarket(){
const interval=document.getElementById("scanInterval").value;
const filter=document.getElementById("scanFilter").value;
const min=document.getElementById("minConfidence").value;
document.getElementById("scanBtn").disabled=true;
showLoad("Scanner V7 en cours... récupération dynamique des paires MEXC.");
document.getElementById("result").style.display="none";
document.getElementById("scannerResult").style.display="none";

try{
const r=await fetch(`/api/scan?interval=${interval}&filter_signal=${filter}&min_confidence=${min}&dynamic=true&limit=40`);
if(!r.ok)throw new Error();
const data=await r.json();
document.getElementById("scanCount").textContent=`${data.count} signaux / ${data.symbols_scanned} scannés`;

document.getElementById("scannerList").innerHTML=data.results.map((item,i)=>{
const a=item.analysis;
const box=a.decision==="Entrée possible"?"good":a.risk==="élevé"?"bad":"warning";
return `<div class="scan-item">
<div class="scan-top">
<div><div class="pair">${i+1}. ${item.symbol}</div><div class="label">${item.interval} • MTF ${item.mtf.alignment} • ${a.breakout}</div></div>
<div class="badge ${cls(a.signal)}">${a.signal} • ${a.confidence}%</div>
</div>
<div class="mini-grid">
<div class="mini"><div class="label">Décision</div><div class="value">${a.decision}</div></div>
<div class="mini"><div class="label">Qualité</div><div class="value">${a.quality}</div></div>
<div class="mini"><div class="label">Opportunité</div><div class="value">${a.opportunity_score}</div></div>
<div class="mini"><div class="label">Danger</div><div class="value">${a.danger_score}</div></div>
<div class="mini"><div class="label">Prix</div><div class="value">${fmt(a.price)}</div></div>
<div class="mini"><div class="label">R/R</div><div class="value">${fmt(a.risk_reward_1)} / ${fmt(a.risk_reward_2)}</div></div>
<div class="mini"><div class="label">Volume</div><div class="value">${a.volume_status} x${a.volume_ratio}</div></div>
<div class="mini"><div class="label">Entrée</div><div class="value">${a.entry_zone ? fmt(a.entry_zone[0])+" - "+fmt(a.entry_zone[1]) : "--"}</div></div>
</div>
<div class="${box}">${a.confirmation}<br>${a.invalidation}</div>
<button onclick='demoEnter(${JSON.stringify(item).replaceAll("'","&apos;")})'>Entrer en démo</button>
</div>`;
}).join("");

document.getElementById("scannerResult").style.display="block";
}catch(e){
showErr("Scanner indisponible. Réessaie dans quelques secondes.");
}
finally{
hideLoad();
document.getElementById("scanBtn").disabled=false;
}
}

function openDemoFromCurrent(){if(!currentSignal)return;demoEnter(currentSignal)}
function demoEnter(item){
const a=item.analysis;
if(a.signal==="WAIT"){alert("Signal WAIT : mieux vaut attendre.");return}
const riskAmount=demo.balance*.02;
const trade={
id:Date.now(),symbol:item.symbol,side:a.signal,entry:a.price,sl:a.stop_loss,tp:a.take_profit_1,
risk:riskAmount,status:"OPEN",pnl:0,time:new Date().toLocaleString(),quality:a.quality,decision:a.decision,
rr:a.risk_reward_1
};
demo.trades.unshift(trade);
saveDemo();
setMode("demo");
}
function closeTrade(id,result){
const t=demo.trades.find(x=>x.id===id);
if(!t||t.status!=="OPEN")return;
if(result==="TP"){t.status="TP";t.pnl=t.risk*2}
if(result==="SL"){t.status="SL";t.pnl=-t.risk}
if(result==="MANUAL"){t.status="MANUAL";t.pnl=0}
demo.balance+=t.pnl;
saveDemo();
renderDemo();
}
function resetDemo(){
if(confirm("Réinitialiser le compte démo à 1000 USDT ?")){
demo={balance:1000,trades:[]};
saveDemo();
renderDemo();
}
}
function renderDemo(){
document.getElementById("demoBalance").textContent=fmt(demo.balance)+" USDT";
document.getElementById("tradeCount").textContent=demo.trades.length;
document.getElementById("demoPerf").textContent=fmt(demo.balance-1000)+" USDT";
document.getElementById("tradeList").innerHTML=demo.trades.length?demo.trades.map(t=>`
<div class="trade-item">
<div class="scan-top">
<div><div class="pair">${t.symbol} ${t.side}</div><div class="label">${t.time} • ${t.status} • ${t.quality||""}</div></div>
<div class="badge">${fmt(t.pnl)} USDT</div>
</div>
<div class="mini-grid">
<div class="mini"><div class="label">Entrée</div><div class="value">${fmt(t.entry)}</div></div>
<div class="mini"><div class="label">SL</div><div class="value">${fmt(t.sl)}</div></div>
<div class="mini"><div class="label">TP</div><div class="value">${fmt(t.tp)}</div></div>
<div class="mini"><div class="label">R/R</div><div class="value">${fmt(t.rr)}</div></div>
</div>
${t.status==="OPEN"?`<div class="controls" style="grid-template-columns:1fr 1fr 1fr;margin-top:12px">
<button onclick="closeTrade(${t.id},'TP')">TP touché</button>
<button onclick="closeTrade(${t.id},'SL')">SL touché</button>
<button onclick="closeTrade(${t.id},'MANUAL')">Clôture manuelle</button>
</div>`:""}
</div>`).join(""):"<div class='warning'>Aucun trade démo pour le moment.</div>";
}
window.addEventListener("load",analyze);
</script>
</body>
</html>
    """


@app.get("/api/analyze")
def api_analyze(symbol: str = Query(default="BTC_USDT"), interval: str = Query(default="Min15")):
    base = analyze_symbol(symbol, interval)

    try:
        explanation = ai_explanation(
            symbol.upper(),
            interval,
            base["analysis"],
            base["mtf"],
        )
    except Exception:
        explanation = "Analyse IA indisponible pour le moment."

    return {
        **base,
        "ai_explanation": explanation,
        "warning": "Analyse informative uniquement. Ce n’est pas un conseil financier.",
        "engine": "V7-A Pro",
    }


@app.get("/api/scan")
def api_scan(
    interval: str = Query(default="Min15"),
    filter_signal: str = Query(default="ALL"),
    min_confidence: int = Query(default=70),
    dynamic: bool = Query(default=True),
    limit: int = Query(default=40),
):
    return scan_market(
        interval=interval,
        filter_signal=filter_signal,
        min_confidence=min_confidence,
        dynamic=dynamic,
        limit=limit,
    )


@app.get("/analyze")
def analyze_legacy(symbol: str = Query(default="BTC_USDT"), interval: str = Query(default="Min15")):
    return api_analyze(symbol, interval)
