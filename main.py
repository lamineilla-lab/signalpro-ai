import requests
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


DEFAULT_SYMBOLS = [
    "BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT", "DOGE_USDT",
    "BNB_USDT", "ADA_USDT", "LINK_USDT", "AVAX_USDT", "SUI_USDT",
    "ENA_USDT", "RSR_USDT", "PEPE_USDT", "WIF_USDT", "TON_USDT",
    "OP_USDT", "ARB_USDT", "APT_USDT", "NEAR_USDT", "INJ_USDT",
    "FET_USDT", "SEI_USDT", "TIA_USDT", "LTC_USDT", "BCH_USDT",
    "ORDI_USDT", "FIL_USDT", "DOT_USDT", "TRX_USDT", "UNI_USDT"
]


def get_dynamic_mexc_symbols(limit: int = 50):
    try:
        url = "https://contract.mexc.com/api/v1/contract/detail"
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            return DEFAULT_SYMBOLS[:limit]

        symbols = []
        for item in data.get("data", []):
            symbol = item.get("symbol")
            state = item.get("state")
            quote_coin = item.get("quoteCoin")

            if symbol and quote_coin == "USDT" and state == 0:
                symbols.append(symbol)

        symbols = list(dict.fromkeys(symbols))
        return symbols[:limit] if symbols else DEFAULT_SYMBOLS[:limit]

    except Exception:
        return DEFAULT_SYMBOLS[:limit]


def get_mexc_klines(symbol: str = "BTC_USDT", interval: str = "Min15", limit: int = 220):
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {"interval": interval, "limit": limit}

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

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

    df["ema9"] = EMAIndicator(close=df["close"], window=9).ema_indicator()
    df["ema20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["ema100"] = EMAIndicator(close=df["close"], window=100).ema_indicator()

    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    df["atr"] = atr.average_true_range()

    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_low"] = bb.bollinger_lband()

    df["volume_avg20"] = df["volume"].rolling(20).mean()
    df["change_pct"] = df["close"].pct_change() * 100

    return df.dropna()


def support_resistance(df: pd.DataFrame):
    recent = df.tail(60)
    price = float(df.iloc[-1]["close"])

    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    distance_support = ((price - support) / price) * 100 if price else 0
    distance_resistance = ((resistance - price) / price) * 100 if price else 0

    return {
        "support": round(support, 8),
        "resistance": round(resistance, 8),
        "distance_support_pct": round(distance_support, 2),
        "distance_resistance_pct": round(distance_resistance, 2),
    }


def volume_analysis(df: pd.DataFrame):
    last = df.iloc[-1]

    volume = float(last["volume"])
    avg = float(last["volume_avg20"]) if last["volume_avg20"] else 0
    ratio = volume / avg if avg > 0 else 0

    if ratio >= 2.2:
        status = "très fort"
    elif ratio >= 1.35:
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
    ema9 = float(last["ema9"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema100 = float(last["ema100"])

    if price > ema9 > ema20 > ema50 > ema100:
        return "fortement haussière"
    if price > ema20 > ema50 > ema100:
        return "haussière"
    if price < ema9 < ema20 < ema50 < ema100:
        return "fortement baissière"
    if price < ema20 < ema50 < ema100:
        return "baissière"
    return "neutre"


def market_structure(df: pd.DataFrame):
    recent = df.tail(24)
    highs = recent["high"].values
    lows = recent["low"].values

    higher_high = highs[-1] > highs[-6] > highs[-12]
    higher_low = lows[-1] > lows[-6] > lows[-12]

    lower_high = highs[-1] < highs[-6] < highs[-12]
    lower_low = lows[-1] < lows[-6] < lows[-12]

    if higher_high and higher_low:
        return "structure haussière"
    if lower_high and lower_low:
        return "structure baissière"
    return "structure mixte"


def detect_breakout_rejection(df: pd.DataFrame):
    last = df.iloc[-1]
    previous = df.iloc[-2]
    sr = support_resistance(df)

    close = float(last["close"])
    high = float(last["high"])
    low = float(last["low"])
    open_price = float(last["open"])

    resistance = sr["resistance"]
    support = sr["support"]

    candle_body = abs(close - open_price)
    candle_range = max(high - low, 0.00000001)
    body_ratio = candle_body / candle_range

    breakout_up = close > resistance and body_ratio > 0.45
    breakout_down = close < support and body_ratio > 0.45

    rejection_resistance = high >= resistance and close < resistance and close < previous["close"]
    rejection_support = low <= support and close > support and close > previous["close"]

    if breakout_up:
        return "cassure haussière"
    if breakout_down:
        return "cassure baissière"
    if rejection_resistance:
        return "rejet résistance"
    if rejection_support:
        return "rejet support"
    return "aucun signal de cassure"


def signal_quality(confidence, risk, mtf_alignment, warnings, volume_status, decision):
    score = confidence

    if risk == "faible":
        score += 8
    elif risk == "élevé":
        score -= 15

    if mtf_alignment in ["haussier", "baissier"]:
        score += 10
    elif mtf_alignment in ["mixte", "neutre"]:
        score -= 8

    if volume_status in ["fort", "très fort"]:
        score += 7
    elif volume_status == "faible":
        score -= 12

    if warnings:
        score -= 10

    if decision == "Entrée possible":
        score += 8
    elif decision == "Éviter":
        score -= 18

    if score >= 92:
        return "Excellent"
    if score >= 78:
        return "Bon"
    if score >= 62:
        return "Moyen"
    return "Faible"


def build_trade_plan(price, atr, signal, support, resistance):
    if signal == "BUY":
        entry_min = price
        entry_max = price + atr * 0.25
        stop_loss = min(price - atr * 1.45, support)
        take_profit_1 = price + atr * 1.8
        take_profit_2 = price + atr * 3.0
        invalidation = f"Annuler le BUY si clôture sous {round(stop_loss, 8)}."
    elif signal == "SELL":
        entry_min = price - atr * 0.25
        entry_max = price
        stop_loss = max(price + atr * 1.45, resistance)
        take_profit_1 = price - atr * 1.8
        take_profit_2 = price - atr * 3.0
        invalidation = f"Annuler le SELL si clôture au-dessus de {round(stop_loss, 8)}."
    else:
        return {
            "entry_zone": None,
            "stop_loss": None,
            "take_profit_1": None,
            "take_profit_2": None,
            "risk_reward_1": None,
            "risk_reward_2": None,
            "invalidation": "Aucun plan actif : attendre un meilleur signal.",
        }

    risk = abs(price - stop_loss)
    reward_1 = abs(take_profit_1 - price)
    reward_2 = abs(take_profit_2 - price)

    rr1 = reward_1 / risk if risk > 0 else None
    rr2 = reward_2 / risk if risk > 0 else None

    return {
        "entry_zone": [round(entry_min, 8), round(entry_max, 8)],
        "stop_loss": round(stop_loss, 8),
        "take_profit_1": round(take_profit_1, 8),
        "take_profit_2": round(take_profit_2, 8),
        "risk_reward_1": round(rr1, 2) if rr1 else None,
        "risk_reward_2": round(rr2, 2) if rr2 else None,
        "invalidation": invalidation,
    }


def estimate_time_to_targets(df: pd.DataFrame, price: float, signal: str, tp1, tp2, sl, confidence: float, volume_ratio: float, mtf_alignment: str):
    if signal == "WAIT" or not tp1 or not tp2 or not sl:
        return {
            "tp1_time": "Non disponible",
            "tp2_time": "Non disponible",
            "sl_risk_time": "Non disponible",
            "projection_note": "Pas de projection active tant que le signal reste WAIT.",
        }

    recent = df.tail(20).copy()
    recent["abs_move"] = recent["close"].diff().abs()
    avg_move = float(recent["abs_move"].mean()) if recent["abs_move"].mean() else 0

    if avg_move <= 0:
        return {
            "tp1_time": "Non disponible",
            "tp2_time": "Non disponible",
            "sl_risk_time": "Non disponible",
            "projection_note": "Volatilité insuffisante pour estimer un temps fiable.",
        }

    distance_tp1 = abs(tp1 - price)
    distance_tp2 = abs(tp2 - price)
    distance_sl = abs(price - sl)

    speed_factor = 1.0

    if volume_ratio >= 2:
        speed_factor *= 0.65
    elif volume_ratio >= 1.35:
        speed_factor *= 0.8
    elif volume_ratio < 0.8:
        speed_factor *= 1.35

    if confidence >= 85:
        speed_factor *= 0.8
    elif confidence < 65:
        speed_factor *= 1.25

    if mtf_alignment in ["haussier", "baissier"]:
        speed_factor *= 0.85
    elif mtf_alignment in ["mixte", "neutre"]:
        speed_factor *= 1.25

    def candles_to_minutes(distance):
        candles = max(1, distance / avg_move)
        estimated_minutes = candles * 5 * speed_factor
        low = int(max(5, estimated_minutes * 0.65))
        high = int(max(low + 5, estimated_minutes * 1.45))
        return low, high

    tp1_low, tp1_high = candles_to_minutes(distance_tp1)
    tp2_low, tp2_high = candles_to_minutes(distance_tp2)
    sl_low, sl_high = candles_to_minutes(distance_sl)

    def fmt_time(low, high):
        if high < 60:
            return f"{low} - {high} min"
        return f"{round(low/60, 1)}h - {round(high/60, 1)}h"

    return {
        "tp1_time": fmt_time(tp1_low, tp1_high),
        "tp2_time": fmt_time(tp2_low, tp2_high),
        "sl_risk_time": fmt_time(sl_low, sl_high),
        "projection_note": "Temps estimé selon volatilité récente, volume, confiance et alignement multi-timeframe.",
    }


def sentiment_from_signal(signal: str, confidence: float, opportunity_score: float, danger_score: float):
    base = confidence

    if opportunity_score:
        base = (base * 0.7) + (opportunity_score * 0.3)

    if danger_score:
        base -= danger_score * 0.12

    base = max(5, min(95, round(base)))

    if signal == "BUY":
        bullish = base
        bearish = 100 - bullish
    elif signal == "SELL":
        bearish = base
        bullish = 100 - bearish
    else:
        bullish = 50
        bearish = 50

    return {
        "bullish_pct": bullish,
        "bearish_pct": bearish,
        "label": "Bullish" if bullish > bearish else "Bearish" if bearish > bullish else "Neutre",
    }


def generate_signal(df: pd.DataFrame, mtf_alignment: str = "non disponible"):
    last = df.iloc[-1]

    price = float(last["close"])
    rsi = float(last["rsi"])
    ema9 = float(last["ema9"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    macd = float(last["macd"])
    macd_signal = float(last["macd_signal"])
    macd_diff = float(last["macd_diff"])
    atr = float(last["atr"])
    bb_high = float(last["bb_high"])
    bb_low = float(last["bb_low"])

    sr = support_resistance(df)
    vol = volume_analysis(df)
    trend = trend_status(df)
    structure = market_structure(df)
    breakout = detect_breakout_rejection(df)

    score = 0
    danger_score = 0
    opportunity_score = 0
    reasons = []
    warnings = []

    if price > ema9 > ema20 > ema50:
        score += 32
        opportunity_score += 18
        reasons.append("Alignement EMA haussier propre.")
    elif price > ema20 > ema50:
        score += 24
        opportunity_score += 12
        reasons.append("Tendance haussière correcte.")
    elif price < ema9 < ema20 < ema50:
        score -= 32
        opportunity_score += 18
        reasons.append("Alignement EMA baissier propre.")
    elif price < ema20 < ema50:
        score -= 24
        opportunity_score += 12
        reasons.append("Tendance baissière correcte.")
    else:
        danger_score += 10
        warnings.append("EMA mal alignées : marché moins propre.")

    if 45 <= rsi <= 62:
        score += 12
        opportunity_score += 10
        reasons.append("RSI dans une zone exploitable.")
    elif 35 <= rsi < 45:
        score += 5
        opportunity_score += 4
    elif rsi < 30:
        score += 6
        danger_score += 18
        warnings.append("RSI très bas : possible rebond violent.")
    elif 62 < rsi <= 70:
        score -= 5
        danger_score += 8
        warnings.append("RSI un peu élevé.")
    elif rsi > 70:
        score -= 20
        danger_score += 22
        warnings.append("RSI suracheté.")

    if macd > macd_signal and macd_diff > 0:
        score += 24
        opportunity_score += 14
        reasons.append("MACD positif.")
    elif macd < macd_signal and macd_diff < 0:
        score -= 24
        opportunity_score += 14
        reasons.append("MACD négatif.")
    else:
        danger_score += 8
        warnings.append("MACD hésitant.")

    if vol["volume_status"] == "très fort":
        score += 16 if score > 0 else -16 if score < 0 else 0
        opportunity_score += 16
        reasons.append("Volume très fort.")
    elif vol["volume_status"] == "fort":
        score += 10 if score > 0 else -10 if score < 0 else 0
        opportunity_score += 10
        reasons.append("Volume fort.")
    elif vol["volume_status"] == "faible":
        score = int(score * 0.62)
        danger_score += 26
        warnings.append("Volume faible : faux signal possible.")

    if sr["distance_resistance_pct"] < 0.7 and score > 0:
        score -= 28
        danger_score += 24
        warnings.append("Prix trop proche d’une résistance.")
    elif sr["distance_resistance_pct"] > 1.5 and score > 0:
        opportunity_score += 8

    if sr["distance_support_pct"] < 0.7 and score < 0:
        score += 28
        danger_score += 24
        warnings.append("Prix trop proche d’un support.")
    elif sr["distance_support_pct"] > 1.5 and score < 0:
        opportunity_score += 8

    if breakout == "cassure haussière":
        score += 18
        opportunity_score += 18
        reasons.append("Cassure haussière détectée.")
    elif breakout == "cassure baissière":
        score -= 18
        opportunity_score += 18
        reasons.append("Cassure baissière détectée.")
    elif breakout == "rejet résistance":
        score -= 14
        danger_score += 16
        warnings.append("Rejet sur résistance.")
    elif breakout == "rejet support":
        score += 14
        danger_score += 16
        warnings.append("Rejet sur support.")

    if price >= bb_high and score > 0:
        score -= 12
        danger_score += 12
        warnings.append("Prix proche de la bande haute Bollinger.")
    if price <= bb_low and score < 0:
        score += 12
        danger_score += 12
        warnings.append("Prix proche de la bande basse Bollinger.")

    if mtf_alignment == "haussier" and score > 0:
        score += 18
        opportunity_score += 15
        reasons.append("Multi-timeframe aligné haussier.")
    elif mtf_alignment == "baissier" and score < 0:
        score -= 18
        opportunity_score += 15
        reasons.append("Multi-timeframe aligné baissier.")
    elif mtf_alignment in ["mixte", "neutre"]:
        danger_score += 14
        warnings.append("Multi-timeframe non aligné.")

    if "haussière" in structure and score > 0:
        score += 8
        opportunity_score += 8
    elif "baissière" in structure and score < 0:
        score -= 8
        opportunity_score += 8
    else:
        danger_score += 8

    if ema50 > price * 3 or ema50 < price / 3:
        danger_score += 50
        warnings.append("Données possiblement anormales.")

    if score >= 55:
        signal = "BUY"
        confidence = min(96, 50 + score)
    elif score <= -55:
        signal = "SELL"
        confidence = min(96, 50 + abs(score))
    else:
        signal = "WAIT"
        confidence = max(40, min(68, 50 + abs(score) / 3))

    if confidence < 65:
        danger_score += 16
    if signal == "WAIT":
        danger_score += 20

    risk = "élevé" if danger_score >= 58 else "moyen" if danger_score >= 30 else "faible"

    trade_plan = build_trade_plan(
        price=price,
        atr=atr,
        signal=signal,
        support=sr["support"],
        resistance=sr["resistance"],
    )

    if signal == "BUY":
        confirmation = f"Attendre clôture au-dessus de {round(price + atr * 0.22, 8)} avec volume normal/fort."
    elif signal == "SELL":
        confirmation = f"Attendre clôture sous {round(price - atr * 0.22, 8)} avec volume normal/fort."
    else:
        confirmation = "Pas d’entrée : attendre cassure claire, volume et alignement multi-timeframe."

    if signal == "WAIT":
        decision = "Éviter"
    elif risk == "élevé":
        decision = "Attendre confirmation"
    elif trade_plan["risk_reward_1"] and trade_plan["risk_reward_1"] < 1:
        decision = "Attendre confirmation"
        warnings.append("Risk/Reward faible.")
    elif confidence >= 78 and risk in ["faible", "moyen"]:
        decision = "Entrée possible"
    else:
        decision = "Attendre confirmation"

    quality = signal_quality(
        confidence=confidence,
        risk=risk,
        mtf_alignment=mtf_alignment,
        warnings=warnings,
        volume_status=vol["volume_status"],
        decision=decision,
    )

    opportunity_score = max(0, min(100, opportunity_score + confidence / 3 - danger_score / 4))
    danger_score = max(0, min(100, danger_score))

    projection = estimate_time_to_targets(
        df=df,
        price=price,
        signal=signal,
        tp1=trade_plan["take_profit_1"],
        tp2=trade_plan["take_profit_2"],
        sl=trade_plan["stop_loss"],
        confidence=confidence,
        volume_ratio=vol["volume_ratio"],
        mtf_alignment=mtf_alignment,
    )

    sentiment = sentiment_from_signal(
        signal=signal,
        confidence=confidence,
        opportunity_score=opportunity_score,
        danger_score=danger_score,
    )

    return {
        "price": round(price, 8),
        "rsi": round(rsi, 2),
        "ema9": round(ema9, 8),
        "ema20": round(ema20, 8),
        "ema50": round(ema50, 8),
        "macd": round(macd, 8),
        "macd_signal": round(macd_signal, 8),
        "signal": signal,
        "confidence": round(confidence, 2),
        "score": int(score),
        "opportunity_score": round(opportunity_score, 2),
        "danger_score": round(danger_score, 2),
        "trend": trend,
        "structure": structure,
        "breakout": breakout,
        "risk": risk,
        "quality": quality,
        "decision": decision,
        "support": sr["support"],
        "resistance": sr["resistance"],
        "distance_support_pct": sr["distance_support_pct"],
        "distance_resistance_pct": sr["distance_resistance_pct"],
        "volume_status": vol["volume_status"],
        "volume_ratio": vol["volume_ratio"],
        "entry_zone": trade_plan["entry_zone"],
        "stop_loss": trade_plan["stop_loss"],
        "take_profit_1": trade_plan["take_profit_1"],
        "take_profit_2": trade_plan["take_profit_2"],
        "risk_reward_1": trade_plan["risk_reward_1"],
        "risk_reward_2": trade_plan["risk_reward_2"],
        "invalidation": trade_plan["invalidation"],
        "confirmation": confirmation,
        "projection": projection,
        "sentiment": sentiment,
        "reasons": reasons,
        "warnings": warnings,
    }


def multi_timeframe(symbol: str):
    frames = ["Min5", "Min15", "Min60"]
    data = []

    for tf in frames:
        try:
            df = calculate_indicators(get_mexc_klines(symbol, tf, limit=220))
            basic_signal = generate_signal(df, "non disponible")
            data.append({
                "interval": tf,
                "trend": trend_status(df),
                "score": basic_signal["score"],
                "signal": basic_signal["signal"],
            })
        except Exception:
            pass

    bullish = sum(1 for x in data if "haussière" in x["trend"] or x["score"] > 35)
    bearish = sum(1 for x in data if "baissière" in x["trend"] or x["score"] < -35)

    if bullish >= 2:
        alignment = "haussier"
    elif bearish >= 2:
        alignment = "baissier"
    elif len(data) < 2:
        alignment = "faible données"
    else:
        alignment = "mixte"

    return {
        "alignment": alignment,
        "frames": data,
    }


def analyze_symbol(symbol: str, interval: str = "Min15", mtf: dict = None):
    if mtf is None:
        mtf = multi_timeframe(symbol)

    df = calculate_indicators(get_mexc_klines(symbol.upper(), interval, limit=220))
    result = generate_signal(df, mtf.get("alignment", "non disponible"))

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "analysis": result,
        "mtf": mtf,
    }


def scan_market(interval: str = "Min15", filter_signal: str = "ALL", min_confidence: int = 70, dynamic: bool = True, limit: int = 40):
    symbols = get_dynamic_mexc_symbols(limit=limit) if dynamic else DEFAULT_SYMBOLS[:limit]
    results = []

    for symbol in symbols:
        try:
            mtf = multi_timeframe(symbol)
            item = analyze_symbol(symbol, interval, mtf)

            if filter_signal != "ALL" and item["analysis"]["signal"] != filter_signal:
                continue

            if item["analysis"]["confidence"] < min_confidence:
                continue

            results.append(item)
        except Exception:
            continue

    def quality_rank(q):
        return {"Excellent": 4, "Bon": 3, "Moyen": 2, "Faible": 1}.get(q, 0)

    def sort_key(item):
        a = item["analysis"]
        decision_bonus = 30 if a["decision"] == "Entrée possible" else 0
        quality_bonus = quality_rank(a["quality"]) * 15
        return (
            quality_bonus,
            decision_bonus,
            a["opportunity_score"],
            a["confidence"],
            -a["danger_score"],
        )

    results = sorted(results, key=sort_key, reverse=True)

    return {
        "interval": interval,
        "symbols_scanned": len(symbols),
        "count": len(results),
        "results": results[:limit],
    }
