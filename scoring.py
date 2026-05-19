"""
scoring.py  —  Stock Health Score (0–100)
Combines Technical Score + Fundamental Score + Sentiment Score
Each component is transparent and explainable.
"""
import pandas as pd
import numpy as np


# ── Technical Score (0–100) ──────────────────────────────────
def technical_score(df: pd.DataFrame) -> dict:
    """
    Score based on:
    - RSI position            (25 pts)
    - MACD signal             (20 pts)
    - Price vs Moving Avgs    (25 pts)
    - Bollinger Band position (15 pts)
    - Volume trend            (15 pts)
    """
    if df.empty or len(df) < 20:
        return {"score": 50, "breakdown": {}, "signals": []}

    latest = df.iloc[-1]
    score  = 0
    breakdown = {}
    signals   = []

    # ── RSI (25 pts) ─────────────────────────────────────
    rsi_val = latest.get("RSI", 50)
    if 40 <= rsi_val <= 60:
        rsi_pts = 25   # neutral sweet spot
        signals.append(("RSI", f"{rsi_val:.1f} — Neutral zone", "neutral"))
    elif 30 <= rsi_val < 40 or 60 < rsi_val <= 70:
        rsi_pts = 18
        lbl = "Slightly oversold" if rsi_val < 50 else "Slightly overbought"
        signals.append(("RSI", f"{rsi_val:.1f} — {lbl}", "caution"))
    elif rsi_val < 30:
        rsi_pts = 12   # oversold — risky
        signals.append(("RSI", f"{rsi_val:.1f} — Oversold, potential reversal", "bearish"))
    else:
        rsi_pts = 10   # > 70 overbought
        signals.append(("RSI", f"{rsi_val:.1f} — Overbought", "caution"))
    breakdown["RSI"] = rsi_pts
    score += rsi_pts

    # ── MACD (20 pts) ────────────────────────────────────
    macd_val  = latest.get("MACD", 0)
    macd_sig  = latest.get("MACD_Signal", 0)
    macd_hist = latest.get("MACD_Hist", 0)
    prev_hist = df["MACD_Hist"].iloc[-2] if len(df) > 1 else 0

    if macd_val > macd_sig and macd_hist > 0 and macd_hist > prev_hist:
        macd_pts = 20
        signals.append(("MACD", "Bullish crossover — momentum rising", "bullish"))
    elif macd_val > macd_sig:
        macd_pts = 14
        signals.append(("MACD", "Above signal line — mild bullish", "bullish"))
    elif macd_val < macd_sig and macd_hist < 0 and macd_hist < prev_hist:
        macd_pts = 2
        signals.append(("MACD", "Bearish crossover — momentum falling", "bearish"))
    else:
        macd_pts = 8
        signals.append(("MACD", "Below signal line — mild bearish", "bearish"))
    breakdown["MACD"] = macd_pts
    score += macd_pts

    # ── Price vs Moving Averages (25 pts) ────────────────
    close = latest["Close"]
    ma_pts = 0
    ma_signals = []
    for col, pts, label in [("SMA20", 8, "20-day SMA"), ("SMA50", 9, "50-day SMA"), ("EMA200", 8, "200-day EMA")]:
        val = latest.get(col)
        if val and not pd.isna(val):
            if close > val:
                ma_pts += pts
                ma_signals.append(f"Above {label} ({val:.2f})")
            else:
                ma_signals.append(f"Below {label} ({val:.2f})")
    trend = "bullish" if ma_pts >= 17 else "bearish" if ma_pts <= 8 else "neutral"
    signals.append(("Moving Avgs", " · ".join(ma_signals) if ma_signals else "Insufficient data", trend))
    breakdown["Moving Averages"] = ma_pts
    score += ma_pts

    # ── Bollinger Band Position (15 pts) ─────────────────
    bb_pos = latest.get("BB_Pos", 0.5)
    bb_bw  = latest.get("BB_BW", 5)
    if 0.3 <= bb_pos <= 0.7:
        bb_pts = 15   # in middle — healthy
        signals.append(("Bollinger", f"Price in mid-band — stable ({bb_pos:.2f})", "neutral"))
    elif bb_pos < 0.1:
        bb_pts = 5    # near lower band
        signals.append(("Bollinger", f"Near lower band — oversold pressure ({bb_pos:.2f})", "bearish"))
    elif bb_pos > 0.9:
        bb_pts = 6    # near upper band
        signals.append(("Bollinger", f"Near upper band — overbought ({bb_pos:.2f})", "caution"))
    else:
        bb_pts = 10
        signals.append(("Bollinger", f"Band position: {bb_pos:.2f}", "neutral"))
    breakdown["Bollinger Bands"] = bb_pts
    score += bb_pts

    # ── Volume Trend (15 pts) ────────────────────────────
    if "Volume" in df.columns and len(df) >= 10:
        recent_vol = df["Volume"].iloc[-5:].mean()
        avg_vol    = df["Volume"].iloc[-20:].mean()
        price_up   = df["Close"].iloc[-1] > df["Close"].iloc[-5]
        if recent_vol > avg_vol * 1.2 and price_up:
            vol_pts = 15
            signals.append(("Volume", "Rising price on above-avg volume — conviction", "bullish"))
        elif recent_vol > avg_vol * 1.2 and not price_up:
            vol_pts = 4
            signals.append(("Volume", "Falling price on above-avg volume — distribution", "bearish"))
        elif recent_vol < avg_vol * 0.8:
            vol_pts = 8
            signals.append(("Volume", "Below-average volume — low conviction", "caution"))
        else:
            vol_pts = 11
            signals.append(("Volume", "Normal volume range", "neutral"))
    else:
        vol_pts = 8
    breakdown["Volume"] = vol_pts
    score += vol_pts

    return {
        "score":     min(100, max(0, score)),
        "breakdown": breakdown,
        "signals":   signals,
    }


# ── Fundamental Score (0–100) ────────────────────────────────
def fundamental_score(info: dict) -> dict:
    """
    Score based on key valuation and quality metrics.
    All thresholds are calibrated for Indian + US markets.
    """
    score = 0
    breakdown = {}
    signals   = []

    # ── P/E Ratio (25 pts) ───────────────────────────────
    pe = info.get("trailingPE") or info.get("forwardPE")
    if pe:
        if 10 <= pe <= 25:
            pe_pts = 25
            signals.append(("P/E Ratio", f"{pe:.1f} — Fairly valued", "bullish"))
        elif pe < 10:
            pe_pts = 18
            signals.append(("P/E Ratio", f"{pe:.1f} — Potentially undervalued", "bullish"))
        elif pe <= 40:
            pe_pts = 15
            signals.append(("P/E Ratio", f"{pe:.1f} — Growth premium", "caution"))
        else:
            pe_pts = 5
            signals.append(("P/E Ratio", f"{pe:.1f} — Highly expensive", "bearish"))
    else:
        pe_pts = 12
        signals.append(("P/E Ratio", "Not available", "neutral"))
    breakdown["P/E Ratio"] = pe_pts
    score += pe_pts

    # ── Revenue Growth (20 pts) ──────────────────────────
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        g = rev_growth * 100
        if g >= 20:
            rev_pts = 20
            signals.append(("Revenue Growth", f"{g:.1f}% YoY — Strong growth", "bullish"))
        elif g >= 10:
            rev_pts = 15
            signals.append(("Revenue Growth", f"{g:.1f}% YoY — Healthy growth", "bullish"))
        elif g >= 0:
            rev_pts = 10
            signals.append(("Revenue Growth", f"{g:.1f}% YoY — Flat", "neutral"))
        else:
            rev_pts = 3
            signals.append(("Revenue Growth", f"{g:.1f}% YoY — Declining revenue", "bearish"))
    else:
        rev_pts = 10
        signals.append(("Revenue Growth", "Not available", "neutral"))
    breakdown["Revenue Growth"] = rev_pts
    score += rev_pts

    # ── Profit Margin (20 pts) ───────────────────────────
    margin = info.get("profitMargins")
    if margin is not None:
        m = margin * 100
        if m >= 20:
            mg_pts = 20
            signals.append(("Profit Margin", f"{m:.1f}% — Excellent margins", "bullish"))
        elif m >= 10:
            mg_pts = 15
            signals.append(("Profit Margin", f"{m:.1f}% — Good margins", "bullish"))
        elif m >= 0:
            mg_pts = 8
            signals.append(("Profit Margin", f"{m:.1f}% — Thin margins", "caution"))
        else:
            mg_pts = 2
            signals.append(("Profit Margin", f"{m:.1f}% — Loss-making", "bearish"))
    else:
        mg_pts = 10
        signals.append(("Profit Margin", "Not available", "neutral"))
    breakdown["Profit Margin"] = mg_pts
    score += mg_pts

    # ── Debt to Equity (20 pts) ──────────────────────────
    de = info.get("debtToEquity")
    if de is not None:
        if de < 30:
            de_pts = 20
            signals.append(("Debt/Equity", f"{de:.1f}% — Low debt, strong balance sheet", "bullish"))
        elif de < 80:
            de_pts = 13
            signals.append(("Debt/Equity", f"{de:.1f}% — Moderate leverage", "neutral"))
        elif de < 150:
            de_pts = 7
            signals.append(("Debt/Equity", f"{de:.1f}% — High leverage", "caution"))
        else:
            de_pts = 2
            signals.append(("Debt/Equity", f"{de:.1f}% — Excessive debt", "bearish"))
    else:
        de_pts = 10
        signals.append(("Debt/Equity", "Not available", "neutral"))
    breakdown["Debt/Equity"] = de_pts
    score += de_pts

    # ── 52-Week Position (15 pts) ────────────────────────
    wk52_low  = info.get("fiftyTwoWeekLow")
    wk52_high = info.get("fiftyTwoWeekHigh")
    curr_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if wk52_low and wk52_high and curr_price and wk52_high > wk52_low:
        pos = (curr_price - wk52_low) / (wk52_high - wk52_low)
        if 0.3 <= pos <= 0.7:
            wk_pts = 15
            signals.append(("52-Week Range", f"{pos*100:.0f}% of range — Mid range", "neutral"))
        elif pos < 0.3:
            wk_pts = 12
            signals.append(("52-Week Range", f"{pos*100:.0f}% of range — Near yearly low", "caution"))
        else:
            wk_pts = 10
            signals.append(("52-Week Range", f"{pos*100:.0f}% of range — Near yearly high", "bullish"))
    else:
        wk_pts = 8
        signals.append(("52-Week Range", "Not available", "neutral"))
    breakdown["52-Week Position"] = wk_pts
    score += wk_pts

    return {
        "score":     min(100, max(0, score)),
        "breakdown": breakdown,
        "signals":   signals,
    }


# ── Composite Health Score ────────────────────────────────────
def health_score(tech: dict, fund: dict, sent: dict) -> dict:
    """
    Weighted composite:
    Technical  : 45%
    Fundamental: 35%
    Sentiment  : 20%
    """
    weights = {"technical": 0.45, "fundamental": 0.35, "sentiment": 0.20}
    t_score = tech.get("score", 50)
    f_score = fund.get("score", 50)
    s_score = sent.get("score", 50)

    composite = (
        t_score * weights["technical"]  +
        f_score * weights["fundamental"] +
        s_score * weights["sentiment"]
    )
    composite = int(round(composite))

    if composite >= 75:
        verdict = "Strong"
        color   = "#2d6a4f"
        desc    = "Technicals, fundamentals, and sentiment all align positively."
    elif composite >= 60:
        verdict = "Positive"
        color   = "#52b788"
        desc    = "More strengths than weaknesses. Proceed with caution."
    elif composite >= 45:
        verdict = "Neutral"
        color   = "#6D8196"
        desc    = "Mixed signals. No strong directional conviction."
    elif composite >= 30:
        verdict = "Weak"
        color   = "#e07070"
        desc    = "Multiple negative signals. Review before acting."
    else:
        verdict = "Poor"
        color   = "#c0392b"
        desc    = "Significant red flags across multiple dimensions."

    return {
        "score":   composite,
        "verdict": verdict,
        "color":   color,
        "desc":    desc,
        "weights": weights,
        "components": {
            "Technical":    {"score": t_score,  "weight": "45%"},
            "Fundamental":  {"score": f_score,  "weight": "35%"},
            "Sentiment":    {"score": s_score,  "weight": "20%"},
        },
    }
