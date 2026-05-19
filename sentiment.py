"""
sentiment.py  —  News sentiment analysis
Uses VADER (fast, no GPU, works offline after install)
VADER is tuned for short financial headlines.
"""
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

_analyzer = SentimentIntensityAnalyzer()


def score_headline(text: str) -> dict:
    """Score a single headline. Returns compound, pos, neg, neu."""
    if not text:
        return {"compound": 0, "pos": 0, "neg": 0, "neu": 1}
    scores = _analyzer.polarity_scores(text)
    return scores


def analyze_news(news_list: list) -> dict:
    """
    Analyze a list of yfinance news dicts.
    Returns aggregated sentiment and per-headline breakdown.
    """
    if not news_list:
        return {
            "score": 0,
            "label": "Neutral",
            "color": "#CBCBCB",
            "headlines": [],
            "avg_compound": 0,
        }

    rows = []
    for item in news_list:
        title = item.get("title", "")
        if not title:
            continue
        sc = score_headline(title)
        ts = item.get("providerPublishTime", 0)
        dt = datetime.fromtimestamp(ts).strftime("%d %b %Y") if ts else "—"
        rows.append({
            "headline":  title,
            "date":      dt,
            "compound":  sc["compound"],
            "positive":  sc["pos"],
            "negative":  sc["neg"],
            "neutral":   sc["neu"],
            "label":     _label(sc["compound"]),
        })

    if not rows:
        return {"score": 0, "label": "Neutral", "color": "#CBCBCB", "headlines": [], "avg_compound": 0}

    df = pd.DataFrame(rows)
    avg = round(df["compound"].mean(), 4)

    # Weighted score: more recent headlines count more
    weights = np.linspace(1.0, 0.5, len(df))
    weighted_avg = round(np.average(df["compound"], weights=weights), 4)

    # Map to 0–100 scale
    sentiment_score = int((weighted_avg + 1) / 2 * 100)

    label, color = _label_color(weighted_avg)

    return {
        "score":        sentiment_score,
        "label":        label,
        "color":        color,
        "avg_compound": weighted_avg,
        "headlines":    df.to_dict("records"),
        "pos_count":    int((df["compound"] > 0.05).sum()),
        "neg_count":    int((df["compound"] < -0.05).sum()),
        "neu_count":    int((df["compound"].between(-0.05, 0.05)).sum()),
    }


def _label(compound: float) -> str:
    if compound >= 0.05:  return "Positive"
    if compound <= -0.05: return "Negative"
    return "Neutral"


def _label_color(compound: float) -> tuple:
    if compound >= 0.15:  return "Bullish",         "#2d6a4f"
    if compound >= 0.05:  return "Mildly Bullish",  "#52b788"
    if compound <= -0.15: return "Bearish",          "#c0392b"
    if compound <= -0.05: return "Mildly Bearish",   "#e07070"
    return "Neutral", "#6D8196"
