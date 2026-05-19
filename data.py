"""
data.py  —  Real market data fetcher
Uses yfinance (15-min delayed, free)
Supports NSE India (.NS suffix) and US stocks
"""
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# ── Popular stocks quick-select ──────────────────────────────
INDIAN_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS":                 "TCS.NS",
    "Infosys":             "INFY.NS",
    "HDFC Bank":           "HDFCBANK.NS",
    "ICICI Bank":          "ICICIBANK.NS",
    "Wipro":               "WIPRO.NS",
    "Bajaj Finance":       "BAJFINANCE.NS",
    "HCL Technologies":    "HCLTECH.NS",
    "Kotak Mahindra":      "KOTAKBANK.NS",
    "Axis Bank":           "AXISBANK.NS",
    "Maruti Suzuki":       "MARUTI.NS",
    "Tata Motors":         "TATAMOTORS.NS",
    "Sun Pharma":          "SUNPHARMA.NS",
    "Adani Enterprises":   "ADANIENT.NS",
    "ITC":                 "ITC.NS",
    "Larsen & Toubro":     "LT.NS",
    "Hindustan Unilever":  "HINDUNILVR.NS",
    "Titan":               "TITAN.NS",
    "Nestle India":        "NESTLEIND.NS",
    "Asian Paints":        "ASIANPAINT.NS",
}

US_STOCKS = {
    "Apple":       "AAPL",
    "Microsoft":   "MSFT",
    "Nvidia":      "NVDA",
    "Google":      "GOOGL",
    "Amazon":      "AMZN",
    "Meta":        "META",
    "Tesla":       "TSLA",
    "Netflix":     "NFLX",
    "AMD":         "AMD",
    "Intel":       "INTC",
}

INDICES = {
    "Nifty 50":    "^NSEI",
    "Sensex":      "^BSESN",
    "S&P 500":     "^GSPC",
    "Nasdaq":      "^IXIC",
    "Dow Jones":   "^DJI",
}

PERIODS = {
    "1 Month":  ("1mo",  "1d"),
    "3 Months": ("3mo",  "1d"),
    "6 Months": ("6mo",  "1d"),
    "1 Year":   ("1y",   "1d"),
    "2 Years":  ("2y",   "1wk"),
    "5 Years":  ("5y",   "1wk"),
}


# ── Cached fetchers ──────────────────────────────────────────
@st.cache_data(ttl=900)  # 15-minute cache matches market delay
def fetch_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV history for a ticker."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=900)
def fetch_info(ticker: str) -> dict:
    """Fetch fundamental data for a ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return info
    except Exception:
        return {}


@st.cache_data(ttl=1800)  # 30-min cache for news
def fetch_news(ticker: str) -> list:
    """Fetch recent news headlines for sentiment analysis."""
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        return news[:15]
    except Exception:
        return []


def get_current_price(ticker: str) -> dict:
    """Get live price snapshot."""
    try:
        t = yf.Ticker(ticker)
        fast = t.fast_info
        return {
            "price":    round(fast.last_price, 2) if fast.last_price else None,
            "prev":     round(fast.previous_close, 2) if fast.previous_close else None,
            "high":     round(fast.day_high, 2) if fast.day_high else None,
            "low":      round(fast.day_low, 2) if fast.day_low else None,
            "volume":   int(fast.three_month_average_volume or 0),
            "mkt_cap":  fast.market_cap if hasattr(fast, "market_cap") else None,
        }
    except Exception:
        return {}
