# Stock Market Analyzer

Real-time market analysis dashboard for Indian (NSE) and US stocks with AI-powered health scoring.

## Features

- **Live Price Data** — 15-min delayed via yfinance (free, no API key needed)
- **Candlestick Charts** — with SMA/EMA overlays, Bollinger Bands, volume, RSI
- **Technical Indicators** — RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic, Support/Resistance
- **Stock Health Score (0–100)** — composite of Technical (45%) + Fundamental (35%) + Sentiment (20%)
- **News Sentiment Analysis** — VADER scoring on live news headlines
- **Portfolio Tracker** — track holdings, P&L, and allocation
- **Stock Comparison** — normalised performance chart for up to 6 stocks

## Supported Markets

**Indian (NSE):** RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, etc.
**US:** AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, etc.
**Indices:** ^NSEI (Nifty), ^BSESN (Sensex), ^GSPC (S&P 500)

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
stock-analyzer/
├── app.py          # Streamlit dashboard (5 tabs)
├── data.py         # yfinance data fetching with 15-min cache
├── indicators.py   # RSI, MACD, BB, ATR, OBV, Stochastic, S&R
├── sentiment.py    # VADER news sentiment analysis
├── scoring.py      # Health Score engine (technical + fundamental + sentiment)
├── charts.py       # Plotly chart builders (dark financial theme)
├── portfolio.py    # Portfolio tracker (session-state based)
└── requirements.txt
```

## Health Score Methodology

| Component    | Weight | What it measures |
|---|---|---|
| Technical    | 45%    | RSI, MACD, Moving Avg position, BB position, Volume |
| Fundamental  | 35%    | P/E ratio, Revenue growth, Profit margin, Debt/Equity, 52-week position |
| Sentiment    | 20%    | VADER score on recent news headlines (weighted by recency) |

**Disclaimer:** For informational purposes only. Not investment advice.
