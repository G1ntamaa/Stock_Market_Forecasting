"""
portfolio.py  —  Portfolio tracker using Streamlit session state
No database needed — all in session during the app lifetime.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from data import fetch_history


def init_portfolio():
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []


def add_holding(ticker: str, name: str, qty: float, buy_price: float, buy_date: str):
    init_portfolio()
    st.session_state["portfolio"].append({
        "ticker":    ticker.upper(),
        "name":      name,
        "qty":       qty,
        "buy_price": buy_price,
        "buy_date":  buy_date,
    })


def remove_holding(idx: int):
    init_portfolio()
    st.session_state["portfolio"].pop(idx)


def get_portfolio_df() -> pd.DataFrame:
    init_portfolio()
    holdings = st.session_state["portfolio"]
    if not holdings:
        return pd.DataFrame()

    rows = []
    for h in holdings:
        try:
            df = fetch_history(h["ticker"], "5d", "1d")
            current_price = df["Close"].iloc[-1] if not df.empty else h["buy_price"]
        except Exception:
            current_price = h["buy_price"]

        invested  = h["qty"] * h["buy_price"]
        current   = h["qty"] * current_price
        pnl       = current - invested
        pnl_pct   = (pnl / invested * 100) if invested > 0 else 0

        rows.append({
            "Ticker":        h["ticker"],
            "Name":          h["name"],
            "Qty":           h["qty"],
            "Buy Price":     round(h["buy_price"], 2),
            "Current Price": round(current_price, 2),
            "Invested":      round(invested, 2),
            "Current Value": round(current, 2),
            "P&L":           round(pnl, 2),
            "P&L %":         round(pnl_pct, 2),
        })

    return pd.DataFrame(rows)


def portfolio_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    total_invested = df["Invested"].sum()
    total_current  = df["Current Value"].sum()
    total_pnl      = total_current - total_invested
    total_pnl_pct  = (total_pnl / total_invested * 100) if total_invested > 0 else 0
    winners = (df["P&L"] > 0).sum()
    losers  = (df["P&L"] < 0).sum()
    return {
        "invested":   round(total_invested, 2),
        "current":    round(total_current, 2),
        "pnl":        round(total_pnl, 2),
        "pnl_pct":    round(total_pnl_pct, 2),
        "winners":    int(winners),
        "losers":     int(losers),
        "best":       df.loc[df["P&L %"].idxmax(), "Ticker"] if not df.empty else "—",
        "worst":      df.loc[df["P&L %"].idxmin(), "Ticker"] if not df.empty else "—",
    }
