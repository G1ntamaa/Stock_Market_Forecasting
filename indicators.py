"""
indicators.py  —  Technical indicator calculations
All computed from raw OHLCV data — no external TA library needed.
"""
import pandas as pd
import numpy as np


# ── Moving Averages ──────────────────────────────────────────
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ── RSI ──────────────────────────────────────────────────────
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── MACD ─────────────────────────────────────────────────────
def macd(series: pd.Series, fast=12, slow=26, signal=9) -> dict:
    ema_fast   = ema(series, fast)
    ema_slow   = ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return {
        "macd":      macd_line,
        "signal":    signal_line,
        "histogram": histogram,
    }


# ── Bollinger Bands ──────────────────────────────────────────
def bollinger_bands(series: pd.Series, window=20, num_std=2) -> dict:
    mid   = sma(series, window)
    std   = series.rolling(window=window, min_periods=1).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    # Bandwidth as % of mid price
    bandwidth = ((upper - lower) / mid) * 100
    return {"upper": upper, "mid": mid, "lower": lower, "bandwidth": bandwidth}


# ── ATR (Average True Range) ─────────────────────────────────
def atr(df: pd.DataFrame, period=14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ── OBV (On-Balance Volume) ──────────────────────────────────
def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()


# ── Stochastic Oscillator ────────────────────────────────────
def stochastic(df: pd.DataFrame, k_period=14, d_period=3) -> dict:
    low_min  = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return {"k": k, "d": d}


# ── Support & Resistance ─────────────────────────────────────
def support_resistance(df: pd.DataFrame, window=10) -> dict:
    """Detect local pivot highs and lows as S/R levels."""
    highs = df["High"].rolling(window=window, center=True).max()
    lows  = df["Low"].rolling(window=window, center=True).min()
    pivot_highs = df["High"][df["High"] == highs].dropna()
    pivot_lows  = df["Low"][df["Low"]  == lows].dropna()
    # Cluster nearby levels
    resistance = _cluster_levels(pivot_highs.values)
    support    = _cluster_levels(pivot_lows.values)
    return {"resistance": resistance, "support": support}

def _cluster_levels(levels, tolerance=0.02):
    """Merge price levels within tolerance %."""
    if len(levels) == 0:
        return []
    levels = sorted(set(levels))
    clusters, current = [], [levels[0]]
    for lv in levels[1:]:
        if (lv - current[-1]) / current[-1] <= tolerance:
            current.append(lv)
        else:
            clusters.append(np.mean(current))
            current = [lv]
    clusters.append(np.mean(current))
    return clusters[-5:]  # return top 5 meaningful levels


# ── Volume Anomaly Detection ─────────────────────────────────
def volume_anomaly(df: pd.DataFrame, z_threshold=2.5) -> pd.Series:
    """Flag days where volume is a Z-score outlier."""
    vol = df["Volume"]
    z   = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-10)
    return z.abs() > z_threshold


# ── Master: compute all indicators ──────────────────────────
def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 5:
        return df
    close = df["Close"]
    df = df.copy()

    # Moving averages
    df["SMA20"]  = sma(close, 20)
    df["SMA50"]  = sma(close, 50)
    df["EMA12"]  = ema(close, 12)
    df["EMA26"]  = ema(close, 26)
    df["EMA200"] = ema(close, 200)

    # RSI
    df["RSI"] = rsi(close)

    # MACD
    m = macd(close)
    df["MACD"]          = m["macd"]
    df["MACD_Signal"]   = m["signal"]
    df["MACD_Hist"]     = m["histogram"]

    # Bollinger Bands
    bb = bollinger_bands(close)
    df["BB_Upper"] = bb["upper"]
    df["BB_Mid"]   = bb["mid"]
    df["BB_Lower"] = bb["lower"]
    df["BB_BW"]    = bb["bandwidth"]

    # ATR & OBV
    df["ATR"] = atr(df)
    df["OBV"] = obv(df)

    # Stochastic
    stoch = stochastic(df)
    df["Stoch_K"] = stoch["k"]
    df["Stoch_D"] = stoch["d"]

    # Volume anomaly
    df["Vol_Anomaly"] = volume_anomaly(df)

    # Price position within BB (0=lower, 1=upper)
    df["BB_Pos"] = (close - bb["lower"]) / (bb["upper"] - bb["lower"] + 1e-10)

    return df
