"""
charts.py  —  All Plotly chart builders
Professional dark financial theme with the project palette.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Theme constants ──────────────────────────────────────────
BG       = "#0f1117"
PAPER    = "#0f1117"
GRID     = "#1e2130"
TEXT     = "#c8cdd8"
BULL     = "#52b788"   # green
BEAR     = "#e07070"   # red
BLUE     = "#6D8196"
CREAM    = "#d4d4b0"
FONT     = dict(family="IBM Plex Mono, monospace", color=TEXT)

LAYOUT = dict(
    paper_bgcolor=PAPER,
    plot_bgcolor=BG,
    font=FONT,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, color=TEXT),
    yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, color=TEXT),
)


# ── Candlestick + Indicators ─────────────────────────────────
def candlestick_chart(df: pd.DataFrame, ticker: str, show_ma=True, show_bb=True, show_volume=True) -> go.Figure:
    rows  = 3 if show_volume else 2
    specs = [[{"secondary_y": False}]] * rows
    row_h = [0.6, 0.2, 0.2] if show_volume else [0.7, 0.3]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_h,
        specs=specs,
    )

    # ── Candlestick ───────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color=BULL,
        decreasing_line_color=BEAR,
        increasing_fillcolor=BULL,
        decreasing_fillcolor=BEAR,
    ), row=1, col=1)

    # ── Moving averages ───────────────────────────────────
    if show_ma:
        for col, color, name in [
            ("SMA20", "#ffd166", "SMA 20"),
            ("SMA50", "#06d6a0", "SMA 50"),
            ("EMA200","#f4a261", "EMA 200"),
        ]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=name,
                    line=dict(color=color, width=1.2),
                    opacity=0.85,
                ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────
    if show_bb and "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color=BLUE, width=0.8, dash="dot"), opacity=0.5,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color=BLUE, width=0.8, dash="dot"), opacity=0.5,
            fill="tonexty", fillcolor="rgba(109,129,150,0.07)",
        ), row=1, col=1)

    # ── Volume ────────────────────────────────────────────
    if show_volume and "Volume" in df.columns:
        colors = [BULL if c >= o else BEAR
                  for c, o in zip(df["Close"], df["Open"])]
        vol_row = 2
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors, opacity=0.7,
        ), row=vol_row, col=1)

        # Volume anomalies
        if "Vol_Anomaly" in df.columns:
            anom = df[df["Vol_Anomaly"]]
            if not anom.empty:
                fig.add_trace(go.Bar(
                    x=anom.index, y=anom["Volume"],
                    name="Volume Spike",
                    marker_color="#f4a261", opacity=1.0,
                ), row=vol_row, col=1)

    # ── RSI ───────────────────────────────────────────────
    rsi_row = 3 if show_volume else 2
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#c77dff", width=1.5),
        ), row=rsi_row, col=1)
        for level, color in [(70, BEAR), (30, BULL)]:
            fig.add_hline(y=level, line_dash="dash",
                          line_color=color, opacity=0.5, row=rsi_row, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(109,129,150,0.05)",
                      layer="below", row=rsi_row, col=1)

    fig.update_layout(
        **LAYOUT,
        title=dict(text=f"{ticker} — Price Chart", font=dict(size=15, color=CREAM)),
        height=620,
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, color=TEXT)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1, color=TEXT)
    fig.update_yaxes(title_text="RSI", row=rsi_row, col=1, color=TEXT, range=[0, 100])

    return fig


# ── MACD Chart ────────────────────────────────────────────────
def macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "MACD" not in df.columns:
        return fig

    colors = [BULL if v >= 0 else BEAR for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_Hist"], name="Histogram",
        marker_color=colors, opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"], name="MACD",
        line=dict(color="#6D8196", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"], name="Signal",
        line=dict(color="#f4a261", width=1.5),
    ))
    fig.add_hline(y=0, line_color=GRID)
    fig.update_layout(**LAYOUT, title="MACD", height=260)
    return fig


# ── Health Score Gauge ────────────────────────────────────────
def health_gauge(score: int, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 48, "color": color, "family": "IBM Plex Mono"}},
        gauge={
            "axis":      {"range": [0, 100], "tickcolor": TEXT, "tickfont": {"color": TEXT}},
            "bar":       {"color": color, "thickness": 0.25},
            "bgcolor":   BG,
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#2a0e0e"},
                {"range": [30, 45], "color": "#2a1a0e"},
                {"range": [45, 60], "color": "#0e1a2a"},
                {"range": [60, 75], "color": "#0e2a1a"},
                {"range": [75,100], "color": "#0e3020"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor=PAPER,
        font=FONT,
        margin=dict(l=30, r=30, t=30, b=10),
        height=240,
    )
    return fig


# ── Score Breakdown Bar ───────────────────────────────────────
def score_breakdown_bar(components: dict) -> go.Figure:
    names  = list(components.keys())
    scores = [v["score"] for v in components.values()]
    colors = [BULL if s >= 60 else BEAR if s < 40 else BLUE for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker_color=colors,
        text=[f"{s}/100" for s in scores],
        textposition="outside",
        textfont=dict(color=TEXT, size=12),
    ))
    # 
    fig.update_layout(
    paper_bgcolor=PAPER,
    plot_bgcolor=BG,
    font=FONT,
    height=180,
    xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False),
    yaxis=dict(showgrid=False),
    margin=dict(l=10, r=60, t=10, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── Sentiment Donut ───────────────────────────────────────────
def sentiment_donut(pos: int, neg: int, neu: int) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Positive", "Negative", "Neutral"],
        values=[pos, neg, neu],
        hole=0.6,
        marker_colors=[BULL, BEAR, BLUE],
        textfont=dict(size=12, color=TEXT),
    ))
    total = pos + neg + neu or 1
    fig.update_layout(
        paper_bgcolor=PAPER,
        font=FONT,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11)),
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(
            text=f"<b>{total}</b><br>news",
            x=0.5, y=0.5, font=dict(size=14, color=CREAM),
            showarrow=False,
        )],
    )
    return fig


# ── Portfolio P&L Bar ─────────────────────────────────────────
def portfolio_pnl_chart(df: pd.DataFrame) -> go.Figure:
    colors = [BULL if v >= 0 else BEAR for v in df["P&L"]]
    fig = go.Figure(go.Bar(
        x=df["Ticker"], y=df["P&L"],
        marker_color=colors,
        text=df["P&L %"].apply(lambda x: f"{x:+.1f}%"),
        textposition="outside",
        textfont=dict(color=TEXT),
    ))
    fig.add_hline(y=0, line_color=GRID)
    fig.update_layout(**LAYOUT, title="Portfolio P&L by Stock", height=300,
                      yaxis_title="P&L (₹ / $)")
    return fig


# ── Comparison Chart ──────────────────────────────────────────
def comparison_chart(dfs: dict) -> go.Figure:
    """Normalised price comparison (base 100) for multiple tickers."""
    fig = go.Figure()
    palette = [BULL, BEAR, BLUE, "#ffd166", "#f4a261", "#c77dff"]
    for i, (ticker, df) in enumerate(dfs.items()):
        if df.empty:
            continue
        norm = (df["Close"] / df["Close"].iloc[0]) * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=norm, name=ticker,
            line=dict(color=palette[i % len(palette)], width=1.8),
        ))
    fig.add_hline(y=100, line_dash="dash", line_color=GRID)
    fig.update_layout(
        **LAYOUT,
        title="Relative Performance (Base = 100)",
        height=380,
        yaxis_title="Normalised Price",
    )
    return fig
