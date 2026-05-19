"""
app.py  —  Stock Market Analyzer
Real market data · Technical Analysis · AI Health Score · Portfolio Tracker
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data       import (fetch_history, fetch_info, fetch_news,
                         get_current_price, INDIAN_STOCKS, US_STOCKS,
                         INDICES, PERIODS)
from indicators import compute_all
from sentiment  import analyze_news
from scoring    import technical_score, fundamental_score, health_score
from charts     import (candlestick_chart, macd_chart, health_gauge,
                         score_breakdown_bar, sentiment_donut,
                         portfolio_pnl_chart, comparison_chart)
from portfolio  import (init_portfolio, add_holding, remove_holding,
                         get_portfolio_df, portfolio_summary)

# ─────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  Global CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html,body,[class*="css"]{ font-family:'IBM Plex Sans',sans-serif; }
.stApp { background:#0f1117; color:#c8cdd8; }
.stSidebar { background:#090c11; border-right:1px solid #1e2130; }
.stSidebar [data-testid="stMarkdownContainer"] { color:#8892a4; }

.metric-card {
  background:#141922; border:1px solid #1e2130; border-radius:10px;
  padding:16px 20px; text-align:center;
}
.metric-big {
  font-family:'IBM Plex Mono',monospace; font-size:2rem;
  font-weight:600; line-height:1;
}
.metric-label {
  font-size:11px; color:#8892a4; letter-spacing:1.5px;
  text-transform:uppercase; margin-top:4px;
}
.metric-delta-pos { color:#52b788; font-size:13px; font-weight:600; }
.metric-delta-neg { color:#e07070; font-size:13px; font-weight:600; }

.signal-row {
  display:flex; align-items:center; gap:10px;
  padding:8px 12px; border-radius:8px; margin:4px 0;
  background:#141922; border:1px solid #1e2130;
}
.signal-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.signal-name { font-size:12px; color:#8892a4; min-width:100px; font-family:'IBM Plex Mono',monospace; }
.signal-desc { font-size:13px; color:#c8cdd8; }

.news-row {
  padding:10px 14px; border-radius:8px; background:#141922;
  border:1px solid #1e2130; margin:4px 0;
}
.news-date { font-size:11px; color:#8892a4; font-family:'IBM Plex Mono',monospace; }
.section-header {
  font-size:11px; font-weight:700; letter-spacing:2px;
  text-transform:uppercase; color:#6D8196; margin:24px 0 10px;
}
.health-verdict {
  font-family:'IBM Plex Mono',monospace; font-size:1.6rem;
  font-weight:600;
}
div[data-testid="stMetric"] {
  background:#141922; border:1px solid #1e2130;
  border-radius:10px; padding:14px;
}
div[data-testid="stMetric"] label { color:#8892a4!important; font-size:12px; }

.stTabs [data-baseweb="tab-list"] {
  background:#090c11; border-radius:8px; gap:4px; padding:4px;
  border:1px solid #1e2130;
}
.stTabs [data-baseweb="tab"] {
  font-family:'IBM Plex Mono',monospace; font-size:12px;
  color:#8892a4; border-radius:6px; padding:6px 14px;
}
.stTabs [aria-selected="true"] {
  background:#6D8196!important; color:#0f1117!important; font-weight:600;
}
.stButton>button {
  background:#6D8196; color:#0f1117; border:none;
  border-radius:8px; font-weight:600; font-size:13px;
}
.stButton>button:hover { background:#8fa3b3; }
.stSelectbox>div,.stTextInput>div>div {
  background:#141922!important; border-color:#1e2130!important;
  color:#c8cdd8!important; border-radius:8px!important;
}
.stNumberInput>div>div { background:#141922!important; border-color:#1e2130!important; }
hr { border-color:#1e2130; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Stock Analyzer")
    st.markdown("*Real market data · AI Health Score*")
    st.markdown("---")

    market = st.radio("Market", ["Indian (NSE)", "US (NYSE/Nasdaq)"], horizontal=True)
    stock_dict = INDIAN_STOCKS if market == "Indian (NSE)" else US_STOCKS
    currency   = "₹" if market == "Indian (NSE)" else "$"

    mode = st.selectbox("Select stock", ["Choose from list", "Enter custom ticker"])
    if mode == "Choose from list":
        stock_name   = st.selectbox("Stock", list(stock_dict.keys()))
        ticker_input = stock_dict[stock_name]
    else:
        ticker_input = st.text_input("Ticker (e.g. INFY.NS or AAPL)", value="TCS.NS").upper().strip()

    period_label = st.selectbox("Time Period", list(PERIODS.keys()), index=3)
    period, interval = PERIODS[period_label]

    st.markdown("---")
    st.markdown("**Chart Options**")
    show_ma  = st.checkbox("Moving Averages", value=True)
    show_bb  = st.checkbox("Bollinger Bands", value=True)
    show_vol = st.checkbox("Volume", value=True)

    st.markdown("---")
    analyze_btn = st.button("Analyze Stock", use_container_width=True)

    st.markdown("---")
    st.markdown("**Compare Stocks**")
    compare_raw = st.text_input("Tickers (comma-separated)", placeholder="TCS.NS,INFY.NS")

# ─────────────────────────────────────────────────────────────
#  Page header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:10px 0 4px'>
  <div style='font-family:IBM Plex Mono,monospace;font-size:2rem;font-weight:600;color:#c8cdd8'>
    Stock Market Analyzer
  </div>
  <div style='color:#8892a4;font-size:13px;margin-top:4px'>
    Technical analysis · AI health scoring · Sentiment · Portfolio tracking
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
#  Run analysis when button clicked or on first load
# ─────────────────────────────────────────────────────────────
if analyze_btn or "analyzed_ticker" not in st.session_state:
    st.session_state["analyzed_ticker"] = ticker_input
    st.session_state["analyzed_period"] = period_label

ticker = st.session_state.get("analyzed_ticker", ticker_input)

# ─────────────────────────────────────────────────────────────
#  Main tabs
# ─────────────────────────────────────────────────────────────
tab_chart, tab_health, tab_sentiment, tab_compare, tab_portfolio = st.tabs([
    "Price Chart",
    "Health Score",
    "Sentiment",
    "Compare",
    "Portfolio",
])

# ════════════════════════════════════════════════════════════
#  TAB 1 — PRICE CHART
# ════════════════════════════════════════════════════════════
with tab_chart:
    with st.spinner(f"Fetching data for {ticker}..."):
        df_raw   = fetch_history(ticker, period, interval)
        info     = fetch_info(ticker)
        live     = get_current_price(ticker)

    if df_raw.empty:
        st.error(f"No data found for **{ticker}**. Check the ticker symbol and try again.")
        st.stop()

    df = compute_all(df_raw)
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    # ── Live price bar ────────────────────────────────────
    price   = live.get("price") or latest["Close"]
    prev_cl = live.get("prev") or prev["Close"]
    chg     = price - prev_cl
    chg_pct = (chg / prev_cl * 100) if prev_cl else 0
    delta_class = "metric-delta-pos" if chg >= 0 else "metric-delta-neg"
    delta_sym   = "+" if chg >= 0 else ""

    c1, c2, c3, c4, c5 = st.columns(5)
    long_name = info.get("longName") or info.get("shortName") or ticker

    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>{long_name}</div>
            <div class='metric-big' style='color:#c8cdd8'>{currency}{price:,.2f}</div>
            <div class='{delta_class}'>{delta_sym}{chg:.2f} ({delta_sym}{chg_pct:.2f}%)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.metric("RSI (14)", f"{latest.get('RSI', 0):.1f}",
                  help="Above 70 = overbought · Below 30 = oversold")
    with c3:
        macd_val = latest.get("MACD", 0)
        sig_val  = latest.get("MACD_Signal", 0)
        st.metric("MACD", f"{macd_val:.3f}",
                  delta=f"Signal: {sig_val:.3f}")
    with c4:
        wk52_l = info.get("fiftyTwoWeekLow", "—")
        wk52_h = info.get("fiftyTwoWeekHigh", "—")
        st.metric("52-Week Range",
                  f"{currency}{wk52_l} – {currency}{wk52_h}" if wk52_l != "—" else "—")
    with c5:
        mc = info.get("marketCap")
        if mc:
            if mc >= 1e12:
                mc_str = f"{currency}{mc/1e12:.2f}T"
            elif mc >= 1e9:
                mc_str = f"{currency}{mc/1e9:.2f}B"
            else:
                mc_str = f"{currency}{mc/1e6:.0f}M"
        else:
            mc_str = "—"
        st.metric("Market Cap", mc_str)

    # ── Main chart ────────────────────────────────────────
    st.plotly_chart(
        candlestick_chart(df, ticker, show_ma, show_bb, show_vol),
        use_container_width=True,
    )

    # ── MACD ─────────────────────────────────────────────
    with st.expander("MACD Chart"):
        st.plotly_chart(macd_chart(df), use_container_width=True)

    # ── Support & Resistance ──────────────────────────────
    with st.expander("Support & Resistance Levels"):
        from indicators import support_resistance
        sr = support_resistance(df)
        col_r, col_s = st.columns(2)
        with col_r:
            st.markdown("<div class='section-header'>Resistance Levels</div>", unsafe_allow_html=True)
            for lvl in sorted(sr["resistance"], reverse=True):
                color = "#e07070" if lvl > price else "#8892a4"
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;color:{color};padding:4px 0'>"
                    f"{currency}{lvl:,.2f}</div>",
                    unsafe_allow_html=True,
                )
        with col_s:
            st.markdown("<div class='section-header'>Support Levels</div>", unsafe_allow_html=True)
            for lvl in sorted(sr["support"], reverse=True):
                color = "#52b788" if lvl < price else "#8892a4"
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;color:{color};padding:4px 0'>"
                    f"{currency}{lvl:,.2f}</div>",
                    unsafe_allow_html=True,
                )

    # ── Fundamentals table ────────────────────────────────
    with st.expander("Fundamental Data"):
        fund_map = {
            "P/E (Trailing)":   info.get("trailingPE"),
            "P/E (Forward)":    info.get("forwardPE"),
            "P/B Ratio":        info.get("priceToBook"),
            "EPS (TTM)":        info.get("trailingEps"),
            "Revenue Growth":   f"{info.get('revenueGrowth',0)*100:.1f}%" if info.get("revenueGrowth") else None,
            "Profit Margin":    f"{info.get('profitMargins',0)*100:.1f}%" if info.get("profitMargins") else None,
            "Debt/Equity":      info.get("debtToEquity"),
            "ROE":              f"{info.get('returnOnEquity',0)*100:.1f}%" if info.get("returnOnEquity") else None,
            "Dividend Yield":   f"{info.get('dividendYield',0)*100:.2f}%" if info.get("dividendYield") else None,
            "Beta":             info.get("beta"),
            "Sector":           info.get("sector"),
            "Industry":         info.get("industry"),
        }
        rows = [(k, str(round(v, 2)) if isinstance(v, float) else str(v))
                for k, v in fund_map.items() if v is not None]
        if rows:
            fd = pd.DataFrame(rows, columns=["Metric", "Value"])
            st.dataframe(fd, use_container_width=True, hide_index=True)
        else:
            st.info("Fundamental data not available for this ticker.")


# ════════════════════════════════════════════════════════════
#  TAB 2 — HEALTH SCORE
# ════════════════════════════════════════════════════════════
with tab_health:
    if df_raw.empty:
        st.warning("Run an analysis first (select a stock and click Analyze).")
    else:
        df_h   = compute_all(df_raw)
        info_h = fetch_info(ticker)
        news_h = fetch_news(ticker)
        sent_h = analyze_news(news_h)

        t_score = technical_score(df_h)
        f_score = fundamental_score(info_h)
        h_score = health_score(t_score, f_score, sent_h)

        # Composite gauge + breakdown
        col_gauge, col_break = st.columns([1, 1.4])
        with col_gauge:
            # st.markdown("<div class='section-header'>Overall Health Score</div>", unsafe_allow_html=True)
            # st.plotly_chart(health_gauge(h_score["score"], h_score["color"]), use_container_width=True)
            # st.markdown(
            #     f"<div class='health-verdict' style='color:{h_score[\"color\"]};text-align:center'>"
            #     f"{h_score['verdict']}</div>"
            #     f"<div style='color:#8892a4;font-size:13px;text-align:center;margin-top:6px'>"
            #     f"{h_score['desc']}</div>",
            #     unsafe_allow_html=True,
            # )
            hs_color   = h_score["color"]
            hs_verdict = h_score["verdict"]
            hs_desc    = h_score["desc"]
            st.markdown(
                f"<div class='health-verdict' style='color:{hs_color};text-align:center'>"
                f"{hs_verdict}</div>"
                f"<div style='color:#8892a4;font-size:13px;text-align:center;margin-top:6px'>"
                f"{hs_desc}</div>",
                unsafe_allow_html=True,
            )
        with col_break:
            st.markdown("<div class='section-header'>Score Components</div>", unsafe_allow_html=True)
            st.plotly_chart(score_breakdown_bar(h_score["components"]), use_container_width=True)
            for comp, vals in h_score["components"].items():
                bar_w = vals["score"]
                color = "#52b788" if bar_w >= 60 else "#e07070" if bar_w < 40 else "#6D8196"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;font-size:13px;margin:3px 0'>"
                    f"<span style='color:#8892a4'>{comp} ({vals['weight']})</span>"
                    f"<span style='color:{color};font-family:IBM Plex Mono,monospace'>{vals['score']}/100</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Technical + Fundamental signals side by side
        col_tech, col_fund = st.columns(2)
        signal_colors = {
            "bullish": "#52b788",
            "bearish": "#e07070",
            "caution": "#ffd166",
            "neutral": "#6D8196",
        }

        with col_tech:
            st.markdown("<div class='section-header'>Technical Signals</div>", unsafe_allow_html=True)
            for (name, desc, tone) in t_score.get("signals", []):
                color = signal_colors.get(tone, "#6D8196")
                st.markdown(
                    f"<div class='signal-row'>"
                    f"<div class='signal-dot' style='background:{color}'></div>"
                    f"<div class='signal-name'>{name}</div>"
                    f"<div class='signal-desc'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with col_fund:
            st.markdown("<div class='section-header'>Fundamental Signals</div>", unsafe_allow_html=True)
            for (name, desc, tone) in f_score.get("signals", []):
                color = signal_colors.get(tone, "#6D8196")
                st.markdown(
                    f"<div class='signal-row'>"
                    f"<div class='signal-dot' style='background:{color}'></div>"
                    f"<div class='signal-name'>{name}</div>"
                    f"<div class='signal-desc'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("""
        <div style='background:#141922;border:1px solid #1e2130;border-radius:8px;padding:14px;margin-top:20px;
             font-size:12px;color:#8892a4'>
        This health score is for informational purposes only and does not constitute investment advice.
        Always consult a qualified financial advisor before making investment decisions.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 3 — SENTIMENT
# ════════════════════════════════════════════════════════════
with tab_sentiment:
    if df_raw.empty:
        st.warning("Run an analysis first.")
    else:
        news_s = fetch_news(ticker)
        sent_s = analyze_news(news_s)

        col_d, col_score = st.columns([1, 1.5])
        with col_d:
            st.markdown("<div class='section-header'>News Sentiment Breakdown</div>", unsafe_allow_html=True)
            st.plotly_chart(sentiment_donut(
                sent_s.get("pos_count", 0),
                sent_s.get("neg_count", 0),
                sent_s.get("neu_count", 0),
            ), use_container_width=True)

        with col_score:
            st.markdown("<div class='section-header'>Overall Sentiment</div>", unsafe_allow_html=True)
            sc     = sent_s["score"]
            label  = sent_s["label"]
            color  = sent_s["color"]
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:3rem;font-weight:600;color:{color}'>{sc}</div>"
                f"<div style='font-size:1.2rem;color:{color};margin-bottom:12px'>{label}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"Analysed **{len(sent_s.get('headlines', []))}** recent headlines")
            st.markdown(
                f"Positive: **{sent_s.get('pos_count',0)}** &nbsp;·&nbsp; "
                f"Negative: **{sent_s.get('neg_count',0)}** &nbsp;·&nbsp; "
                f"Neutral: **{sent_s.get('neu_count',0)}**"
            )

        st.markdown("<div class='section-header'>Recent Headlines</div>", unsafe_allow_html=True)
        if not sent_s.get("headlines"):
            st.info("No recent news found for this ticker.")
        for hl in sent_s.get("headlines", []):
            c = hl["compound"]
            color = "#52b788" if c > 0.05 else "#e07070" if c < -0.05 else "#6D8196"
            bar   = int((c + 1) / 2 * 100)
            st.markdown(
                f"<div class='news-row'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div style='color:#c8cdd8;font-size:14px;flex:1'>{hl['headline']}</div>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:12px;color:{color};margin-left:12px;white-space:nowrap'>"
                f"{c:+.3f} {hl['label']}</div>"
                f"</div>"
                f"<div class='news-date'>{hl['date']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════
#  TAB 4 — COMPARE
# ════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("<div class='section-header'>Compare Multiple Stocks</div>", unsafe_allow_html=True)
    st.markdown("Enter comma-separated tickers in the sidebar or below:")

    raw_input = st.text_input("Tickers to compare", value=compare_raw or ticker,
                               placeholder="RELIANCE.NS, TCS.NS, INFY.NS")

    compare_period = st.selectbox("Period", list(PERIODS.keys()), index=3, key="cmp_period")
    c_period, c_interval = PERIODS[compare_period]

    if st.button("Compare", key="cmp_btn") and raw_input:
        tickers_cmp = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
        if len(tickers_cmp) < 2:
            st.warning("Please enter at least 2 tickers.")
        elif len(tickers_cmp) > 6:
            st.warning("Maximum 6 tickers for comparison.")
        else:
            dfs_cmp = {}
            with st.spinner("Fetching data..."):
                for t_cmp in tickers_cmp:
                    dfs_cmp[t_cmp] = fetch_history(t_cmp, c_period, c_interval)

            st.plotly_chart(comparison_chart(dfs_cmp), use_container_width=True)

            # Quick stats table
            rows_cmp = []
            for t_cmp, d in dfs_cmp.items():
                if d.empty: continue
                ret = ((d["Close"].iloc[-1] / d["Close"].iloc[0]) - 1) * 100
                vol = d["Close"].pct_change().std() * (252**0.5) * 100
                rows_cmp.append({
                    "Ticker": t_cmp,
                    "Return (%)": round(ret, 2),
                    "Annualised Vol (%)": round(vol, 2),
                    "Current": round(d["Close"].iloc[-1], 2),
                    "Period High": round(d["High"].max(), 2),
                    "Period Low":  round(d["Low"].min(), 2),
                })
            if rows_cmp:
                cmp_df = pd.DataFrame(rows_cmp)
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
#  TAB 5 — PORTFOLIO
# ════════════════════════════════════════════════════════════
with tab_portfolio:
    init_portfolio()

    st.markdown("<div class='section-header'>Add a Holding</div>", unsafe_allow_html=True)
    pa, pb, pc, pd_, pe, pf = st.columns([1.5, 2, 1, 1.2, 1.2, 1])
    with pa:
        p_ticker = st.text_input("Ticker", placeholder="RELIANCE.NS", key="p_tick").upper().strip()
    with pb:
        p_name   = st.text_input("Name", placeholder="Reliance Industries", key="p_name")
    with pc:
        p_qty    = st.number_input("Quantity", min_value=0.0, step=1.0, key="p_qty")
    with pd_:
        p_price  = st.number_input("Buy Price", min_value=0.0, step=0.01, key="p_price")
    with pe:
        p_date   = st.text_input("Buy Date", placeholder="DD-MM-YYYY", key="p_date")
    with pf:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add", key="p_add"):
            if p_ticker and p_qty > 0 and p_price > 0:
                add_holding(p_ticker, p_name or p_ticker, p_qty, p_price, p_date)
                st.success(f"Added {p_ticker}")
                st.rerun()
            else:
                st.error("Fill in ticker, qty, and buy price.")

    # Portfolio table
    port_df = get_portfolio_df()
    if port_df.empty:
        st.markdown("<div style='color:#8892a4;padding:30px 0;text-align:center;font-size:14px'>"
                    "No holdings yet. Add your first stock above.</div>", unsafe_allow_html=True)
    else:
        summary = portfolio_summary(port_df)
        s1, s2, s3, s4 = st.columns(4)
        pnl_color = "#52b788" if summary["pnl"] >= 0 else "#e07070"
        s1.metric("Total Invested",    f"{currency}{summary['invested']:,.2f}")
        s2.metric("Current Value",     f"{currency}{summary['current']:,.2f}")
        s3.metric("Total P&L",
                  f"{currency}{abs(summary['pnl']):,.2f}",
                  delta=f"{summary['pnl_pct']:+.2f}%")
        s4.metric("Winners / Losers",  f"{summary['winners']} / {summary['losers']}")

        st.markdown("<div class='section-header'>Holdings</div>", unsafe_allow_html=True)

        def color_pnl(val):
            color = "#52b788" if val >= 0 else "#e07070"
            return f"color: {color}"

        display_df = port_df.drop(columns=["Invested"])
        st.dataframe(
            display_df.style.applymap(color_pnl, subset=["P&L", "P&L %"]),
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(portfolio_pnl_chart(port_df), use_container_width=True)

        # Remove holding
        with st.expander("Remove a holding"):
            for i, row in port_df.iterrows():
                c_l, c_r = st.columns([4, 1])
                c_l.markdown(f"**{row['Ticker']}** — Qty: {row['Qty']} @ {currency}{row['Buy Price']}")
                if c_r.button("Remove", key=f"del_{i}"):
                    remove_holding(i)
                    st.rerun()
