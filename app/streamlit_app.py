"""
streamlit_app.py — NIFTY-50 Investment Intelligence Platform
Main Streamlit application combining all modules.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json, warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #0f9d58);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 16px; border-left: 4px solid #1a73e8;
        margin-bottom: 8px;
    }
    .signal-buy   { color: #0f9d58; font-weight: 700; font-size: 1.1rem; }
    .signal-sell  { color: #ea4335; font-weight: 700; font-size: 1.1rem; }
    .risk-low     { color: #0f9d58; font-weight: 600; }
    .risk-medium  { color: #f9ab00; font-weight: 600; }
    .risk-high    { color: #e37400; font-weight: 600; }
    .risk-veryhigh{ color: #ea4335; font-weight: 600; }
    [data-testid="stSidebar"] { background-color: #0d1117; }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ── Imports from project ──────────────────────────────────────────────────────
from src.data_loader import (load_stock, get_symbol_list, get_sector_map,
                              load_metadata, get_close_price_matrix)
from src.indicators import add_all_indicators
from src.predictor import EnsemblePredictor, prepare_stock_df, get_latest_signal
from src.portfolio import (get_price_data, load_portfolios, simulate_portfolio_performance,
                            compute_efficient_frontier, get_sector_exposure,
                            get_discrete_allocation, build_all_portfolios, save_portfolios)
from src.risk import (compute_stock_metrics, compute_portfolio_metrics,
                      stress_test_portfolio, classify_risk, compute_rolling_risk)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load_stock(symbol):
    return load_stock(symbol)

@st.cache_data(show_spinner=False)
def cached_metadata():
    return load_metadata()

@st.cache_data(show_spinner=False)
def cached_sector_map():
    return get_sector_map()

@st.cache_data(show_spinner=False)
def cached_price_data():
    return get_price_data(start_date='2015-01-01')

@st.cache_data(show_spinner=False)
def cached_portfolios():
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'portfolios.json')
    if os.path.exists(path):
        return load_portfolios(path)
    prices = cached_price_data()
    portfolios = build_all_portfolios(prices)
    save_portfolios(portfolios, path)
    return portfolios

@st.cache_data(show_spinner=False)
def cached_risk_scorecard():
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'risk_scorecard.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data(show_spinner=False)
def cached_signals():
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'latest_signals.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data(show_spinner=False)
def get_stock_with_indicators(symbol):
    df = load_stock(symbol)
    df = add_all_indicators(df)
    df['Price_vs_SMA20']  = (df['Close'] - df['SMA_20'])  / df['SMA_20']  * 100
    df['Price_vs_SMA50']  = (df['Close'] - df['SMA_50'])  / df['SMA_50']  * 100
    df['Price_vs_SMA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200'] * 100
    df['Bull_Trend'] = (df['SMA_50'] > df['SMA_200']).astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month']     = df['Date'].dt.month
    df['Quarter']   = df['Date'].dt.quarter
    return df

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## 📈 NIFTY-50 Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🔍 Stock Analyser",
    "🤖 AI Predictor",
    "💼 Portfolio Builder",
    "⚠️ Risk Assessment",
    "📊 Market Overview",
    "🧠 Explainable AI",
    "🚨 Anomaly Detection",
])
st.sidebar.markdown("---")
st.sidebar.caption("Data: NSE NIFTY-50 | Jan 2000 – Apr 2021")
st.sidebar.caption("Models: XGBoost + Random Forest Ensemble")

meta     = cached_metadata()
sector_map = cached_sector_map()
symbols  = get_symbol_list()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-header">NIFTY-50 Investment Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown("**AI-powered decision support for Indian equity markets — NSE NIFTY-50 universe**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📁 Stocks Covered",   "50 Companies")
    col2.metric("📅 Data Range",        "2000 – 2021")
    col3.metric("🏭 Sectors",           f"{meta['Industry'].nunique()} Sectors")
    col4.metric("📊 Trading Days",      "~5,300 / stock")

    st.markdown("---")
    st.subheader("Platform Capabilities")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🤖 AI Stock Predictor")
        st.markdown("""
- XGBoost + Random Forest ensemble  
- 5-day price direction prediction  
- Expected return forecasting  
- Buy / Sell signal generation  
- Walk-forward cross-validation  
        """)
    with c2:
        st.markdown("### 💼 Portfolio Builder")
        st.markdown("""
- Conservative (Min Volatility)  
- Balanced (Max Sharpe Ratio)  
- Aggressive (Max Return)  
- Risk Parity allocation  
- Efficient frontier visualisation  
- Discrete share allocation  
        """)
    with c3:
        st.markdown("### ⚠️ Risk Assessment")
        st.markdown("""
- Sharpe & Sortino ratios  
- VaR 95% / 99% (Historical + Parametric)  
- CVaR / Expected Shortfall  
- Maximum Drawdown analysis  
- Stress testing (2008 GFC, 2020 COVID)  
- Rolling risk metrics  
        """)

    st.markdown("---")
    st.subheader("Sector Composition")
    sector_counts = meta['Industry'].value_counts().reset_index()
    sector_counts.columns = ['Sector', 'Count']
    fig = px.bar(sector_counts, x='Count', y='Sector', orientation='h',
                 color='Count', color_continuous_scale='Blues',
                 title="NIFTY-50 Companies by Sector")
    fig.update_layout(height=400, showlegend=False, yaxis_title="",
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: STOCK ANALYSER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Stock Analyser":
    st.title("🔍 Stock Analyser")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.selectbox("Select Stock", symbols,
                               format_func=lambda s: f"{s}  —  {sector_map.get(s,'')}")
    with col2:
        start_yr = st.selectbox("From Year", list(range(2000, 2022)), index=15)
    with col3:
        indicators_shown = st.multiselect("Indicators", ["SMA 50", "SMA 200", "EMA 12", "Bollinger Bands"],
                                           default=["SMA 50", "SMA 200", "Bollinger Bands"])

    with st.spinner(f"Loading {symbol}..."):
        df = get_stock_with_indicators(symbol)

    df_plot = df[df['Date'].dt.year >= start_yr].copy()

    # ── Price chart ───────────────────────────────────────────────────────────
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20],
                        vertical_spacing=0.04)

    fig.add_trace(go.Candlestick(
        x=df_plot['Date'], open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'], name='Price',
        increasing_line_color='#26a641', decreasing_line_color='#ea4335',
    ), row=1, col=1)

    if "SMA 50" in indicators_shown and 'SMA_50' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['SMA_50'],
                                  name='SMA 50', line=dict(color='orange', width=1.2)), row=1, col=1)
    if "SMA 200" in indicators_shown and 'SMA_200' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['SMA_200'],
                                  name='SMA 200', line=dict(color='purple', width=1.2)), row=1, col=1)
    if "EMA 12" in indicators_shown and 'EMA_12' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['EMA_12'],
                                  name='EMA 12', line=dict(color='cyan', width=1)), row=1, col=1)
    if "Bollinger Bands" in indicators_shown and 'BB_Upper' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['BB_Upper'],
                                  name='BB Upper', line=dict(color='gray', dash='dot', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['BB_Lower'],
                                  name='BB Lower', line=dict(color='gray', dash='dot', width=1),
                                  fill='tonexty', fillcolor='rgba(128,128,128,0.05)'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['RSI'],
                              name='RSI', line=dict(color='#1a73e8', width=1.2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red",   row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # Volume
    colors_vol = ['#26a641' if c >= o else '#ea4335'
                  for c, o in zip(df_plot['Close'], df_plot['Open'])]
    fig.add_trace(go.Bar(x=df_plot['Date'], y=df_plot['Volume'],
                          name='Volume', marker_color=colors_vol, opacity=0.6), row=3, col=1)

    fig.update_layout(height=650, xaxis_rangeslider_visible=False,
                      title=f"{symbol} — {sector_map.get(symbol, '')}",
                      legend=dict(orientation='h', y=1.02),
                      margin=dict(l=0, r=0, t=60, b=0))
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="RSI",       row=2, col=1)
    fig.update_yaxes(title_text="Volume",    row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Key stats ─────────────────────────────────────────────────────────────
    st.subheader("Key Statistics")
    latest = df_plot.iloc[-1]
    risk_m = compute_stock_metrics(df.set_index('Date')['Close'], symbol)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Price",  f"₹{latest['Close']:,.2f}")
    c2.metric("52W High",       f"₹{df_plot['High'].tail(252).max():,.2f}")
    c3.metric("52W Low",        f"₹{df_plot['Low'].tail(252).min():,.2f}")
    c4.metric("CAGR",           f"{risk_m['cagr_pct']:.1f}%")
    c5.metric("Annual Vol",     f"{risk_m['annual_volatility_pct']:.1f}%")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sharpe Ratio",   f"{risk_m['sharpe_ratio']:.3f}")
    c2.metric("Sortino Ratio",  f"{risk_m['sortino_ratio']:.3f}")
    c3.metric("Max Drawdown",   f"{risk_m['max_drawdown_pct']:.1f}%")
    c4.metric("VaR 95% (daily)",f"{risk_m['var_95_daily_pct']:.2f}%")
    rl = risk_m.get('risk_level', classify_risk(risk_m))
    c5.metric("Risk Level",     rl)

    # MACD chart
    with st.expander("MACD Detail"):
        fig2 = make_subplots(rows=1, cols=1)
        fig2.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['MACD'],
                                   name='MACD', line=dict(color='blue', width=1.2)))
        fig2.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['MACD_Signal'],
                                   name='Signal', line=dict(color='red', width=1.2)))
        colors_hist = ['#26a641' if v >= 0 else '#ea4335' for v in df_plot['MACD_Hist']]
        fig2.add_trace(go.Bar(x=df_plot['Date'], y=df_plot['MACD_Hist'],
                               name='Histogram', marker_color=colors_hist, opacity=0.6))
        fig2.update_layout(height=300, title="MACD", margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: AI PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Predictor":
    st.title("🤖 AI Stock Predictor")
    st.markdown("Ensemble model (XGBoost + Random Forest) predicting 5-day price direction.")

    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.selectbox("Select Stock", symbols,
                               format_func=lambda s: f"{s}  —  {sector_map.get(s,'')}")
    with col2:
        horizon = st.selectbox("Prediction Horizon", [5, 10, 20], index=0,
                                format_func=lambda x: f"{x} trading days")

    if st.button("🚀 Run Prediction", type="primary"):
        with st.spinner(f"Training / loading model for {symbol}..."):
            df = get_stock_with_indicators(symbol)
            model_path = os.path.join('models', f'{symbol}_direction_clf.pkl')
            predictor = EnsemblePredictor(horizon=horizon)
            if os.path.exists(model_path):
                try:
                    predictor.load(symbol)
                except:
                    predictor.train(df)
                    predictor.save(symbol)
            else:
                predictor.train(df)
                predictor.save(symbol)
            preds = predictor.predict(df)

        latest = preds.iloc[-1]
        signal = latest['Signal']
        prob   = float(latest['Up_Probability'])
        exp_r  = float(latest['Expected_Return_Pct'])

        # ── Signal card ───────────────────────────────────────────────────────
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price",     f"₹{latest['Close']:,.2f}")
        col2.metric("Signal",            signal)
        col3.metric("Up Probability",    f"{prob*100:.1f}%")
        col4.metric(f"Expected {horizon}d Return", f"{exp_r:.2f}%")

        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            title={'text': "Bullish Probability (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1a73e8"},
                'steps': [
                    {'range': [0,  40], 'color': '#fce8e6'},
                    {'range': [40, 60], 'color': '#fef9c3'},
                    {'range': [60, 100],'color': '#e6f4ea'},
                ],
                'threshold': {'line': {'color': "red", 'width': 2},
                               'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Prediction history chart ──────────────────────────────────────────
        recent_preds = preds.tail(252).copy()
        df_recent = df[df['Date'].isin(recent_preds['Date'])].copy()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=df_recent['Date'], y=df_recent['Close'],
                                  name='Close', line=dict(color='#2c3e50', width=1.5)), row=1, col=1)

        buys  = recent_preds[recent_preds['Signal'].isin(['BUY','STRONG BUY'])]
        sells = recent_preds[recent_preds['Signal'].isin(['SELL','STRONG SELL'])]
        close_map = df_recent.set_index('Date')['Close']

        if len(buys):
            fig.add_trace(go.Scatter(x=buys['Date'],
                                      y=buys['Date'].map(close_map),
                                      mode='markers', name='BUY',
                                      marker=dict(symbol='triangle-up', size=8, color='#26a641')), row=1, col=1)
        if len(sells):
            fig.add_trace(go.Scatter(x=sells['Date'],
                                      y=sells['Date'].map(close_map),
                                      mode='markers', name='SELL',
                                      marker=dict(symbol='triangle-down', size=8, color='#ea4335')), row=1, col=1)

        fig.add_trace(go.Scatter(x=recent_preds['Date'], y=recent_preds['Up_Probability'],
                                  name='P(Up)', line=dict(color='#9b59b6', width=1.2),
                                  fill='tozeroy', fillcolor='rgba(155,89,182,0.08)'), row=2, col=1)
        fig.add_hline(y=0.5, line_dash='dot', line_color='gray', row=2, col=1)

        fig.update_layout(height=500, title=f"{symbol} — AI Signals (Last 252 days)",
                          margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── All signals table ─────────────────────────────────────────────────────
    signals_df = cached_signals()
    if signals_df is not None:
        st.markdown("---")
        st.subheader("Latest Signals — All Stocks")
        col1, col2 = st.columns(2)
        with col1:
            sector_filter = st.multiselect("Filter by Sector",
                                            sorted(set(sector_map.values())), default=[])
        with col2:
            signal_filter = st.multiselect("Filter by Signal",
                                            ['STRONG BUY','BUY','SELL','STRONG SELL'], default=[])

        disp = signals_df.copy()
        if sector_filter:
            disp = disp[disp['sector'].isin(sector_filter)]
        if signal_filter:
            disp = disp[disp['signal'].isin(signal_filter)]

        disp = disp.sort_values('up_probability', ascending=False)
        disp['up_probability'] = (disp['up_probability'] * 100).round(1).astype(str) + '%'
        st.dataframe(disp[['symbol','sector','signal','up_probability',
                            'expected_return_pct','current_price']].reset_index(drop=True),
                     use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PORTFOLIO BUILDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💼 Portfolio Builder":
    st.title("💼 Portfolio Builder")

    with st.spinner("Loading portfolios..."):
        portfolios = cached_portfolios()
        prices     = cached_price_data()

    # ── Profile selector ──────────────────────────────────────────────────────
    profile_map = {
        "🛡️ Conservative — Capital Preservation": "conservative",
        "⚖️ Balanced — Growth & Stability":        "balanced",
        "🚀 Aggressive — Maximum Growth":           "aggressive",
        "🔢 Risk Parity":                           "risk_parity",
    }
    chosen_label = st.selectbox("Investor Profile", list(profile_map.keys()))
    chosen       = profile_map[chosen_label]
    p            = portfolios[chosen]

    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Strategy",         p['strategy'])
    col2.metric("Expected Return",  f"{p['expected_annual_return']*100:.1f}% p.a.")
    col3.metric("Annual Volatility",f"{p['annual_volatility']*100:.1f}%")
    col4.metric("Sharpe Ratio",     f"{p['sharpe_ratio']:.3f}")
    col5.metric("Holdings",         f"{len(p['weights'])} stocks")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Weights", "🌍 Sector Exposure",
                                       "📈 Performance", "💰 Allocation"])

    with tab1:
        weights = dict(sorted(p['weights'].items(), key=lambda x: x[1]))
        fig = px.bar(x=list(weights.values()), y=list(weights.keys()),
                     orientation='h', color=list(weights.values()),
                     color_continuous_scale='Blues',
                     labels={'x': 'Weight', 'y': 'Stock'},
                     title=f"{p['label']} — Portfolio Weights")
        fig.update_layout(height=max(350, len(weights)*22),
                          coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=50,b=0))
        fig.update_traces(text=[f"{v*100:.1f}%" for v in weights.values()],
                          textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        exposure = get_sector_exposure(p['weights'], sector_map)
        fig = px.pie(values=list(exposure.values()),
                     names=list(exposure.keys()),
                     title="Sector Exposure",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Historical Performance vs Benchmarks")
        fig = go.Figure()
        profile_configs = [
            ('conservative', '#2ecc71', 'Conservative'),
            ('balanced',     '#3498db', 'Balanced'),
            ('aggressive',   '#e74c3c', 'Aggressive'),
            ('equal_weight', '#95a5a6', 'Equal Weight (BM)'),
        ]
        for pname, color, label in profile_configs:
            pp = portfolios[pname]
            perf = simulate_portfolio_performance(pp['weights'], prices)
            lw = 2.5 if pname == chosen else 1
            dash = 'solid' if pname == chosen else 'dot'
            fig.add_trace(go.Scatter(x=perf.index, y=perf,
                                      name=f"{label} ({perf.iloc[-1]/100-1:.0%})",
                                      line=dict(color=color, width=lw, dash=dash)))
        fig.add_vrect(x0="2020-01-15", x1="2020-06-01",
                       fillcolor="red", opacity=0.05, annotation_text="COVID-19")
        fig.update_layout(height=420, yaxis_title="Portfolio Value (base 100)",
                          hovermode='x unified', margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Efficient Frontier")
        with st.spinner("Computing efficient frontier..."):
            ef_df = compute_efficient_frontier(prices, n_portfolios=2000)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ef_df['Volatility']*100, y=ef_df['Return']*100,
            mode='markers',
            marker=dict(color=ef_df['Sharpe'], colorscale='RdYlGn',
                        size=4, opacity=0.5, showscale=True,
                        colorbar=dict(title='Sharpe')),
            name='Random Portfolios'))
        colors_p = {'conservative':'#2ecc71','balanced':'#3498db',
                    'aggressive':'#e74c3c','risk_parity':'#9b59b6','equal_weight':'#95a5a6'}
        for pname, pp in portfolios.items():
            fig2.add_trace(go.Scatter(
                x=[pp['annual_volatility']*100], y=[pp['expected_annual_return']*100],
                mode='markers+text',
                marker=dict(size=14, color=colors_p.get(pname,'gray'),
                            line=dict(color='black', width=1.5)),
                text=[pp['label'].split()[0]], textposition='top center',
                name=pp['label']))
        fig2.update_layout(height=430, xaxis_title="Annual Volatility (%)",
                            yaxis_title="Expected Return (%)",
                            margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.subheader("Discrete Share Allocation")
        investment = st.slider("Investment Amount (₹)", 100_000, 10_000_000,
                                1_000_000, step=100_000,
                                format="₹%d")
        alloc = get_discrete_allocation(p['weights'], prices, investment)
        shares = alloc['shares']
        leftover = alloc['leftover_cash']

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Investment",  f"₹{investment:,.0f}")
        col2.metric("Stocks Bought",     f"{len(shares)}")
        col3.metric("Leftover Cash",     f"₹{leftover:,.2f}")

        rows = []
        for sym, n in sorted(shares.items(), key=lambda x: -x[1]*prices[x[0]].iloc[-1] if x[0] in prices.columns else 0):
            if sym in prices.columns:
                price = prices[sym].iloc[-1]
                val   = n * price
                rows.append({'Symbol': sym, 'Sector': sector_map.get(sym,''),
                              'Shares': n, 'Price (₹)': round(price,2),
                              'Value (₹)': round(val,2),
                              'Weight %': round(val/investment*100,2)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Assessment":
    st.title("⚠️ Risk Assessment")

    tab1, tab2, tab3 = st.tabs(["🔬 Single Stock", "💼 Portfolio Risk", "🔥 Stress Test"])

    with tab1:
        symbol = st.selectbox("Select Stock", symbols,
                               format_func=lambda s: f"{s}  —  {sector_map.get(s,'')}")
        with st.spinner("Computing risk metrics..."):
            df = cached_load_stock(symbol)
            m  = compute_stock_metrics(df.set_index('Date')['Close'], symbol)
            rl = classify_risk(m)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Return Metrics")
            st.metric("Total Return",  f"{m['total_return_pct']:.1f}%")
            st.metric("CAGR",          f"{m['cagr_pct']:.2f}% p.a.")
            st.metric("Win Rate",      f"{m['win_rate_pct']:.1f}% of days")
            st.metric("Gain/Loss Ratio", f"{m['gain_loss_ratio']:.2f}x")
        with col2:
            st.subheader("Risk Metrics")
            risk_color = {'LOW':'🟢','MEDIUM':'🟡','HIGH':'🟠','VERY HIGH':'🔴'}
            st.metric("Risk Level",    f"{risk_color.get(rl,'')} {rl}")
            st.metric("Annual Volatility", f"{m['annual_volatility_pct']:.2f}%")
            st.metric("Max Drawdown",  f"{m['max_drawdown_pct']:.2f}%")
            st.metric("Sharpe Ratio",  f"{m['sharpe_ratio']:.3f}")
            st.metric("Sortino Ratio", f"{m['sortino_ratio']:.3f}")

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VaR 95% (daily)",  f"{m['var_95_daily_pct']:.2f}%")
        col2.metric("VaR 99% (daily)",  f"{m['var_99_daily_pct']:.2f}%")
        col3.metric("CVaR 95% (daily)", f"{m['cvar_95_daily_pct']:.2f}%")
        col4.metric("CVaR 99% (daily)", f"{m['cvar_99_daily_pct']:.2f}%")

        # Return distribution
        returns = df['Close'].pct_change().dropna() * 100
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns, nbinsx=80, name='Daily Returns',
                                    marker_color='#3498db', opacity=0.7,
                                    histnorm='probability density'))
        fig.add_vline(x=m['var_95_daily_pct'], line_dash='dash', line_color='red',
                       annotation_text=f"VaR 95%: {m['var_95_daily_pct']:.2f}%")
        fig.add_vline(x=m['cvar_95_daily_pct'], line_dash='dash', line_color='darkred',
                       annotation_text=f"CVaR 95%: {m['cvar_95_daily_pct']:.2f}%")
        fig.update_layout(height=350, title=f"{symbol} — Daily Return Distribution",
                          xaxis_title="Daily Return (%)", margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Rolling risk
        with st.expander("Rolling Risk Metrics"):
            roll = compute_rolling_risk(df.set_index('Date')['Close'], window=126)
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            fig2.add_trace(go.Scatter(x=roll.index, y=roll['Rolling_Vol_pct'],
                                       name='Rolling Vol%', line=dict(color='orange')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=roll.index, y=roll['Rolling_Sharpe'],
                                       name='Rolling Sharpe', line=dict(color='blue')), row=2, col=1)
            fig2.add_hline(y=0, line_dash='dot', row=2, col=1)
            fig2.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        with st.spinner("Loading portfolio risk..."):
            portfolios = cached_portfolios()
            prices     = cached_price_data()

        profile_names = ['conservative', 'balanced', 'aggressive', 'risk_parity', 'equal_weight']
        port_data = []
        for name in profile_names:
            pp = portfolios[name]
            pm = compute_portfolio_metrics(pp['weights'], prices)
            port_data.append({
                'Profile':      pp['label'],
                'CAGR%':        pm['cagr_pct'],
                'Vol%':         pm['annual_volatility_pct'],
                'Sharpe':       pm['sharpe_ratio'],
                'Sortino':      pm['sortino_ratio'],
                'Max DD%':      pm['max_drawdown_pct'],
                'VaR95%':       pm['var_95_daily_pct'],
                'Eff. Stocks':  pm['effective_n_stocks'],
            })
        port_table = pd.DataFrame(port_data)
        st.dataframe(port_table.set_index('Profile').round(3), use_container_width=True)

        chosen_profile = st.selectbox("View Risk Contribution for",
                                       [portfolios[n]['label'] for n in profile_names[:3]])
        chosen_key = [k for k in profile_names[:3]
                      if portfolios[k]['label'] == chosen_profile][0]
        pm_chosen = compute_portfolio_metrics(portfolios[chosen_key]['weights'], prices)
        rc = pd.Series(pm_chosen['risk_contributions']).sort_values(ascending=False).head(15)
        fig = px.bar(x=rc.values, y=rc.index, orientation='h',
                     labels={'x':'Risk Contribution %','y':'Stock'},
                     title=f"Top 15 Risk Contributors — {chosen_profile}",
                     color=rc.values, color_continuous_scale='Reds')
        fig.update_layout(height=420, coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Portfolio Stress Testing")
        st.markdown("Performance during major market stress events.")
        with st.spinner("Running stress tests..."):
            portfolios = cached_portfolios()
            prices     = cached_price_data()
            stress_data = {}
            for name in ['conservative','balanced','aggressive']:
                stress_data[name] = stress_test_portfolio(portfolios[name]['weights'], prices)

        periods = ['2011_EuroDebt','2015_ChinaSlowdown','2020_COVID','2020_Recovery']
        period_labels = ['2011 Euro Debt','2015 China Slowdown','2020 COVID Crash','2020 Recovery']

        rows = []
        for period, label in zip(periods, period_labels):
            row = {'Event': label}
            for name in ['conservative','balanced','aggressive']:
                r = stress_data[name].get(period, {})
                row[portfolios[name]['label'].split()[0]] = (
                    f"{r['return_pct']:.1f}%" if r.get('return_pct') is not None else "N/A"
                )
            rows.append(row)
        st.table(pd.DataFrame(rows).set_index('Event'))

        # Stress test bar chart
        fig = go.Figure()
        colors3 = ['#2ecc71','#3498db','#e74c3c']
        for name, color in zip(['conservative','balanced','aggressive'], colors3):
            vals = []
            for period in periods:
                r = stress_data[name].get(period, {})
                vals.append(r.get('return_pct') or 0)
            fig.add_trace(go.Bar(name=portfolios[name]['label'].split()[0],
                                  x=period_labels, y=vals,
                                  marker_color=color, opacity=0.85))
        fig.add_hline(y=0, line_color='black', line_width=0.8)
        fig.update_layout(barmode='group', height=380,
                          title="Portfolio Returns During Stress Events",
                          yaxis_title="Period Return (%)",
                          margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Market Overview":
    st.title("📊 Market Overview")

    with st.spinner("Loading market data..."):
        prices = cached_price_data()
        risk_df = cached_risk_scorecard()

    # ── Heatmap of 1-year returns ─────────────────────────────────────────────
    st.subheader("1-Year Return Heatmap")
    returns_1y = {}
    for sym in prices.columns:
        s = prices[sym].dropna()
        if len(s) >= 252:
            r = (s.iloc[-1] / s.iloc[-252] - 1) * 100
            returns_1y[sym] = round(r, 2)

    ret_series = pd.Series(returns_1y).sort_values(ascending=False)
    ret_df = ret_series.reset_index()
    ret_df.columns = ['Symbol','Return_1Y']
    ret_df['Sector'] = ret_df['Symbol'].map(sector_map)
    ret_df['Color']  = ret_df['Return_1Y'].apply(
        lambda x: '#2ecc71' if x > 20 else '#27ae60' if x > 0 else '#e74c3c' if x < -20 else '#e67e22')

    fig = px.bar(ret_df, x='Symbol', y='Return_1Y', color='Return_1Y',
                 color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                 hover_data=['Sector'],
                 title="1-Year Price Return by Stock (Latest 252 trading days)")
    fig.add_hline(y=0, line_color='black', line_width=0.8)
    fig.update_layout(height=400, xaxis_tickangle=-45,
                      coloraxis_showscale=True, margin=dict(l=0,r=0,t=50,b=60))
    st.plotly_chart(fig, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.subheader("Return Correlations")
    n_stocks = st.slider("Number of stocks (by data coverage)", 10, 48, 25)
    daily_ret = prices.pct_change().dropna()
    coverage  = daily_ret.notna().sum().sort_values(ascending=False)
    top_syms  = coverage.head(n_stocks).index.tolist()
    corr = daily_ret[top_syms].corr()

    fig2 = px.imshow(corr, color_continuous_scale='RdYlGn',
                     zmin=-0.2, zmax=1.0,
                     title=f"Correlation Matrix — Top {n_stocks} stocks by coverage")
    fig2.update_layout(height=550, margin=dict(l=0,r=0,t=50,b=0))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Risk scorecard table ──────────────────────────────────────────────────
    if risk_df is not None:
        st.subheader("Full Risk Scorecard")
        display_cols = ['symbol','sector','cagr_pct','annual_volatility_pct',
                        'sharpe_ratio','sortino_ratio','max_drawdown_pct',
                        'var_95_daily_pct','risk_level']
        avail = [c for c in display_cols if c in risk_df.columns]
        st.dataframe(risk_df[avail].round(3), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: EXPLAINABLE AI (SHAP)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainable AI":
    st.title("🧠 Explainable AI — SHAP Analysis")
    st.markdown("Understand **why** the model makes each prediction using SHAP (SHapley Additive exPlanations).")

    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.selectbox("Select Stock", symbols,
                               format_func=lambda s: f"{s}  —  {sector_map.get(s,'')}")
    with col2:
        n_samples = st.slider("Recent days to analyse", 100, 500, 300, step=50)

    if st.button("🔍 Compute SHAP Explanations"):
        import shap
        from src.predictor import _align_features, BASE_FEATURES
        import joblib

        with st.spinner("Loading model and computing SHAP values..."):
            df = get_stock_with_indicators(symbol)

            model_path = os.path.join('models', f'{symbol}_direction_clf.pkl')
            if not os.path.exists(model_path):
                st.warning(f"No saved model for {symbol}. Training now — this takes ~1 minute...")
                from src.predictor import EnsemblePredictor
                predictor = EnsemblePredictor(horizon=5)
                predictor.train(df)
                predictor.save(symbol)

            clf_obj       = joblib.load(model_path)
            xgb_model     = clf_obj['model']
            scaler        = clf_obj['scaler']
            feature_names = clf_obj['features']

            X_df     = _align_features(df, feature_names).fillna(0).tail(n_samples)
            X_scaled = pd.DataFrame(scaler.transform(X_df.values), columns=feature_names)

            explainer   = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_scaled)

        st.success(f"SHAP values computed for {n_samples} recent trading days.")

        # ── Global importance ─────────────────────────────────────────────────
        st.subheader("Global Feature Importance")
        st.markdown("Mean absolute SHAP value — how much each feature impacts predictions on average.")

        mean_shap = pd.Series(np.abs(shap_values).mean(axis=0),
                              index=feature_names).sort_values(ascending=False).head(20)
        fig = px.bar(x=mean_shap.values, y=mean_shap.index,
                     orientation='h', color=mean_shap.values,
                     color_continuous_scale='Teal',
                     labels={'x': 'Mean |SHAP value|', 'y': 'Feature'},
                     title=f"{symbol} — Top 20 Features by Global SHAP Importance")
        fig.update_layout(height=550, coloraxis_showscale=False,
                          yaxis={'categoryorder': 'total ascending'},
                          margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # ── Local explanation for latest prediction ───────────────────────────
        st.subheader("Local Explanation — Most Recent Prediction")
        st.markdown("Why did the model predict BUY or SELL for the latest data point?")

        row_shap  = shap_values[-1]
        base_val  = explainer.expected_value
        final_val = base_val + row_shap.sum()

        top12_idx    = np.argsort(np.abs(row_shap))[-12:]
        top12_sorted = top12_idx[np.argsort(row_shap[top12_idx])]
        feat_labels  = [feature_names[i] for i in top12_sorted]
        sv           = row_shap[top12_sorted]

        col1, col2, col3 = st.columns(3)
        col1.metric("Base value (avg)",   f"{base_val:.3f}")
        col2.metric("SHAP adjustment",    f"{row_shap.sum():+.3f}")
        col3.metric("Final output",        f"{final_val:.3f}",
                    delta="→ BUY" if final_val > 0.5 else "→ SELL")

        fig2 = go.Figure(go.Bar(
            x=sv, y=feat_labels, orientation='h',
            marker_color=['#2ecc71' if v > 0 else '#e74c3c' for v in sv],
            opacity=0.85,
        ))
        fig2.add_vline(x=0, line_color='black', line_width=1)
        fig2.update_layout(
            height=420, margin=dict(l=0, r=0, t=40, b=0),
            title="Feature contributions: green = pushes BUY, red = pushes SELL",
            xaxis_title="SHAP contribution",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── SHAP over time for top feature ────────────────────────────────────
        st.subheader("Top Feature SHAP Over Time")
        top_feat_idx  = int(np.argmax(np.abs(shap_values).mean(axis=0)))
        top_feat_name = feature_names[top_feat_idx]
        dates_shap    = df['Date'].tail(n_samples).values
        shap_series   = shap_values[:, top_feat_idx]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=dates_shap, y=shap_series,
                               marker_color=['#2ecc71' if v > 0 else '#e74c3c' for v in shap_series],
                               name=f'SHAP ({top_feat_name})', opacity=0.7))
        fig3.add_hline(y=0, line_color='gray', line_width=0.8)
        fig3.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0),
                            title=f"SHAP value of '{top_feat_name}' over time",
                            yaxis_title="SHAP contribution")
        st.plotly_chart(fig3, use_container_width=True)

        st.info("💡 **How to read this:** Positive SHAP = feature pushed the model toward BUY. "
                "Negative SHAP = feature pushed toward SELL. The magnitude shows how strongly.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Market Anomaly Detection")
    st.markdown("Identify unusual market events: **volatility spikes**, **extreme price moves**, and **abnormal trading volume**.")

    tab1, tab2 = st.tabs(["🔬 Single Stock", "🌐 Universe Scan"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            symbol = st.selectbox("Select Stock", symbols,
                                   format_func=lambda s: f"{s}  —  {sector_map.get(s,'')}",
                                   key="anom_stock")
        with col2:
            start_yr = st.selectbox("From Year", list(range(2005, 2022)), index=10, key="anom_yr")

        with st.spinner("Detecting anomalies..."):
            from src.anomaly import detect_all_anomalies, isolation_forest_anomalies, anomaly_summary
            df_raw  = cached_load_stock(symbol)
            df_anom = detect_all_anomalies(df_raw.copy())
            summary = anomaly_summary(df_anom, symbol)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Volatility Spikes",  summary['vol_spikes'])
        col2.metric("Extreme Drops",      summary['extreme_drops'])
        col3.metric("Volume Spikes",      summary['volume_spikes'])
        col4.metric("Total Anomalies",    f"{summary['any_anomaly']}  ({summary['anomaly_rate_pct']}%)")

        # Filter to selected period
        df_plot = df_anom[df_anom['Date'].dt.year >= start_yr].copy()

        # Price + anomaly markers
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.25, 0.25],
                            vertical_spacing=0.04)

        fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Close'],
                                  name='Close', line=dict(color='#2c3e50', width=1)), row=1, col=1)

        any_anom = df_plot[df_plot['AnyAnomaly'] == 1]
        fig.add_trace(go.Scatter(x=any_anom['Date'], y=any_anom['Close'],
                                  mode='markers', name='Anomaly',
                                  marker=dict(color='#e74c3c', size=6, symbol='x')), row=1, col=1)

        # Volatility z-score
        fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Vol_Zscore'],
                                  name='Vol Z-score', line=dict(color='#e67e22', width=1)), row=2, col=1)
        fig.add_hline(y=2.5, line_dash='dot', line_color='red', row=2, col=1)

        # Volume z-score
        fig.add_trace(go.Scatter(x=df_plot['Date'],
                                  y=df_plot['Volume_Zscore'].clip(-2, 10),
                                  name='Vol Z-score', line=dict(color='#3498db', width=1)), row=3, col=1)
        fig.add_hline(y=2.5, line_dash='dot', line_color='blue', row=3, col=1)

        fig.update_layout(height=600, title=f"{symbol} — Anomaly Detection",
                          showlegend=True, margin=dict(l=0, r=0, t=50, b=0))
        fig.update_yaxes(title_text="Price (₹)",     row=1, col=1)
        fig.update_yaxes(title_text="Vol Z-score",   row=2, col=1)
        fig.update_yaxes(title_text="Volume Z-score", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Anomalies by year bar chart
        st.subheader("Anomalies by Year")
        by_year = pd.Series(summary['anomalies_by_year']).sort_index()
        fig2 = px.bar(x=by_year.index.astype(str), y=by_year.values,
                      color=by_year.values, color_continuous_scale='Reds',
                      labels={'x': 'Year', 'y': 'Anomaly count'},
                      title=f"{symbol} — Annual Anomaly Count")
        crisis_years = {'2008': '2008 GFC', '2011': 'Euro Crisis',
                        '2015': 'China Slowdown', '2020': 'COVID'}
        for yr, label in crisis_years.items():
            if int(yr) in by_year.index:
                fig2.add_annotation(x=yr, y=by_year[int(yr)],
                            text=label, showarrow=True,
                            arrowhead=2, arrowcolor='gray',
                            font=dict(size=10, color='gray'),
                            ax=0, ay=-30)
        fig2.update_layout(height=320, coloraxis_showscale=False,
                            margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig2, use_container_width=True)

        # Top anomaly events table
        with st.expander("📋 Top 10 Most Extreme Events"):
            top_events = df_anom[df_anom['AnyAnomaly'] == 1].copy()
            top_events['AbsReturn'] = top_events['DailyReturn'].abs()
            top_events = top_events.nlargest(10, 'AbsReturn')
            display_cols = ['Date', 'Close', 'DailyReturn', 'Vol_Zscore', 'Volume_Zscore',
                            'VolSpike', 'ExtremeDrawdown', 'VolumeSpike']
            avail = [c for c in display_cols if c in top_events.columns]
            st.dataframe(top_events[avail].round(3).reset_index(drop=True),
                         use_container_width=True)

    with tab2:
        st.subheader("Universe-Wide Anomaly Scan")
        st.markdown("Compare anomaly rates across all 50 stocks to identify structurally riskier names.")

        if st.button("🔍 Run Full Universe Scan (~2 min)"):
            with st.spinner("Scanning all stocks for anomalies..."):
                from src.anomaly import scan_universe
                scan_df = scan_universe()
                scan_df['sector'] = scan_df.index.map(sector_map)
                scan_df = scan_df.sort_values('anomaly_rate_pct', ascending=False)

            fig3 = px.bar(scan_df.reset_index(), x='symbol', y='anomaly_rate_pct',
                          color='sector', title="Anomaly Rate by Stock (% of trading days)",
                          labels={'anomaly_rate_pct': 'Anomaly Rate (%)', 'symbol': 'Stock'},
                          color_discrete_sequence=px.colors.qualitative.Set3)
            fig3.update_layout(height=420, xaxis_tickangle=-45,
                               margin=dict(l=0, r=0, t=50, b=60))
            st.plotly_chart(fig3, use_container_width=True)

            st.dataframe(scan_df[['sector', 'vol_spikes', 'extreme_drops',
                                   'volume_spikes', 'any_anomaly', 'anomaly_rate_pct']].round(2),
                         use_container_width=True)
        else:
            st.info("Click the button above to scan all 50 stocks. Results take about 2 minutes to compute.")
