import streamlit as st
from utils.theme import apply_custom_css

st.set_page_config(
    page_title="CryptoLens: Advanced Quantitative Analytics Platform",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global premium theme overrides
apply_custom_css()



# Hero Section
st.markdown("<h1 style='text-align: center; margin-top: 1.5rem; font-size: 3.5rem; background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🌌 CryptoLens Quant Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.35rem; color: #94a3b8; max-width: 800px; margin: 0 auto 3rem auto; line-height: 1.6;'>An advanced mathematical intelligence platform combining deep learning LSTMs, traditional statistical estimators (ARIMA), Bayesian frameworks (Meta Prophet), and real-time Natural Language Processing.</p>", unsafe_allow_html=True)

# Interactive Project Overview & Architecture Pipeline
st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>🖥️ Processing Pipeline & Architecture</h2>", unsafe_allow_html=True)

architecture_html = """
<div style="display: flex; flex-direction: row; justify-content: space-between; align-items: stretch; gap: 15px; margin-bottom: 3rem; flex-wrap: wrap;">
    <div style="flex: 1; min-width: 220px; background: rgba(15, 23, 42, 0.4); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
        <div style="font-size: 2.2rem; margin-bottom: 10px;">📡</div>
        <h4 style="margin: 0 0 8px 0; color: #38bdf8; font-size: 1.1rem;">1. Data Ingestion</h4>
        <p style="margin: 0; font-size: 0.88rem; color: #94a3b8; line-height: 1.4;">Fetches spot tick candlesticks from Binance API & scrapes global RSS cryptographic feeds natively.</p>
    </div>
    <div style="flex: 0.1; display: flex; justify-content: center; align-items: center; min-width: 20px; color: #38bdf8; font-size: 1.5rem;">➡️</div>
    <div style="flex: 1; min-width: 220px; background: rgba(15, 23, 42, 0.4); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
        <div style="font-size: 2.2rem; margin-bottom: 10px;">⚙️</div>
        <h4 style="margin: 0 0 8px 0; color: #38bdf8; font-size: 1.1rem;">2. Feature Engineering</h4>
        <p style="margin: 0; font-size: 0.88rem; color: #94a3b8; line-height: 1.4;">Executes MinMax normalization, calculates technical indicators (RSI, MA, Volatility) & processes NLP text.</p>
    </div>
    <div style="flex: 0.1; display: flex; justify-content: center; align-items: center; min-width: 20px; color: #38bdf8; font-size: 1.5rem;">➡️</div>
    <div style="flex: 1; min-width: 220px; background: rgba(15, 23, 42, 0.4); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
        <div style="font-size: 2.2rem; margin-bottom: 10px;">🧠</div>
        <h4 style="margin: 0 0 8px 0; color: #38bdf8; font-size: 1.1rem;">3. Model Inference</h4>
        <p style="margin: 0; font-size: 0.88rem; color: #94a3b8; line-height: 1.4;">Deploys deep recurrent LSTM nodes, Bayesian trend propagators, and autoregressive models.</p>
    </div>
    <div style="flex: 0.1; display: flex; justify-content: center; align-items: center; min-width: 20px; color: #38bdf8; font-size: 1.5rem;">➡️</div>
    <div style="flex: 1; min-width: 220px; background: rgba(15, 23, 42, 0.4); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
        <div style="font-size: 2.2rem; margin-bottom: 10px;">📊</div>
        <h4 style="margin: 0 0 8px 0; color: #38bdf8; font-size: 1.1rem;">4. Quant Dashboard</h4>
        <p style="margin: 0; font-size: 0.88rem; color: #94a3b8; line-height: 1.4;">Renders interactive quant charting, news NLP sentiment index, and automated trading signal verdicts.</p>
    </div>
</div>
"""
st.markdown(architecture_html, unsafe_allow_html=True)

st.divider()

# Features Grid Layout (Using our custom glassmorphic premium card class)
st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Core Analytical Modules</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="premium-card">
            <h3 style="margin-top:0;">📈 Live Spot Market Ticker</h3>
            <p style="color: #94a3b8; font-size: 1rem; line-height: 1.5; margin-bottom: 15px;">
                Stream real-time exchange rates for Bitcoin (BTC), Ethereum (ETH), and Solana (SOL) from the Binance spot market API. Inspect detailed statistics, volume structures, and calculate dynamic trading buy/sell signals on-the-fly.
            </p>
            <span style="background-color: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 500;">Binance Integration</span>
            <span style="background-color: rgba(16, 185, 129, 0.15); color: #10b981; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 500; margin-left: 5px;">Live Feeds</span>
        </div>
        
        <div class="premium-card" style="margin-top: 25px;">
            <h3 style="margin-top:0;">🔮 Dynamic Forecasting Suite</h3>
            <p style="color: #94a3b8; font-size: 1rem; line-height: 1.5; margin-bottom: 15px;">
                Run dynamic multi-horizon predictions. Compare state-of-the-art Deep Learning Recurrent Networks (LSTM) with statistical ARIMA and Bayesian Prophet frameworks. Fully interactive parameters allow adjustments to rolling horizons and confidence limits.
            </p>
            <span style="background-color: rgba(139, 92, 246, 0.15); color: #a78bfa; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 500;">Deep Learning</span>
            <span style="background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 500; margin-left: 5px;">Multi-Model Predictors</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="premium-card">
            <h3 style="margin-top:0;">📊 Exploratory Quantitative Analysis</h3>
            <p style="color: #94a3b8; font-size: 1rem; line-height: 1.5; margin-bottom: 15px;">
                Analyze mathematical statistics of the asset returns distribution, joint correlations heatmaps, and rolling 30-day historical volatility metrics. Investigate a newly introduced Cumulative Growth Benchmark to compare long-term return potentials.
            </p>
            <span style="background-color: rgba(236, 72, 153, 0.15); color: #f472b6; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 500;">Statistical Mathematics</span>
            <span style="background-color: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 500; margin-left: 5px;">Quant Diagnostics</span>
        </div>
        
        <div class="premium-card" style="margin-top: 25px;">
            <h3 style="margin-top:0;">🧠 Real-Time NLP Sentiment Processing</h3>
            <p style="color: #94a3b8; font-size: 1rem; line-height: 1.5; margin-bottom: 15px;">
                Harvest headlines from global cryptocurrency news feeds and process emotional polarities. Utilize a state-of-the-art interactive sentiment sandbox to evaluate customized headlines and gauge direct price-velocity correlations.
            </p>
            <span style="background-color: rgba(16, 185, 129, 0.15); color: #10b981; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 500;">Natural Language Processing</span>
            <span style="background-color: rgba(239, 68, 68, 0.15); color: #f87171; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 500; margin-left: 5px;">Text Mining</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# Call to Action
st.markdown("""
<div style="text-align: center; background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%); padding: 35px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 10px 30px rgba(0,0,0,0.4);">
    <h3 style="margin-top: 0; background: linear-gradient(135deg, #38bdf8 0%, #10b981 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.5rem;">Begin Quantitative Analysis 🚀</h3>
    <p style="color: #cbd5e1; font-size: 1.1rem; margin-bottom: 0;">Use the sidebar navigator on the left to explore the specialized modules and start live estimations.</p>
</div>
""", unsafe_allow_html=True)