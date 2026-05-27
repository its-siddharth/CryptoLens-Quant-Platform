import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils.data_loader import get_historical_klines, load_csv
from utils.theme import apply_custom_css

st.set_page_config(page_title="Quantitative EDA", page_icon="📊", layout="wide")
apply_custom_css()

st.markdown("<h1>📊 Quantitative Market Exploratory Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-top:-10px; margin-bottom: 2rem;'>Explore statistical features, mathematical return behaviors, asset dependencies, and risk metrics dynamically sourced from active markets.</p>", unsafe_allow_html=True)

symbols = {"BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum", "SOLUSDT": "Solana"}

# Quant Workbench Sidebar Options
st.sidebar.markdown("### 🎛️ Analysis Parameters")
days = st.sidebar.slider("Select Historical Horizon (Days)", min_value=30, max_value=730, value=365, step=15)
vol_window = st.sidebar.selectbox("Volatility Rolling Window", [7, 14, 30, 90], index=2, format_func=lambda x: f"{x}-Day Window")

@st.cache_data(ttl=3600)
def load_all_data(days):
    dfs = {}
    for sym, name in symbols.items():
        df = get_historical_klines(sym, interval='1d', limit=days)
        if not df.empty:
            df.set_index('Date', inplace=True)
            dfs[name] = df
        else:
            # Fallback to local raw CSV
            local_file = f"{sym[:3].lower()}_raw.csv" # e.g. btc_raw.csv
            df_local = load_csv(local_file)
            if not df_local.empty:
                df_local.columns = [c.capitalize() if c.lower() != 'date' else 'Date' for c in df_local.columns]
                df_local.set_index('Date', inplace=True)
                dfs[name] = df_local.tail(days)
    return dfs

with st.spinner("Harvesting historical data structures from Binance network..."):
    dfs = load_all_data(days)

if not dfs:
    st.error("Failed to establish high-speed datastream with Binance network. Attempting backend buffer reload...")
    st.stop()

# Build mutual price metrics
price_df = pd.DataFrame({name: df['Close'] for name, df in dfs.items()}).dropna()
returns_df = price_df.pct_change().dropna() * 100

tabs = st.tabs(["💵 Normalized Movements & Growth", "📐 Joint Return Distribution & Dependencies", "⚡ Risk & Volatility Analytics"])

# TABS 1: Price Movements & Cumulative Growth
with tabs[0]:
    st.markdown("### Asset Growth Benchmarking ($1,000 Base)")
    st.markdown(
        "This metric compares how a base investment of **$1,000 USD** would have grew over the selected timeline. "
        "Unlike raw pricing, cumulative growth scales all assets proportionally, highlighting total compound yield performance."
    )
    
    if not price_df.empty:
        # Calculate cumulative growth from daily percentage changes
        daily_returns_frac = returns_df / 100
        cumulative_growth = (1 + daily_returns_frac).cumprod() * 1000
        # Set base value at start as 1000
        first_row = pd.DataFrame([[1000.0, 1000.0, 1000.0]], columns=cumulative_growth.columns, index=[price_df.index[0]])
        cumulative_growth = pd.concat([first_row, cumulative_growth])
        
        st.line_chart(cumulative_growth)
        
        st.divider()
        
        st.markdown("### Normalized Relative Price Index (Base = 100)")
        st.markdown(
            "Visualizes the percentage changes relative to the starting day of the selected period. "
            "Helps evaluate co-movement trends and general price-action beta correlations across all three networks."
        )
        normalized_prices = (price_df / price_df.iloc[0]) * 100
        st.line_chart(normalized_prices)
    else:
        st.info("Pricing data currently loading. Check pipeline socket.")

# TABS 2: Distributions & Correlations
with tabs[1]:
    st.markdown("### Joint Mathematical return Characteristics")
    st.markdown(
        "Analyzing return metrics provides insights into market efficiency and systemic dependency structures. "
        "Standard asset pricing theories rely on return stationarity and joint correlation dynamics."
    )
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("#### Daily returns Distribution Histogram")
        st.markdown(
            "Plots the frequency distribution of daily returns. Notice how cryptocurrency returns exhibit "
            "**leptokurtosis** (fat tails and high peaks), indicating a higher probability of extreme events than a standard normal curve."
        )
        melted_returns = returns_df.reset_index().melt(id_vars='Date', var_name='Asset', value_name='Daily Return (%)')
        
        chart = alt.Chart(melted_returns).mark_bar(opacity=0.65, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            alt.X('Daily Return (%):Q', bin=alt.Bin(maxbins=40), title="Daily Return (%)"),
            alt.Y('count()', title='Observations Frequency'),
            color=alt.Color('Asset:N', scale=alt.Scale(range=['#38bdf8', '#ec4899', '#f59e0b']))
        ).properties(height=320).configure_view(strokeWidth=0)
        
        st.altair_chart(chart, use_container_width=True)
        
    with col2:
        st.markdown("#### Pearson Correlation Coefficient Matrix")
        st.markdown(
            "Measures the linear dependence between daily asset returns (scaled -1 to 1). "
            "A high positive correlation indicates that assets move in tandem, implying shared systematic market risk factors."
        )
        corr = returns_df.corr().reset_index().melt(id_vars='index')
        corr.columns = ['Asset 1', 'Asset 2', 'Correlation']
        
        heat = alt.Chart(corr).mark_rect().encode(
            x=alt.X('Asset 1:O', title=''),
            y=alt.Y('Asset 2:O', title=''),
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis', domain=[-0.2, 1.0])),
            tooltip=['Asset 1', 'Asset 2', 'Correlation']
        ).properties(height=320)
        
        text = heat.mark_text(baseline='middle', font='Outfit', fontWeight='bold', fontSize=14).encode(
            text=alt.Text('Correlation:Q', format='.2f'),
            color=alt.condition(
                alt.datum.Correlation > 0.6,
                alt.value('black'),
                alt.value('white')
            )
        )
        st.altair_chart(heat + text, use_container_width=True)

# TABS 3: Volatility & Volume
with tabs[2]:
    st.markdown("### Risk Dynamics & Trading Volumes")
    st.markdown(
        "Risk is mathematically expressed as historical volatility. We evaluate the annualized standard deviation "
        "of price returns to observe volatility clustering."
    )
    
    col_v1, col_v2 = st.columns([2, 1], gap="medium")
    
    with col_v1:
        st.markdown(f"#### Rolling {vol_window}-Day Volatility Curves (Annualized)")
        st.markdown(
            f"Tracks the rolling standard deviation of daily returns over a {vol_window}-day lookback window. "
            "Spikes in this metric represent **volatility clustering**, where periods of high variance are followed by high variance."
        )
        # Annualized volatility: std dev of daily returns * sqrt(365)
        volatility_df = returns_df.rolling(window=vol_window).std() * np.sqrt(365)
        st.line_chart(volatility_df.dropna())
        
    with col_v2:
        st.markdown("#### Trading Volume Distribution")
        st.markdown("Examine liquid transactional volume across networks. Liquidity directly influences market slippage.")
        selected_asset_vol = st.selectbox("Choose Target Asset for Volume Logs:", list(symbols.values()))
        if selected_asset_vol in dfs:
            vol_df = dfs[selected_asset_vol][['Volume']]
            st.bar_chart(vol_df)
            
st.divider()

# Quantitative Summary Box
st.markdown(
    """
    <div style="background: rgba(30, 41, 59, 0.3); border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.05);">
        <h4 style="margin:0 0 10px 0; color:#38bdf8; font-size:1.15rem;">📐 Seminar Takeaway Note</h4>
        <p style="margin:0; font-size:0.95rem; color:#cbd5e1; line-height:1.5;">
            These statistics indicate that while Solana (SOL) consistently offers higher potential yields (observable in the Cumulative Growth tab), it carries significantly higher volatility exposure (reflected in the fat tails of its return distribution and elevated annualized volatility). Bitcoin remains the pricing beta of the digital assets market, displaying strong systematic correlation boundaries.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)