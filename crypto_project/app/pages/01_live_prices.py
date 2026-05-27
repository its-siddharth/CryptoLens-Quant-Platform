import streamlit as st
import pandas as pd
import requests
import numpy as np
from utils.data_loader import load_csv, get_historical_klines
from utils.theme import apply_custom_css

# Try importing Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None

st.set_page_config(page_title="Live Prices & Charts", page_icon="📈", layout="wide")
apply_custom_css()


st.markdown("<h1>📈 Live Market Quant Ticker</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-top:-10px; margin-bottom: 2rem;'>Stream real-time price tick data from Binance API and compute dynamic algorithmic trading signals natively.</p>", unsafe_allow_html=True)

coins = {
    "Bitcoin (BTC)": {"file": "btc_raw.csv", "symbol": "BTCUSDT", "desc": "Digital Gold / Layer 1 Settlement"}, 
    "Ethereum (ETH)": {"file": "eth_raw.csv", "symbol": "ETHUSDT", "desc": "Smart Contracts / World Computer"}, 
    "Solana (SOL)": {"file": "sol_raw.csv", "symbol": "SOLUSDT", "desc": "Ultra-High Throughput L1"}
}

@st.cache_data(ttl=30)
def get_live_price(symbol):
    """Fetch live ticker details from Binance API."""
    try:
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "price": float(data['lastPrice']),
                "change": float(data['priceChange']),
                "perc": float(data['priceChangePercent']),
                "high": float(data['highPrice']),
                "low": float(data['lowPrice']),
                "volume": float(data['volume']),
                "quote_vol": float(data['quoteVolume'])
            }
    except Exception:
        pass
    return None

def compute_technical_indicators(df):
    """Compute SMA, EMA, and RSI over historical dataframe."""
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # RSI 14 Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def generate_ai_verdict(df, live_data):
    """Compute technical consensus and output a human-like smart trading recommendation."""
    if df.empty or len(df) < 50:
        return "HOLD", "Insufficient historical nodes to run algorithmic evaluation.", "#64748b"
    
    df = compute_technical_indicators(df)
    latest = df.iloc[-1]
    
    rsi = latest['RSI']
    sma = latest['SMA_20']
    ema = latest['EMA_50']
    close = live_data['price'] if live_data else latest['Close']
    
    # Quantitative scoring system
    bullish_signals = 0
    total_signals = 3
    
    # RSI Signal
    if rsi < 30:
        rsi_state = "Oversold 🟢"
        bullish_signals += 1
    elif rsi > 70:
        rsi_state = "Overbought 🔴"
    else:
        rsi_state = "Neutral ⚖️"
        bullish_signals += 0.5
        
    # Moving Average Positioning
    if close > sma:
        bullish_signals += 1
    if close > ema:
        bullish_signals += 1
        
    score = bullish_signals / total_signals
    
    if score >= 0.8:
        verdict = "STRONG BUY"
        description = f"Algorithmic model indicates strong upward momentum. Price (${close:,.2f}) is firmly above the 20-Day SMA and 50-Day EMA with a supportive RSI ({rsi:.1f}) in the accumulation phase."
        color = "#10b981"
    elif score >= 0.5:
        verdict = "BUY"
        description = f"Model signals moderate bullish posture. Momentum is turning positive with close price above key historical moving averages; RSI is comfortable at {rsi:.1f}."
        color = "#34d399"
    elif score >= 0.35:
        verdict = "HOLD"
        description = f"Consensus is rangebound. Asset exhibits volatile oscillation between support and resistance layers. Current RSI of {rsi:.1f} suggests market consolidation."
        color = "#f59e0b"
    else:
        verdict = "SELL"
        description = f"Bearish breakout signal. Price has slid below intermediate moving averages (SMA/EMA), hinting at distribution pressure. Proceed with caution."
        color = "#ef4444"
        
    return verdict, description, color

# Render Live Metrics Grid
cols = st.columns(3)
dfs = {}
live_metrics = {}

for idx, (name, info) in enumerate(coins.items()):
    # Pre-load data to ensure we have it for fallback and signal calculations
    df = load_csv(info["file"])
    if not df.empty:
        # Standardize column names to Title Case to match Binance API structure exactly
        df.columns = [c.capitalize() if c.lower() != 'date' else 'Date' for c in df.columns]
        dfs[name] = df
        
    # Live API fetch
    live = get_live_price(info["symbol"])
    
    if live:
        live_metrics[name] = live
        cols[idx].metric(
            label=f"🟢 {name} (Live)", 
            value=f"${live['price']:,.2f}", 
            delta=f"${live['change']:,.2f} ({live['perc']:.2f}%)"
        )
    else:
        # Fallback to local
        if name in dfs:
            close_col = 'Close' if 'Close' in dfs[name].columns else 'close' if 'close' in dfs[name].columns else None
            if close_col and len(dfs[name]) > 1:
                latest = dfs[name].iloc[-1][close_col]
                prev = dfs[name].iloc[-2][close_col]
                diff = latest - prev
                perc = (diff / prev) * 100
                cols[idx].metric(
                    label=f"💾 {name} (Archived)", 
                    value=f"${latest:,.2f}", 
                    delta=f"${diff:,.2f} ({perc:.2f}%)"
                )

st.divider()

# Selected Asset Quant Workbench
st.markdown("### 🛠️ Interactive Asset Workbench")
coin_select = st.selectbox("Select Target Cryptographic Asset to Analyze:", list(coins.keys()))

if coin_select in dfs:
    df_asset = dfs[coin_select]
    live_data = live_metrics.get(coin_select, None)
    
    # Calculate detailed statistics on the fly
    high_24h = live_data['high'] if live_data else df_asset['High'].max()
    low_24h = live_data['low'] if live_data else df_asset['Low'].min()
    vol_24h = live_data['quote_vol'] if live_data else (df_asset['Volume'].mean() * df_asset['Close'].mean())
    volatility = df_asset['Close'].pct_change().std() * np.sqrt(365) * 100
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.markdown(f"<div style='text-align:center; padding:10px; background:rgba(255,255,255,0.03); border-radius:10px;'><b>24h Spot High</b><br><span style='color:#38bdf8; font-size:1.25rem;'>${high_24h:,.2f}</span></div>", unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"<div style='text-align:center; padding:10px; background:rgba(255,255,255,0.03); border-radius:10px;'><b>24h Spot Low</b><br><span style='color:#38bdf8; font-size:1.25rem;'>${low_24h:,.2f}</span></div>", unsafe_allow_html=True)
    with col_stat3:
        st.markdown(f"<div style='text-align:center; padding:10px; background:rgba(255,255,255,0.03); border-radius:10px;'><b>24h Volume (USD)</b><br><span style='color:#38bdf8; font-size:1.25rem;'>${vol_24h:,.0f}</span></div>", unsafe_allow_html=True)
    with col_stat4:
        st.markdown(f"<div style='text-align:center; padding:10px; background:rgba(255,255,255,0.03); border-radius:10px;'><b>Annualized Volatility</b><br><span style='color:#f43f5e; font-size:1.25rem;'>{volatility:.2f}%</span></div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # AI Algorithmic Trading Consensus Panel
    verdict, description, color = generate_ai_verdict(df_asset, live_data)
    
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.4); border-left: 5px solid {color}; padding: 20px; border-radius: 12px; border-top: 1px solid rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.04); border-bottom: 1px solid rgba(255,255,255,0.04); margin-bottom: 25px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <h4 style="margin: 0; color: #94a3b8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em;">AI Algorithmic Verdict</h4>
                    <h2 style="margin: 5px 0 0 0; color: {color}; font-size: 2rem; font-weight: 800;">{verdict}</h2>
                </div>
                <div style="background: {color}15; color: {color}; padding: 6px 14px; border-radius: 20px; font-weight: bold; border: 1px solid {color}30;">
                    Technical Consensus Signal
                </div>
            </div>
            <p style="margin: 15px 0 0 0; color: #e2e8f0; font-size: 1rem; line-height: 1.5;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Dynamic Interactive Plotly Candlestick Chart (Quant standard)
    col_chart_header, col_chart_ctrl = st.columns([3, 2])
    with col_chart_header:
        st.markdown(f"#### Interactive Candlestick Technical Chart ({coin_select})")
    with col_chart_ctrl:
        indicators = st.multiselect("Toggle Chart Overlays", ["SMA (20-Day)", "EMA (50-Day)"], default=["SMA (20-Day)"], label_visibility="collapsed")
    
    if go is not None:
        df_chart = df_asset.tail(150).copy()
        df_chart = compute_technical_indicators(df_chart)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.08, 
                            row_width=[0.2, 0.8])
        
        # 1. Candlestick
        fig.add_trace(go.Candlestick(
            x=df_chart['Date'],
            open=df_chart['Open'],
            high=df_chart['High'],
            low=df_chart['Low'],
            close=df_chart['Close'],
            name="Spot Price",
            increasing_line_color='#10b981', 
            decreasing_line_color='#ef4444',
            showlegend=True
        ), row=1, col=1)
        
        # 2. Moving Average Overlays
        if "SMA (20-Day)" in indicators:
            fig.add_trace(go.Scatter(
                x=df_chart['Date'],
                y=df_chart['SMA_20'],
                mode='lines',
                line=dict(color='#38bdf8', width=1.5),
                name="20-Day SMA"
            ), row=1, col=1)
            
        if "EMA (50-Day)" in indicators:
            fig.add_trace(go.Scatter(
                x=df_chart['Date'],
                y=df_chart['EMA_50'],
                mode='lines',
                line=dict(color='#ec4899', width=1.5),
                name="50-Day EMA"
            ), row=1, col=1)
            
        # 3. Volume bars
        colors = ['#10b981' if row['Close'] >= row['Open'] else '#ef4444' 
                  for _, row in df_chart.iterrows()]
        fig.add_trace(go.Bar(
            x=df_chart['Date'],
            y=df_chart['Volume'],
            marker_color=colors,
            name="Volume",
            showlegend=False
        ), row=2, col=1)
        
        # Custom layout styling
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(rangeslider=dict(visible=False), gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Price (USD)"),
            yaxis2=dict(gridcolor='rgba(255,255,255,0.05)', title="Volume"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to standard line chart if Plotly hasn't completed loading
        chart_df = df_asset[['Date', 'Close']].set_index('Date').tail(150)
        st.line_chart(chart_df)
        st.caption("Plotly libraries are currently configuring in the background. Displaying intermediate line charts.")

    st.markdown("#### Detailed Historical Ledger Logs")
    # Clean output columns
    disp_cols = [c for c in df_asset.columns if c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    st.dataframe(df_asset[disp_cols].tail(10).sort_values('Date', ascending=False), use_container_width=True, hide_index=True)

else:
    st.info("Configuring live datastream connection nodes. Attempting backfill...")