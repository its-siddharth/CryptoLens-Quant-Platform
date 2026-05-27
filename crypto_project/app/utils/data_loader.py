import pandas as pd
import streamlit as st
import os
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')

@st.cache_data
def load_csv(filename):
    """Load a CSV file from the data directory."""
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        # Read the file
        df = pd.read_csv(path)
        # Try to parse date columns if they exist
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_historical_klines(symbol, interval='1d', limit=365):
    """Fetch live historical kline (candlestick) data from Binance."""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data, columns=[
                'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 
                'CloseTime', 'QuoteVolume', 'NumTrades', 
                'TakerBuyBase', 'TakerBuyQuote', 'Ignore'
            ])
            df['Date'] = pd.to_datetime(df['OpenTime'], unit='ms')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'QuoteVolume']:
                df[col] = df[col].astype(float)
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching historical klines for {symbol}: {e}")
    return pd.DataFrame()