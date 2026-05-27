# 🌌 CryptoLens: Advanced Quantitative Analytics & Forecasting Platform

[![Live App](https://img.shields.io/badge/Live%20Platform-Online-10b981?style=for-the-badge&logo=streamlit&logoColor=white)](https://cryptolens-quant-platform.streamlit.app/)


CryptoLens is a state-of-the-art financial engineering workbench combining deep learning Recurrent Neural Networks (LSTM), statistical autoregression (ARIMA), and Bayesian time-series curve fitting (Meta Prophet) with live Natural Language Processing (NLP) sentiment metrics. 

Designed for portfolio managers, quant developers, and academic research, it harvests live order book tick feeds from the Binance API and scrapes social feeds to deliver dynamic consensus trading signals natively in your browser.

---

## 🖥️ System Architecture & Processing Pipeline

```
📡 Ingestion Layer       ➡️     ⚙️ Feature Engineering    ➡️    🧠 Model Inference Layer   ➡️    📊 UI Quant Dashboard
Binance Spot Candlesticks      MinMax Price Scaling           Deep Recurrent LSTM Nodes       Plotly Candlesticks
RSS Global News Streams        Technical Indicators (RSI/MA)  Bayesian Piecewise Growth      Live NLP Sandbox
News Text Feeds                Polarity NLP Classification    ARIMA(5,1,0) Autoregressive    Volatility Metrics
```

1. **Data Ingestion Layer**: Harvesting real-time spot candlesticks from Binance API and scraping Cointelegraph RSS networks.
2. **Feature Engineering**: Standardizing inputs with MinMax scaling, calculating technical overlays (SMA, EMA, RSI, Rolling Volatility), and extracting textual subjectivity metrics.
3. **Model Inference Layer**: Deploying statistical estimators (ARIMA), Bayesian curve propagators, and recurrent networks (LSTM).
4. **Quant Dashboard**: Renders interactive subplots, actual-vs-predicted benchmarks, news feeds, and an interactive NLP sentiment sandbox.

---

## 🔮 Forecasting Models & Performance Metrics

CryptoLens evaluates out-of-sample performance across three distinct statistical and neural frameworks:

| Model | MAE ($) | RMSE ($) | MAPE (%) | Optimal Horizon | Algorithmic Suited Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM** | $6,137.16 | $6,892.80 | 6.50% | Short-Term (1-7 Days) | Non-Linear Speculative Momentum |
| **Prophet** | $16,044.31 | $17,267.28 | 16.82% | Long-Term (30-180 Days) | Piecewise Macro Trend Planning |
| **ARIMA** | $22,489.16 | $24,295.43 | 23.41% | Medium-Term (7-21 Days) | Autoregressive Mean Reversion |

### Mathematical Formulations:
*   **Mean Absolute Error (MAE)**: Measures average absolute dollar accuracy linearly.
    $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
*   **Root Mean Squared Error (RMSE)**: Squaring the error terms exponentially penalizes large black-swan outliers (critical for risk-limit limits).
    $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
*   **Mean Absolute Percentage Error (MAPE)**: Scale-independent relative metric for multi-asset benchmarking.
    $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

---

## 🧠 Real-Time NLP Sentiment Sandbox
An interactive sandbox utilizing NLTK TextBlob lexicons classifying text headlines or tweets into real-time polarity scores (scaled -1 to +1) and subjectivity values (scaled 0 to 1), outputting dynamic consensus orders to buy/sell assets.

---

## 🛠️ Installation & Execution

### Prerequisites
*   Python 3.11 or 3.12
*   Pip package manager

### Standard setup:
1. Clone the repository:
   ```bash
   git clone https://github.com/its-siddharth/CryptoLens-Quant-Platform.git
   cd CryptoLens-Quant-Platform
   ```
2. Install quantitative and UI dependencies:
   ```bash
   pip install streamlit altair plotly textblob scipy pandas numpy requests
   ```
3. Initialize the Streamlit platform locally:
   ```bash
   streamlit run crypto_project/app/main.py
   ```
4. Access the workspace in your browser at: `http://localhost:8501`
