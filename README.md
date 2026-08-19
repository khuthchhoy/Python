# 📈 Real-World AI Stock Price Prediction Engine & Quantitative Trading System

An end-to-end financial machine learning system engineered to predict stock prices and trend direction across multiple horizons (**10m, 20m, 30m, 1h, 4h, 1d, 1w**) using multi-model ensembles (Quantile Boosted Trees + PyTorch Temporal Sequence Attention), direct Yahoo Finance v8/v10 live chart data, quantitative technical indicators (Garman-Klass Volatility, CMF, StochRSI, overnight gaps), cross-asset macro market regime features, mark-to-market walk-forward backtesting, FastAPI REST/WebSocket backend, and a native SwiftUI iOS app.

---

## 🌟 Key Architecture & Methodology

### 1. Direct Live Market Data Ingestion
- **Ultra-Low Latency Yahoo Finance v8/v10 Chart Engine**: Directly queries Yahoo chart APIs with browser headers, bypassing crumb/session throttle issues.
- **Intraday & Daily Multi-Interval Feeds**: Full support for `1m`, `5m`, `15m`, `30m`, `1h`, and `1d` bars.
- **Smart Local Parquet Caching**: Interval-aware caching prevents redundant network queries while keeping quotes fresh.
- **High-Fidelity Stochastic Generator Fallback**: Calibrated Geometric Brownian Motion with stochastic volatility and realistic anchor prices when offline.

### 2. Stationary Target Formulation
Predicting non-stationary raw prices directly causes standard machine learning models to collapse into a trivial persistence baseline ($P_{t+h} \approx P_t$), displaying high $R^2$ during training but producing zero directional alpha in live trading.
To solve this, this system formulates:
- **Stationary Forward Log Returns**:
  $$r_{t+h} = \ln\left(\frac{P_{t+h}}{P_t}\right)$$
- **Forward Price Target Calculation**:
  $$\hat{P}_{t+h} = P_t \cdot \exp(\hat{r}_{t+h})$$
- **Monotonic Uncertainty Bounds (10th - 90th Percentile Quantiles)**:
  $$P_{lower} = P_t \cdot \exp(\hat{q}_{10}), \quad P_{upper} = P_t \cdot \exp(\hat{q}_{90}) \quad \text{where } \hat{q}_{10} \le \hat{r}_{t+h} \le \hat{q}_{90}$$

### 3. Over 40+ Engineered Alpha Features
- **Extreme-Value Volatility Estimators**: Garman-Klass Volatility, Parkinson Volatility, Normalized ATR (NATR 14), 10d & 30d Historical Volatility.
- **Money Flow & Volume Dynamics**: Chaikin Money Flow (CMF 20), On-Balance Volume (OBV trend), VWAP Ratio, 20-day Volume SMA Ratio.
- **Microstructure & Session Dynamics**: Overnight Gap return ($Open_t / Close_{t-1} - 1$) vs Intraday return ($Close_t / Open_t - 1$), candle spreads, and shadows.
- **Momentum & Oscillators**: Multi-Period RSI (7, 14, 21), Stochastic RSI (%K, %D), Stochastic Oscillator (%K, %D), Price-Normalized MACD (Line, Signal, Hist), Rate of Change (ROC 3, 5, 6, 10, 12), Williams %R.
- **Trend & Moving Averages**: SMA 5/10/20/50, EMA 5/9/13/21/50, 20/50 Day Cross ratio.
- **Market Context & Regimes**: S&P 500 (`SPY`) Beta (60-day), Relative Strength vs SPY (5d, 20d), SPY Trend Regime (>20d SMA), CBOE Volatility Index (`^VIX`) levels & changes.
- **Calendar Seasonality**: Cyclical sine/cosine transforms for day-of-week, month-of-year, and hour-of-day.

### 4. Multi-Model Meta-Ensemble
- **Gradient Boosted Trees (HistGradientBoosting)**: High non-linear feature interaction modeling with Quantile Pinball loss regressors enforcing monotonic intervals.
- **Deep Temporal Neural Network (PyTorch Temporal Attention Sequence Network)**: Multi-step sliding lookback sequences ($T=20$ bars) with Additive Temporal Attention dynamically weighting historical time steps.
- **Directional Movement Classifier**: Calibrated probability $P(\text{Up})$ for generating actionable signals (`STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL`).
- **Multi-Horizon Term Structure Diffusion**: Diffusion-scaled trajectory projections across milestones (10m, 20m, 30m, 1h, 4h, 1d, 1w).

### 5. Mark-to-Market Walk-Forward Backtesting
Purged and embargoed splits prevent future data leakage, evaluating genuine out-of-sample directional hit rate, MAE, RMSE, Sharpe Ratio, Sortino Ratio, Calmar Ratio, and maximum drawdown with daily mark-to-market P&L.

---

## 🚀 Quickstart Guide

### Installation
```bash
cd python
pip install -r requirements.txt
```

### 1. Launch FastAPI Backend
```bash
python3 api.py
# Runs on http://0.0.0.0:8000
```

#### Available API Endpoints:
- `GET /api/health` — API health and model engine status.
- `GET /api/forecast?ticker=NVDA&timeframe=1w` — Complete forecast with candles, signal, intervals, and backtest metrics.
- `GET /api/quote?ticker=AAPL` — Fast real-time quote (price, day change, volume, high, low).
- `GET /api/watchlist` — Batch scan and signals for watchlist stocks.
- `GET /api/screener` — Market opportunity screener ranking top liquidity leaders.
- `GET /api/backtest?ticker=NVDA` — Detailed out-of-sample backtest equity curve.
- `WS /ws/live/{ticker}` — Real-time live price and prediction streaming.

### 2. Launch Interactive Streamlit Web Dashboard
```bash
python3 main.py --web
# or
streamlit run stock_predictor/app/dashboard.py
```
Open your browser at `http://localhost:8501`.

### 3. Run CLI Forecast
```bash
# Predict 1-week ahead for NVDA
python3 cli.py --ticker NVDA --timeframe 1w

# Predict 10-minute intraday for AAPL
python3 cli.py --ticker AAPL --timeframe 10m

# Run live streaming progress in terminal
python3 cli.py --ticker TSLA --live
```

### 4. Run Automated Tests
```bash
pytest -v
```

---

## 📱 iOS SwiftUI Native Client (`StockPredictor`)

The SwiftUI app provides a financial terminal experience for iPhone and iPad:
- **Live Stream Status**: Auto-refreshes every 5 seconds or on-demand.
- **Configurable Backend Host**: Switch between `http://127.0.0.1:8000` (Simulator) and your Mac's LAN IP (e.g. `http://192.168.1.50:8000`) for running on physical iPhones.
- **Watchlist & Signals Strip**: 1-tap switching across popular tickers (NVDA, AAPL, MSFT, TSLA, AMZN, META, SPY, etc.).
- **Multi-Horizon Trajectory**: Term structure projection cards from 10 minutes to 1 week.
- **Interactive Price Chart**: Smooth Swift Charts area and candlestick line with 80% confidence cone.
- **Predictive Alpha Drivers**: Visual breakdown of top technical indicators driving the forecast.

---

## ⚠️ Disclaimer
*This system is intended for quantitative research, backtesting, and algorithmic modeling. Stock markets are subject to high volatility and risk. Predictions are probabilistic estimates, not guaranteed financial advice.*
