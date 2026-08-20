# 📈 Real-World AI Stock Price Prediction & Autonomous Quantitative Analyst System

An end-to-end institutional-grade financial machine learning system engineered to predict stock prices across multiple horizons (**10m, 20m, 30m, 1h, 4h, 1d, 1w**), equipped with **Continuous Self-Learning & Adaptive Calibration**, **Autonomous AI Stock Analyst Thesis Synthesis**, **Algorithmic Trade Execution Planning**, and multi-factor quantitative modeling with a FastAPI REST/WebSocket backend and native SwiftUI iOS app.

---

## 🌟 Key Architecture & Capabilities

### 1. 🤖 Autonomous Quantitative AI Stock Analyst
- **Wall-Street Grade Narrative Synthesis**: Synthesizes multi-model ML outputs, macro beta, support/resistance geometry, and factor scores into an executive thesis.
- **Actionable Conviction & Verdict**: Classifies stance (`HIGH-CONVICTION ACCUMULATION`, `TACTICAL LONG BIAS`, `CONSOLIDATION WATCH`, `DISTRIBUTION ALERT`) with 0–100 conviction score.
- **Key Quantitative Catalysts**: Identifies 4–6 concrete numeric catalysts driving price movement.
- **Autonomous Contrarian Self-Critique**: Automatically formulates downside tail-risks, invalidation triggers, and volatility squeeze vulnerabilities.

### 2. 🧠 Continuous Self-Learning & Adaptive Feedback Meta-Layer
- **Persistent Prediction Journal**: Automatically records all live predictions with target execution timestamps, bounds, and active model weights.
- **Ground Truth Feedback Evaluator**: Scans pending predictions when target timestamps elapse and verifies directional hit-rate, percentage error (MAPE), and interval coverage against realized prices.
- **Adaptive Multi-Armed Bandit / Hedge Weighting**: Dynamically rebalances ensemble weights between Gradient Boosted Trees and PyTorch Sequence Attention based on empirical exponential-decay hit-rate per ticker.
- **Online Residual Drift Correction**: Estimates and adjusts for systematic directional drift ($\Delta \mu_t$).

### 3. 🎯 Algorithmic Trade Execution Planner
- **Actionable Setup Formulation**: Recommends `ACCUMULATE`, `BUY LIMIT`, `BREAKOUT BUY`, `HOLD`, or `SCALE OUT / SHORT`.
- **Volatility-Adjusted Dynamic Stop Loss**: Computes dynamic stop loss based on $1.5\times ATR$ and primary support/resistance buffers.
- **Multi-Stage Take-Profit Targets ($TP_1, TP_2$)**: Statistical upside targets mapped to resistance and Fibonacci extensions.
- **Risk-Reward Ratio ($RRR$) & Probabilistic VaR**: Computes exact reward per unit risk and 95% / 99% Value at Risk.
- **Kelly Criterion Portfolio Sizing**: Calculates optimal conservative half-Kelly portfolio fraction allocation.

### 4. 📊 Multi-Factor Quantitative Scoring & Structural Geometry
- **Multi-Factor Scoring (0–100)**: Momentum Score, Trend Quality Score, Volatility Stability Score, and Institutional Money Flow (CMF/OBV).
- **Dynamic Support & Resistance (KDE)**: Kernel Density Estimation clustering of volume nodes, Pivot Points ($S_1, S_2, Pivot, R_1, R_2$), and Fibonacci Retracements (23.6%, 38.2%, 50%, 61.8%).
- **Technical & Candlestick Pattern Recognition**: RSI/MACD divergences, Bollinger Band squeezes, EMA 20/50 crosses, pin bars, engulfing candles, and volume surges.

### 5. ⚡ Direct Live Market Data & Multi-Model Meta-Ensemble
- **Ultra-Low Latency Yahoo Finance v8/v10 Chart Engine**: Direct JSON chart streaming with parquet cache and calibrated stochastic fallback.
- **Stationary Forward Log Returns**: $r_{t+h} = \ln(P_{t+h} / P_t)$ with monotonic quantile regression bounds.
- **Gradient Boosted Trees + PyTorch Temporal Sequence Attention**: Non-linear tabular interaction modeling with dynamic historical sequence weighting.

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
- `GET /api/health` — API health, self-learning status, and model engine status.
- `GET /api/forecast?ticker=NVDA&timeframe=1w` — Complete forecast with trade plan, support/resistance, factor scores, regime, and analyst report.
- `GET /api/analyst?ticker=NVDA&timeframe=1w` — Autonomous Wall-Street style AI stock analyst synthesis report.
- `GET /api/learning?ticker=NVDA` — Real-time self-learning track record, rolling empirical accuracy, and adaptive weights.
- `POST /api/learning/evaluate?ticker=NVDA` — Trigger ground truth evaluation against recent market prices.
- `GET /api/quote?ticker=AAPL` — Fast real-time quote (price, day change, volume, high, low).
- `GET /api/watchlist` — Real-time multi-stock watchlist scan with action recommendations and composite scores.
- `GET /api/screener` — Market opportunity screener ranking top liquidity leaders.
- `GET /api/backtest?ticker=NVDA` — Out-of-sample mark-to-market walk-forward backtest equity curve.
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
- **🤖 Autonomous AI Stock Analyst**: Executive thesis, conviction score, primary catalysts, and contrarian risks.
- **🎯 Institutional Trade Execution Plan**: Action badge, entry zone, dynamic stop-loss, target 1 & 2, risk/reward ratio, and Kelly allocation.
- **🧠 Continuous Self-Learning Telemetry**: Real-time hit rate %, total verified predictions, and dynamically adapted model weights.
- **📐 Support & Resistance Geometry**: Dynamic $S_1, S_2, Pivot, R_1, Breakout$ levels.
- **📊 Quantitative Factor Meters**: Visual bars for Momentum, Trend Quality, Volatility Stability, and Money Flow.
- **Live Stream Status**: Auto-refreshes every 5 seconds or on-demand.
- **Configurable Backend Host**: Switch between local simulator and remote cloud server.
- **Watchlist & Signals Strip**: 1-tap switching across popular tickers (NVDA, AAPL, MSFT, TSLA, AMZN, META, SPY, etc.).
- **Multi-Horizon Trajectory**: Term structure projection cards from 10 minutes to 1 week.

---

## ⚠️ Disclaimer
*This system is intended for quantitative research, backtesting, and algorithmic modeling. Stock markets are subject to high volatility and risk. Predictions are probabilistic estimates, not guaranteed financial advice.*
