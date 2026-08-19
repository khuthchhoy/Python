"""Interactive Streamlit Financial Dashboard for Multi-Horizon Stock Price Forecasting with Live Stream Progress."""

import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np

from stock_predictor.config import PredictionConfig, parse_timeframe
from stock_predictor.data.downloader import StockDataDownloader
from stock_predictor.features.pipeline import FeaturePipeline
from stock_predictor.models.ensemble import EnsembleStockPredictor
from stock_predictor.evaluation.backtest import WalkForwardBacktester
from stock_predictor.utils.plotting import (
    create_forecast_chart,
    create_feature_importance_chart,
    create_backtest_chart
)

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor | Live Multi-Horizon Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for sleek dark terminal theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: #1e222d;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #2e3546;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .signal-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        text-align: center;
    }
    .signal-strong-buy { background-color: #00c853; color: #000000; }
    .signal-buy { background-color: #69f0ae; color: #000000; }
    .signal-hold { background-color: #ffd600; color: #000000; }
    .signal-sell { background-color: #ff5252; color: #ffffff; }
    .signal-strong-sell { background-color: #d50000; color: #ffffff; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def load_and_train(ticker: str, timeframe: str, start_date: str, gbm_w: float, lstm_w: float, force_syn: bool = False):
    spec = parse_timeframe(timeframe)
    config = PredictionConfig(
        forecast_horizon=spec.horizon_bars,
        data_interval=spec.data_interval,
        timeframe=spec.timeframe_id,
        random_state=42
    )
    
    downloader = StockDataDownloader(config)
    target_df, benchmarks = downloader.fetch_market_dataset(
        target_ticker=ticker,
        start_date=start_date if spec.timeframe_id in ["1d", "1w", "5d"] else None,
        interval=spec.data_interval,
        period=spec.default_period,
        use_cache=True,
        force_synthetic=force_syn
    )
    is_synthetic = downloader.last_was_synthetic
    
    pipeline = FeaturePipeline(config)
    dataset = pipeline.prepare_dataset(target_df, benchmarks, horizon=spec.horizon_bars)
    splits = pipeline.train_val_test_split(dataset)
    
    # Train Ensemble
    ensemble = EnsembleStockPredictor(
        config=config,
        gbm_weight=gbm_w,
        lstm_weight=lstm_w
    )
    
    train_data = splits["train"]
    ensemble.fit(
        X=train_data["X"],
        y_return=train_data["y_return"],
        y_dir=train_data["y_dir"]
    )
    
    # Generate Forecast
    forecast = ensemble.generate_forecast(
        ticker=ticker,
        X_latest=dataset["X_latest"],
        current_price=dataset["latest_price"],
        latest_date=dataset["latest_date"],
        horizon_days=max(1, spec.minutes_ahead // 1440),
        timeframe=spec.timeframe_id,
        minutes_ahead=spec.minutes_ahead,
        is_synthetic=is_synthetic
    )
    
    # Run Backtest
    backtester = WalkForwardBacktester(config)
    backtest_df, backtest_metrics = backtester.evaluate_test_set(
        model=ensemble,
        test_data=splits["test"]
    )
    
    return target_df, dataset, forecast, backtest_df, backtest_metrics, spec


def render_dashboard():
    # Sidebar
    st.sidebar.title("⚙️ Model & Live Stream")
    
    popular_tickers = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "SPY"]
    selected_option = st.sidebar.selectbox("Select Benchmark Asset", popular_tickers + ["Custom..."])
    
    if selected_option == "Custom...":
        ticker_input = st.sidebar.text_input("Enter Ticker Symbol", value="AMD").upper().strip()
    else:
        ticker_input = selected_option
        
    timeframe_options = {
        "10 Minutes (Intraday)": "10m",
        "20 Minutes (Intraday)": "20m",
        "30 Minutes (Intraday)": "30m",
        "1 Hour (Intraday)": "1h",
        "4 Hours (Intraday)": "4h",
        "1 Day (Short-Term)": "1d",
        "1 Week / 5 Days (Swing)": "1w"
    }
    tf_label = st.sidebar.selectbox("Forecast Horizon", list(timeframe_options.keys()), index=0)
    timeframe_selected = timeframe_options[tf_label]
    
    live_streaming = st.sidebar.toggle("🔴 Enable Real-Time Live Streaming", value=False)
    
    start_date = st.sidebar.date_input("Training History Start", value=pd.to_datetime("2020-01-01"))
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    force_synthetic = st.sidebar.checkbox("Force Offline Synthetic Generator", value=False)
    
    st.sidebar.subheader("Ensemble Model Weights")
    gbm_weight = st.sidebar.slider("Gradient Boosted Trees Weight", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    lstm_weight = round(1.0 - gbm_weight, 2)
    st.sidebar.caption(f"PyTorch Temporal Attention Weight: **{lstm_weight:.2f}**")
    
    run_btn = st.sidebar.button("🚀 Run Forecast & Backtest", type="primary", use_container_width=True)

    # Header
    st.title(f"📈 AI Stock Price Predictor — {tf_label.split('(')[0].strip()} ({ticker_input})")
    
    if live_streaming:
        st.caption("🔴 **LIVE STREAMING ACTIVE** • Auto-refreshing price and multi-horizon trajectory in real time.")
    else:
        st.caption("Quantitative machine learning engine utilizing Quantile Boosted Trees, PyTorch Temporal Attention LSTM, and cross-asset market regime features.")
    
    with st.spinner(f"Ingesting market data & training ensemble models for {ticker_input} ({timeframe_selected})..."):
        try:
            target_df, dataset, forecast, backtest_df, backtest_metrics, spec = load_and_train(
                ticker=ticker_input,
                timeframe=timeframe_selected,
                start_date=start_date_str,
                gbm_w=gbm_weight,
                lstm_w=lstm_weight,
                force_syn=force_synthetic
            )
        except Exception as e:
            st.error(f"Error executing prediction engine: {e}")
            return

    if forecast.is_synthetic:
        st.warning("⚠️ **Simulated Data Mode**: Live feed was unavailable or synthetic mode was selected. Calculations reflect Monte Carlo Geometric Brownian Motion.")

    # Key Metrics Cards Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Live Price ({forecast.forecast_date})</div>
            <div class="metric-value" style="color: #ffffff;">${forecast.current_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        delta_color = "#00e676" if forecast.predicted_return_pct >= 0 else "#ff5252"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{spec.human_label} Target ({forecast.target_date})</div>
            <div class="metric-value" style="color: {delta_color};">
                ${forecast.predicted_price:.2f} 
                <span style="font-size: 1rem;">({forecast.predicted_return_pct:+.2f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">80% Confidence Interval</div>
            <div class="metric-value" style="color: #b388ff; font-size: 1.45rem;">
                ${forecast.lower_bound_price:.2f} – ${forecast.upper_bound_price:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        badge_cls = {
            "STRONG_BUY": "signal-strong-buy",
            "BUY": "signal-buy",
            "HOLD": "signal-hold",
            "SELL": "signal-sell",
            "STRONG_SELL": "signal-strong-sell"
        }.get(forecast.signal, "signal-hold")
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Trading Signal & Win Prob</div>
            <div style="margin-top: 6px;">
                <span class="signal-badge {badge_cls}">{forecast.signal.replace('_', ' ')}</span>
                <span style="margin-left: 10px; color: #8b949e; font-size: 0.9rem;">
                    P(Up): <b>{forecast.direction_prob*100:.1f}%</b>
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Multi-Horizon Trajectory Table
    if forecast.multi_horizon_path:
        st.subheader("📈 Multi-Horizon Term Structure Trajectory Path")
        mh_data = [
            {
                "Horizon": hp.timeframe.upper(),
                "Target Time": hp.target_time,
                "Predicted Price": f"${hp.predicted_price:.2f}",
                "Expected Return": f"{hp.predicted_return_pct:+.2f}%",
                "80% Confidence Cone": f"${hp.lower_bound_price:.2f} – ${hp.upper_bound_price:.2f}",
                "Bias": hp.direction
            }
            for hp in forecast.multi_horizon_path
        ]
        st.dataframe(pd.DataFrame(mh_data), use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Main Tabs
    tab_chart, tab_backtest, tab_explain, tab_details = st.tabs([
        "📊 Interactive Forecast Chart",
        "🧪 Walk-Forward Backtest",
        "🧠 Model Explainability",
        "📋 Forecast Data & Metrics"
    ])
    
    with tab_chart:
        fig_forecast = create_forecast_chart(target_df, forecast, lookback_days=120)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.info(
            f"💡 **Forecast Interpretation**: The model projects **{ticker_input}** to move from **${forecast.current_price:.2f}** to **${forecast.predicted_price:.2f}** ({forecast.predicted_return_pct:+.2f}%) over the next {spec.human_label} with an 80% statistical confidence interval of **[${forecast.lower_bound_price:.2f}, ${forecast.upper_bound_price:.2f}]**."
        )

    with tab_backtest:
        st.subheader("Out-of-Sample Walk-Forward Backtesting Performance")
        st.caption("Evaluated strictly on out-of-sample data with daily mark-to-market returns and purged & embargoed splits.")
        
        b_col1, b_col2, b_col3, b_col4, b_col5, b_col6 = st.columns(6)
        b_col1.metric("Directional Hit Rate", f"{backtest_metrics.directional_accuracy:.1f}%")
        b_col2.metric("Interval Coverage (80%)", f"{backtest_metrics.interval_coverage:.1f}%")
        b_col3.metric("Strategy Total Return", f"{backtest_metrics.strategy_return_pct:+.1f}%")
        b_col4.metric("Annualized Sharpe", f"{backtest_metrics.sharpe_ratio:.2f}")
        b_col5.metric("Sortino Ratio", f"{backtest_metrics.sortino_ratio:.2f}")
        b_col6.metric("Max Drawdown", f"{backtest_metrics.max_drawdown_pct:.1f}%")
        
        fig_backtest = create_backtest_chart(backtest_df)
        st.plotly_chart(fig_backtest, use_container_width=True)

    with tab_explain:
        st.subheader("Predictive Alpha Drivers & Feature Importance")
        st.caption("Shows relative contribution of each technical, momentum, volatility, and market regime feature.")
        
        if forecast.feature_importances:
            fig_imp = create_feature_importance_chart(forecast.feature_importances, top_n=15)
            st.plotly_chart(fig_imp, use_container_width=True)

    with tab_details:
        st.subheader("Model Specification & Quantitative Formulation")
        st.markdown(r"""
        - **Intraday & Multi-Horizon Modeling**: Supports high-frequency intervals (1m, 5m, 15m, 1h) and horizons from **10 minutes to 1 week**.
        - **Target Formulation**: Models forward log-returns:
          $$r_{t+h} = \ln\left(\frac{P_{t+h}}{P_t}\right)$$
          $$\hat{P}_{t+h} = P_t \cdot \exp(\hat{r}_{t+h})$$
        - **Ensemble Architecture**: Blends non-linear Gradient Boosted Trees (XGBoost/sklearn) and Deep Temporal Sequences (PyTorch GRU with Additive Temporal Attention).
        """)

    # Live auto-refresh loop
    if live_streaming:
        time.sleep(3.0)
        st.rerun()


if __name__ == "__main__":
    render_dashboard()
