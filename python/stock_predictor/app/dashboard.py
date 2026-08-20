"""Interactive Streamlit Financial Dashboard for Multi-Horizon Stock Price Forecasting, Autonomous AI Analyst, and Self-Learning Hub."""

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
from stock_predictor.learning.engine import get_global_learning_engine
from stock_predictor.utils.plotting import (
    create_forecast_chart,
    create_feature_importance_chart,
    create_backtest_chart
)

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor & Autonomous Analyst | Live System",
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
    
    .analyst-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .trade-plan-card {
        background: #1e222d;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
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

    # Evaluate pending forecasts against fresh data
    learning_engine = get_global_learning_engine()
    learning_engine.evaluate_realizations(ticker, latest_df=target_df)
    
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
    
    # Generate Forecast with Full Analytics Pipeline
    forecast = ensemble.generate_forecast(
        ticker=ticker,
        X_latest=dataset["X_latest"],
        current_price=dataset["latest_price"],
        latest_date=dataset["latest_date"],
        horizon_days=max(1, spec.minutes_ahead // 1440),
        timeframe=spec.timeframe_id,
        minutes_ahead=spec.minutes_ahead,
        is_synthetic=is_synthetic,
        raw_df=target_df,
        benchmarks=benchmarks
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
    tf_label = st.sidebar.selectbox("Forecast Horizon", list(timeframe_options.keys()), index=6)
    timeframe_selected = timeframe_options[tf_label]
    
    live_streaming = st.sidebar.toggle("🔴 Enable Real-Time Live Streaming", value=False)
    
    start_date = st.sidebar.date_input("Training History Start", value=pd.to_datetime("2020-01-01"))
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    force_synthetic = st.sidebar.checkbox("Force Offline Synthetic Generator", value=False)
    
    st.sidebar.subheader("Ensemble Model Weights")
    gbm_weight = st.sidebar.slider("Gradient Boosted Trees Weight", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    lstm_weight = round(1.0 - gbm_weight, 2)
    st.sidebar.caption(f"PyTorch Temporal Attention Weight: **{lstm_weight:.2f}**")
    
    run_btn = st.sidebar.button("🚀 Run Forecast & Analysis", type="primary", use_container_width=True)

    # Header
    st.title(f"📈 AI Stock Price Predictor & Autonomous Analyst ({ticker_input})")
    
    if live_streaming:
        st.caption("🔴 **LIVE STREAMING ACTIVE** • Auto-refreshing price, self-learning weights, and trade execution plan in real time.")
    else:
        st.caption("Next-Generation Quantitative Machine Learning Engine with Continuous Self-Learning, Dynamic Trade Planning, and Autonomous AI Analyst Synthesis.")
    
    with st.spinner(f"Ingesting market data & executing AI multi-model analysis for {ticker_input} ({timeframe_selected})..."):
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
            <div class="metric-title">Action & Win Prob</div>
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
    tab_analyst, tab_chart, tab_tradeplan, tab_learning, tab_backtest, tab_explain = st.tabs([
        "🤖 AI Self-Analyst & Thesis",
        "📊 Interactive Forecast Chart",
        "🎯 Trade Plan & Risk Levels",
        "🧠 Continuous Self-Learning Hub",
        "🧪 Walk-Forward Backtest",
        "🔍 Alpha Drivers & Features"
    ])
    
    # TAB 1: AI Self-Analyst
    with tab_analyst:
        if forecast.analyst_report:
            rep = forecast.analyst_report
            st.markdown(f"""
            <div class="analyst-box">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h2 style="margin: 0; color: #38bdf8;">🤖 Institutional AI Analyst Report: {ticker_input}</h2>
                    <span style="background: #3b82f6; color: white; padding: 4px 12px; border-radius: 16px; font-weight: bold; font-size: 0.9rem;">
                        VERDICT: {rep.verdict} ({rep.conviction_score:.1f}/100 Conviction)
                    </span>
                </div>
                <hr style="border-color: #334155; margin: 16px 0;">
                <p style="font-size: 1.1rem; line-height: 1.6; color: #f1f5f9;">{rep.executive_summary}</p>
            </div>
            """, unsafe_allow_html=True)

            col_cat, col_risk = st.columns(2)
            with col_cat:
                st.subheader("💡 Primary Quantitative Catalysts & Drivers")
                for cat in rep.primary_catalysts:
                    st.markdown(f"- {cat}")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("🌐 Macro & Market Regime")
                st.info(rep.macro_regime_analysis)

            with col_risk:
                st.subheader("⚠️ Autonomous Contrarian Self-Critique & Risks")
                for r in rep.contrarian_risks:
                    st.markdown(f"- {r}")

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📐 Structural Price Geometry")
                st.caption(rep.key_levels_summary)

            st.markdown("---")
            st.subheader("📊 Multi-Factor Quantitative Score Breakdown")
            if forecast.factor_scores:
                fs = forecast.factor_scores
                f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
                f_col1.metric("Momentum Score", f"{fs.momentum_score:.1f}/100")
                f_col2.metric("Trend Quality", f"{fs.trend_score:.1f}/100")
                f_col3.metric("Volatility Stability", f"{fs.volatility_score:.1f}/100")
                f_col4.metric("Money Flow & Liquidity", f"{fs.flow_score:.1f}/100")
                f_col5.metric("Composite Alpha", f"{fs.composite_score:.1f}/100")

    # TAB 2: Interactive Forecast Chart
    with tab_chart:
        fig_forecast = create_forecast_chart(target_df, forecast, lookback_days=120)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.info(
            f"💡 **Forecast Interpretation**: The model projects **{ticker_input}** to move from **${forecast.current_price:.2f}** to **${forecast.predicted_price:.2f}** ({forecast.predicted_return_pct:+.2f}%) over the next {spec.human_label} with an 80% statistical confidence interval of **[${forecast.lower_bound_price:.2f}, ${forecast.upper_bound_price:.2f}]**."
        )

    # TAB 3: Trade Plan & Risk Levels
    with tab_tradeplan:
        if forecast.trade_plan:
            tp = forecast.trade_plan
            st.subheader("🎯 Institutional Algorithmic Trade Execution Plan")
            st.markdown(f"**Strategy**: {tp.execution_strategy}")
            
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            p_col1.markdown(f"""
            <div class="trade-plan-card">
                <div class="metric-title">RECOMMENDED ACTION</div>
                <div style="font-size: 1.4rem; font-weight: bold; color: #38bdf8; margin-top: 4px;">{tp.action}</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Entry Zone: ${tp.entry_zone_low:.2f} – ${tp.entry_zone_high:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            p_col2.markdown(f"""
            <div class="trade-plan-card">
                <div class="metric-title">DYNAMIC STOP LOSS</div>
                <div style="font-size: 1.4rem; font-weight: bold; color: #ef4444; margin-top: 4px;">${tp.stop_loss:.2f}</div>
                <div style="font-size: 0.85rem; color: #ef4444; margin-top: 4px;">Downside Risk: {tp.stop_loss_pct:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            p_col3.markdown(f"""
            <div class="trade-plan-card">
                <div class="metric-title">TARGETS (TP1 / TP2)</div>
                <div style="font-size: 1.4rem; font-weight: bold; color: #22c55e; margin-top: 4px;">${tp.target_1:.2f} / ${tp.target_2:.2f}</div>
                <div style="font-size: 0.85rem; color: #22c55e; margin-top: 4px;">Upside: {tp.target_1_return_pct:+.1f}% / {tp.target_2_return_pct:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            p_col4.markdown(f"""
            <div class="trade-plan-card">
                <div class="metric-title">RISK/REWARD & KELLY</div>
                <div style="font-size: 1.4rem; font-weight: bold; color: #a855f7; margin-top: 4px;">1 : {tp.risk_reward_ratio:.2f}</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Kelly Size: <b>{tp.kelly_size_pct:.1f}%</b> | 95% VaR: -{tp.var_95_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        if forecast.support_resistance:
            sr = forecast.support_resistance
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📐 Dynamic Support & Resistance / Fibonacci Retracements")
            s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)
            s_col1.metric("Support 1 (Immediate)", f"${sr.support_1:.2f}")
            s_col2.metric("Support 2 (Deep)", f"${sr.support_2:.2f}")
            s_col3.metric("Pivot Point", f"${sr.pivot_point:.2f}")
            s_col4.metric("Resistance 1", f"${sr.resistance_1:.2f}")
            s_col5.metric("Breakout Trigger", f"${sr.breakout_level:.2f}")

    # TAB 4: Continuous Self-Learning Hub
    with tab_learning:
        st.subheader("🧠 Continuous Self-Learning & Calibration Hub")
        st.caption("The AI continuously journals live predictions, compares them to ground truth when horizons elapse, and dynamically adapts model weights via Multi-Armed Bandit optimization.")
        
        if forecast.learning_metrics:
            lm = forecast.learning_metrics
            l_col1, l_col2, l_col3, l_col4 = st.columns(4)
            l_col1.metric("Directional Hit Rate", f"{lm.directional_accuracy_pct:.1f}%")
            l_col2.metric("80% Interval Coverage", f"{lm.interval_coverage_pct:.1f}%")
            l_col3.metric("Mean Error (MAPE)", f"{lm.mape_pct:.2f}%")
            l_col4.metric("Active Model Weights", f"GBM: {lm.active_gbm_weight*100:.0f}% | LSTM: {lm.active_lstm_weight*100:.0f}%")
            
            st.info(f"📊 **Self-Learning Track Record**: {forecast.analyst_report.model_track_record_summary if forecast.analyst_report else 'Actively learning from realized market returns.'}")
            
            if lm.recent_records:
                st.subheader("📋 Recent Verified Prediction Log")
                st.dataframe(pd.DataFrame(lm.recent_records), use_container_width=True)

    # TAB 5: Backtest
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

    # TAB 6: Alpha Drivers & Features
    with tab_explain:
        st.subheader("Predictive Alpha Drivers & Feature Importance")
        st.caption("Shows relative contribution of each technical, momentum, volatility, and market regime feature.")
        
        if forecast.feature_importances:
            fig_imp = create_feature_importance_chart(forecast.feature_importances, top_n=15)
            st.plotly_chart(fig_imp, use_container_width=True)

    # Live auto-refresh loop
    if live_streaming:
        time.sleep(3.0)
        st.rerun()


if __name__ == "__main__":
    render_dashboard()
