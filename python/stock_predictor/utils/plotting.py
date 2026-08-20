"""Interactive Plotly Visualization suite for stock forecasts and backtest metrics."""

from typing import Dict, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock_predictor.models.base import ForecastResult


def create_forecast_chart(
    full_df: pd.DataFrame,
    forecast: ForecastResult,
    lookback_days: int = 90
) -> go.Figure:
    """
    Creates an interactive financial chart showing:
    1. Candlestick OHLC historical prices
    2. 20-day and 50-day Simple Moving Averages
    3. 1-Week Ahead Forecast point and 10%-90% Confidence Cone
    4. Volume subchart
    """
    df_recent = full_df.iloc[-lookback_days:].copy()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        specs=[[{"type": "xy"}], [{"type": "xy"}]]
    )
    
    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_recent.index,
            open=df_recent["Open"],
            high=df_recent["High"],
            low=df_recent["Low"],
            close=df_recent["Close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )
    
    # 2. Moving Averages
    sma20 = df_recent["Close"].rolling(20).mean()
    sma50 = df_recent["Close"].rolling(50).mean()
    
    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=sma20,
            line=dict(color="#29b6f6", width=1.5),
            name="SMA 20"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_recent.index,
            y=sma50,
            line=dict(color="#ffa726", width=1.5),
            name="SMA 50"
        ),
        row=1, col=1
    )
    
    # 3. Forecast Cone (Projection into t+5)
    last_date = df_recent.index[-1]
    target_dt = pd.to_datetime(forecast.target_date)
    
    forecast_x = [last_date, target_dt]
    upper_y = [forecast.current_price, forecast.upper_bound_price]
    lower_y = [forecast.current_price, forecast.lower_bound_price]
    pred_y = [forecast.current_price, forecast.predicted_price]
    
    # Upper boundary
    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=upper_y,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="Upper Bound (90%)"
        ),
        row=1, col=1
    )
    
    # Lower boundary with fill
    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=lower_y,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(124, 77, 255, 0.22)",
            name="80% Confidence Cone"
        ),
        row=1, col=1
    )
    
    # Forecast median line
    forecast_color = "#00e676" if forecast.predicted_return_pct >= 0 else "#ff5252"
    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=pred_y,
            mode="lines+markers",
            line=dict(color=forecast_color, width=3, dash="dash"),
            marker=dict(size=8, color=forecast_color),
            name=f"1-Wk Target: ${forecast.predicted_price:.2f} ({forecast.predicted_return_pct:+.2f}%)"
        ),
        row=1, col=1
    )
    
    # 4. Volume Bar Chart
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df_recent["Close"], df_recent["Open"])]
    fig.add_trace(
        go.Bar(
            x=df_recent.index,
            y=df_recent["Volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # Layout adjustments
    fig.update_layout(
        template="plotly_dark",
        title=f"<b>{forecast.ticker} — 1-Week Ahead Forecast & Technical Structure</b>",
        title_font_size=18,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=620
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def create_feature_importance_chart(importances: Dict[str, float], top_n: int = 15) -> go.Figure:
    """Create a horizontal bar chart of top predictive features."""
    sorted_items = list(importances.items())[:top_n]
    if not sorted_items:
        fig = go.Figure()
        fig.add_annotation(text="Feature importances not available", showarrow=False)
        return fig
        
    feats, scores = zip(*reversed(sorted_items))
    
    fig = go.Figure(
        go.Bar(
            x=[s * 100 for s in scores],
            y=feats,
            orientation="h",
            marker=dict(
                color=[s for s in scores],
                colorscale="Viridis",
                showscale=False
            )
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="<b>Top Predictive Alpha Drivers</b> (Importance %)",
        xaxis_title="Relative Importance (%)",
        yaxis_title="Feature",
        margin=dict(l=140, r=40, t=50, b=40),
        height=450
    )
    return fig


def create_backtest_chart(results_df: pd.DataFrame) -> go.Figure:
    """Create equity growth chart comparing AI strategy vs Benchmark."""
    fig = go.Figure()
    
    # Normalize initial equity to $10,000 for intuitive interpretation
    strat_wealth = results_df["Strategy_Equity"] * 10000.0
    bench_wealth = results_df["Benchmark_Equity"] * 10000.0
    
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=strat_wealth,
            mode="lines",
            line=dict(color="#00e676", width=2.5),
            name="AI Model Strategy"
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=bench_wealth,
            mode="lines",
            line=dict(color="#78909c", width=1.5, dash="dot"),
            name="Buy & Hold Benchmark"
        )
    )
    
    fig.update_layout(
        template="plotly_dark",
        title="<b>Out-of-Sample Walk-Forward Equity Growth ($10,000 Initial)</b>",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420
    )
    return fig
