"""Unit tests for Feature Engineering components."""

import pytest
import numpy as np
import pandas as pd

from stock_predictor.data.synthetic import generate_synthetic_stock_data
from stock_predictor.features.technical import calculate_technical_features
from stock_predictor.features.market_context import calculate_market_context_features
from stock_predictor.features.calendar import calculate_calendar_features


def test_synthetic_data_generator():
    df = generate_synthetic_stock_data(ticker="TEST", n_days=300, seed=42)
    assert len(df) == 300
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns)
    assert (df["High"] >= df["Low"]).all()
    assert (df["Volume"] > 0).all()
    assert not df.isna().any().any()


def test_technical_features_calculation():
    df = generate_synthetic_stock_data(n_days=300, seed=42)
    tech = calculate_technical_features(df)
    
    # Check key indicator columns exist
    expected_cols = [
        "return_1d", "return_5d", "sma_20", "sma_50", "ema_21",
        "rsi_14", "rsi_7", "rsi_21", "stoch_rsi",
        "macd_line_norm", "macd_signal_norm", "bb_pct_b",
        "natr_14", "stoch_k", "roc_5", "volume_ratio_20", "zscore_20d",
        "garman_klass_vol_20", "parkinson_vol_20", "cmf_20",
        "overnight_gap", "intraday_return",
        "supertrend_dir", "supertrend_dist",
        "ttm_squeeze_on", "ttm_squeeze_momentum",
        "ema_ribbon_alignment", "ema_ribbon_spread",
        "donchian_20_pos", "donchian_breakout_20",
        "vpt_norm", "rvol_20", "hma_14_ratio"
    ]
    for col in expected_cols:
        assert col in tech.columns, f"Missing feature: {col}"
        
    # Check RSI range [0, 1]
    valid_rsi = tech["rsi_14"].dropna()
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 1.0).all()


def test_market_context_features():
    target_df = generate_synthetic_stock_data(ticker="AAPL", n_days=300, seed=1)
    spy_df = generate_synthetic_stock_data(ticker="SPY", n_days=300, seed=2)
    vix_df = generate_synthetic_stock_data(ticker="^VIX", n_days=300, seed=3)
    
    benchmarks = {"SPY": spy_df, "^VIX": vix_df}
    mkt_feats = calculate_market_context_features(target_df, benchmarks)
    
    assert "spy_ret_5d" in mkt_feats.columns
    assert "rel_strength_5d" in mkt_feats.columns
    assert "vix_level" in mkt_feats.columns
    assert "beta_spy_60d" in mkt_feats.columns
    assert "spy_trend_regime" in mkt_feats.columns


def test_calendar_features():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    cal_feats = calculate_calendar_features(dates)
    
    assert "day_of_week_sin" in cal_feats.columns
    assert "month_sin" in cal_feats.columns
    assert len(cal_feats) == 100
    assert not cal_feats.isna().any().any()
