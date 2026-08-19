"""Unit tests for Evaluation & Backtesting."""

import pytest
import numpy as np
import pandas as pd

from stock_predictor.config import PredictionConfig
from stock_predictor.data.synthetic import generate_synthetic_stock_data
from stock_predictor.features.pipeline import FeaturePipeline
from stock_predictor.models.ensemble import EnsembleStockPredictor
from stock_predictor.evaluation.backtest import WalkForwardBacktester
from stock_predictor.evaluation.metrics import calculate_forecast_metrics, calculate_strategy_performance


def test_metrics_calculation():
    true_prices = np.array([100.0, 105.0, 102.0, 110.0])
    pred_prices = np.array([101.0, 104.0, 103.0, 109.0])
    true_rets = np.array([0.01, 0.05, -0.02, 0.08])
    pred_rets = np.array([0.02, 0.04, -0.01, 0.07])
    
    metrics = calculate_forecast_metrics(
        true_prices, pred_prices, true_rets, pred_rets
    )
    assert metrics["mae"] > 0
    assert metrics["directional_accuracy"] == 100.0  # all signs matched


def test_backtest_runner():
    config = PredictionConfig(forecast_horizon=5, lstm_epochs=3)
    target_df = generate_synthetic_stock_data(n_days=300, seed=42)
    pipeline = FeaturePipeline(config)
    dataset = pipeline.prepare_dataset(target_df, horizon=5)
    splits = pipeline.train_val_test_split(dataset)
    
    ensemble = EnsembleStockPredictor(config)
    ensemble.fit(splits["train"]["X"], splits["train"]["y_return"])
    
    backtester = WalkForwardBacktester(config)
    df_results, summary = backtester.evaluate_test_set(ensemble, splits["test"])
    
    assert len(df_results) == len(splits["test"]["X"])
    assert "Strategy_Equity" in df_results.columns
    assert summary.total_samples == len(splits["test"]["X"])
    assert 0.0 <= summary.directional_accuracy <= 100.0
