"""Unit and integration tests for Intraday Multi-Horizon Forecasting (10m, 20m, 30m, 1h)."""

import pytest
import numpy as np
import pandas as pd

from stock_predictor.config import PredictionConfig, parse_timeframe
from stock_predictor.data.synthetic import generate_synthetic_intraday_data
from stock_predictor.features.pipeline import FeaturePipeline
from stock_predictor.models.ensemble import EnsembleStockPredictor


def test_timeframe_parsing():
    s10m = parse_timeframe("10m")
    assert s10m.data_interval == "5m"
    assert s10m.minutes_ahead == 10
    
    s30m = parse_timeframe("30m")
    assert s30m.data_interval == "5m"
    assert s30m.minutes_ahead == 30
    
    s1h = parse_timeframe("1h")
    assert s1h.data_interval == "5m"
    assert s1h.minutes_ahead == 60
    
    s1w = parse_timeframe("1w")
    assert s1w.data_interval == "1d"
    assert s1w.minutes_ahead == 7200


def test_intraday_synthetic_data():
    df = generate_synthetic_intraday_data(ticker="NVDA", interval="5m", n_bars=200)
    assert len(df) == 200
    assert "Open" in df.columns and "Close" in df.columns
    assert (df["High"] >= df["Low"]).all()
    assert (df["Volume"] > 0).all()


def test_intraday_pipeline_and_features():
    df = generate_synthetic_intraday_data(ticker="AAPL", interval="5m", n_bars=300)
    config = PredictionConfig(forecast_horizon=2, data_interval="5m")
    pipeline = FeaturePipeline(config)
    
    dataset = pipeline.prepare_dataset(df, horizon=2)
    assert "vwap_ratio" in dataset["feature_names"]
    assert "time_of_day_sin" in dataset["feature_names"]
    assert "is_market_open_hour" in dataset["feature_names"]
    
    splits = pipeline.train_val_test_split(dataset)
    assert len(splits["train"]["X"]) > 50
    assert len(splits["test"]["X"]) > 10


def test_intraday_ensemble_forecast():
    df = generate_synthetic_intraday_data(ticker="TSLA", interval="5m", n_bars=300)
    config = PredictionConfig(forecast_horizon=6, data_interval="5m", lstm_epochs=3)
    pipeline = FeaturePipeline(config)
    dataset = pipeline.prepare_dataset(df, horizon=6)
    splits = pipeline.train_val_test_split(dataset)
    
    ensemble = EnsembleStockPredictor(config)
    ensemble.fit(splits["train"]["X"], splits["train"]["y_return"])
    
    forecast = ensemble.generate_forecast(
        ticker="TSLA",
        X_latest=dataset["X_latest"],
        current_price=dataset["latest_price"],
        latest_date=dataset["latest_date"],
        horizon_days=1,
        timeframe="30m",
        minutes_ahead=30,
        is_synthetic=True
    )
    
    assert forecast.ticker == "TSLA"
    assert forecast.timeframe == "30m"
    assert forecast.multi_horizon_path is not None
    assert len(forecast.multi_horizon_path) >= 4
    assert forecast.multi_horizon_path[0].timeframe == "10m"
