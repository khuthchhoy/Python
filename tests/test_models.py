"""Unit tests for Forecasting Models and Ensembles."""

import pytest
import numpy as np
import pandas as pd

from stock_predictor.config import PredictionConfig
from stock_predictor.data.synthetic import generate_synthetic_stock_data
from stock_predictor.features.pipeline import FeaturePipeline
from stock_predictor.models.gbm_model import GBMStockModel
from stock_predictor.models.lstm_model import LSTMStockModel
from stock_predictor.models.ensemble import EnsembleStockPredictor


@pytest.fixture
def sample_dataset():
    config = PredictionConfig(forecast_horizon=5, lstm_epochs=5)
    target_df = generate_synthetic_stock_data(n_days=300, seed=42)
    pipeline = FeaturePipeline(config)
    dataset = pipeline.prepare_dataset(target_df, horizon=5)
    splits = pipeline.train_val_test_split(dataset)
    return config, dataset, splits


def test_gbm_model(sample_dataset):
    config, dataset, splits = sample_dataset
    train = splits["train"]
    test = splits["test"]
    
    gbm = GBMStockModel(config)
    gbm.fit(train["X"], train["y_return"], train["y_dir"])
    
    preds = gbm.predict_returns(test["X"])
    assert len(preds) == len(test["X"])
    assert not np.isnan(preds).any()
    
    low, high = gbm.predict_intervals(test["X"])
    assert (high >= low).all()
    
    probs = gbm.predict_direction_prob(test["X"])
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    
    imp = gbm.get_feature_importances()
    assert len(imp) > 0


def test_lstm_model(sample_dataset):
    config, dataset, splits = sample_dataset
    train = splits["train"]
    test = splits["test"]
    
    lstm = LSTMStockModel(config)
    lstm.fit(train["X"], train["y_return"], train["y_dir"])
    
    preds = lstm.predict_returns(test["X"])
    assert len(preds) == len(test["X"])
    assert not np.isnan(preds).any()


def test_ensemble_model(sample_dataset):
    config, dataset, splits = sample_dataset
    train = splits["train"]
    
    ensemble = EnsembleStockPredictor(config)
    ensemble.fit(train["X"], train["y_return"], train["y_dir"])
    
    forecast = ensemble.generate_forecast(
        ticker="AAPL",
        X_latest=dataset["X_latest"],
        current_price=dataset["latest_price"],
        latest_date=dataset["latest_date"],
        horizon_days=5
    )
    
    assert forecast.ticker == "AAPL"
    assert forecast.current_price > 0
    assert forecast.predicted_price > 0
    assert forecast.lower_bound_price <= forecast.upper_bound_price
    assert forecast.signal in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    assert 0.0 <= forecast.direction_prob <= 1.0
