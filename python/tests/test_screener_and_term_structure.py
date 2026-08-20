"""Unit and integration tests for multi-horizon forward prediction term structure and AI market screener."""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from api import app
from stock_predictor.config import PredictionConfig
from stock_predictor.data.synthetic import generate_synthetic_stock_data
from stock_predictor.features.pipeline import FeaturePipeline
from stock_predictor.models.ensemble import EnsembleStockPredictor


@pytest.fixture
def client():
    return TestClient(app)


def test_screener_endpoint_live_structure(client):
    """Tests the concurrent multi-asset screener endpoint."""
    response = client.get("/api/screener?timeframe=1w")
    assert response.status_code == 200
    items = response.json()
    
    # Verify universe size
    assert len(items) >= 10
    
    for item in items:
        assert "ticker" in item
        assert "current_price" in item
        assert "predicted_price" in item
        assert "predicted_return_pct" in item
        assert "change" in item
        assert "change_pct" in item
        assert "signal" in item
        assert "action" in item
        assert "direction_prob" in item
        assert "composite_score" in item
        assert "sharpe_ratio" in item
        assert "risk_reward_ratio" in item
        
        assert item["current_price"] > 0
        assert item["predicted_price"] > 0
        assert 0.0 <= item["direction_prob"] <= 1.0
        assert 0.0 <= item["composite_score"] <= 100.0
        assert item["sharpe_ratio"] > 0.0
        assert item["risk_reward_ratio"] > 0.0

    # Verify descending sort order by expected return
    returns = [it["predicted_return_pct"] for it in items]
    assert returns == sorted(returns, reverse=True)


def test_multi_horizon_term_structure_generation():
    """Tests that the multi-horizon forward curve produces non-crossing diffusion bounds."""
    config = PredictionConfig(forecast_horizon=5, data_interval="1d", timeframe="1w", xgb_n_estimators=20)
    df = generate_synthetic_stock_data(ticker="NVDA", n_days=120, initial_price=130.0, seed=42)
    pipeline = FeaturePipeline(config)
    dataset = pipeline.prepare_dataset(df, horizon=5)
    splits = pipeline.train_val_test_split(dataset)
    
    model = EnsembleStockPredictor(config=config)
    model.fit(splits["train"]["X"], splits["train"]["y_return"], splits["train"]["y_dir"])
    
    forecast = model.generate_forecast(
        ticker="NVDA",
        X_latest=dataset["X_latest"],
        current_price=dataset["latest_price"],
        latest_date=dataset["latest_date"],
        horizon_days=5,
        timeframe="1w",
        minutes_ahead=7200,
        raw_df=df
    )
    
    assert forecast.multi_horizon_path is not None
    assert len(forecast.multi_horizon_path) >= 4
    
    for pt in forecast.multi_horizon_path:
        assert pt.predicted_price > 0
        assert pt.lower_bound_price <= pt.predicted_price, f"Lower bound {pt.lower_bound_price} > predicted {pt.predicted_price}"
        assert pt.predicted_price <= pt.upper_bound_price, f"Predicted {pt.predicted_price} > upper bound {pt.upper_bound_price}"
        assert pt.minutes_ahead > 0
        assert pt.direction in ["BULLISH", "BEARISH", "NEUTRAL"]
