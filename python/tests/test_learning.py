"""Unit tests for Self-Learning Engine (Journaling, Realization Evaluation, Adaptive Weighting)."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from stock_predictor.learning.engine import SelfLearningEngine, PredictionRecord


@pytest.fixture
def temp_learning_engine():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        path = Path(tf.name)
    engine = SelfLearningEngine(storage_path=path)
    yield engine
    if path.exists():
        path.unlink()


def test_forecast_journaling_and_telemetry(temp_learning_engine):
    engine = temp_learning_engine
    pred_id = engine.record_forecast(
        ticker="NVDA",
        timeframe="1w",
        current_price=200.0,
        predicted_price=210.0,
        predicted_return_pct=5.0,
        direction="BULLISH",
        direction_prob=0.72,
        lower_bound_price=195.0,
        upper_bound_price=220.0,
        gbm_weight=0.65,
        lstm_weight=0.35,
        minutes_ahead=7200
    )
    
    assert pred_id.startswith("NVDA_1w_")
    assert len(engine.records) == 1
    
    telemetry = engine.get_learning_telemetry("NVDA")
    assert telemetry.ticker == "NVDA"
    assert telemetry.total_predictions == 1
    assert telemetry.active_gbm_weight == 0.65
    assert telemetry.active_lstm_weight == 0.35


def test_realization_evaluation_and_adaptive_weights(temp_learning_engine):
    engine = temp_learning_engine
    
    # Record multiple predictions
    for i in range(5):
        engine.record_forecast(
            ticker="AAPL",
            timeframe="1d",
            current_price=150.0 + i,
            predicted_price=155.0 + i,
            predicted_return_pct=3.3,
            direction="BULLISH",
            direction_prob=0.68,
            lower_bound_price=148.0,
            upper_bound_price=160.0,
            gbm_weight=0.65,
            lstm_weight=0.35,
            minutes_ahead=0 # Immediate evaluation
        )
        
    latest_df = pd.DataFrame({
        "Close": [156.0, 157.0, 158.0]
    })
    
    evaluated = engine.evaluate_realizations("AAPL", latest_df=latest_df)
    assert evaluated == 5
    
    telemetry = engine.get_learning_telemetry("AAPL")
    assert telemetry.evaluated_predictions == 5
    assert telemetry.directional_accuracy_pct > 0.0
    assert 0.0 <= telemetry.calibration_score <= 1.0
    
    gbm_w, lstm_w = engine.get_adaptive_weights("AAPL")
    assert abs((gbm_w + lstm_w) - 1.0) < 1e-4
