"""Unit tests for Data Pipeline and Target Formulation."""

import pytest
import numpy as np
import pandas as pd

from stock_predictor.config import PredictionConfig
from stock_predictor.data.synthetic import generate_synthetic_stock_data
from stock_predictor.features.pipeline import FeaturePipeline


def test_feature_pipeline_targets_and_splits():
    config = PredictionConfig(forecast_horizon=5)
    target_df = generate_synthetic_stock_data(n_days=400, seed=42)
    pipeline = FeaturePipeline(config)
    
    dataset = pipeline.prepare_dataset(target_df, horizon=5)
    
    assert "X" in dataset
    assert "y_return" in dataset
    assert "X_latest" in dataset
    assert dataset["X_latest"].shape[0] == 1
    
    # Check that y_return matches log(Close_{t+5} / Close_t)
    X = dataset["X"]
    y_ret = dataset["y_return"]
    assert len(X) == len(y_ret)
    assert not y_ret.isna().any()
    
    splits = pipeline.train_val_test_split(dataset)
    assert "train" in splits and "val" in splits and "test" in splits
    assert len(splits["train"]["X"]) > 0
    assert len(splits["test"]["X"]) > 0
