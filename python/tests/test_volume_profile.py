"""Unit tests for Trader Dale Volume Profile Engine, Setups, and Analytics Integration."""

import pytest
import numpy as np
import pandas as pd

from stock_predictor.analytics.volume_profile import (
    VolumeProfileDetector,
    VolumeProfileResult,
    VolumeProfileNode,
    VolumeSetupSignal
)
from stock_predictor.analytics.support_resistance import SupportResistanceEngine
from stock_predictor.analytics.trade_planner import AlgorithmicTradePlanner
from stock_predictor.analytics.factors import QuantitativeFactorScorer
from stock_predictor.analytics.analyst import AIStockAnalyst
from stock_predictor.analytics.regime import MarketRegimeDetector
from stock_predictor.models.ensemble import EnsembleStockPredictor
from stock_predictor.features.pipeline import FeaturePipeline
from stock_predictor.features.technical import calculate_technical_features
from stock_predictor.config import DEFAULT_CONFIG


def generate_synthetic_candles(n_bars: int = 50, base_price: float = 100.0) -> pd.DataFrame:
    """Generates synthetic OHLCV dataframe with realistic price movements."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=n_bars, freq="B")
    
    close_prices = [base_price]
    for _ in range(n_bars - 1):
        ret = rng.normal(0.001, 0.015)
        close_prices.append(close_prices[-1] * (1.0 + ret))
        
    close = np.array(close_prices)
    high = close * (1.0 + rng.uniform(0.002, 0.015, n_bars))
    low = close * (1.0 - rng.uniform(0.002, 0.015, n_bars))
    open_p = low + (high - low) * rng.uniform(0.2, 0.8, n_bars)
    volume = rng.integers(500000, 2500000, n_bars)

    return pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    }, index=dates)


def test_volume_profile_poc_and_value_area():
    """Test calculation of Point of Control (POC) and 70% Value Area."""
    df = generate_synthetic_candles(n_bars=60, base_price=150.0)
    detector = VolumeProfileDetector(num_bins=40, value_area_pct=0.70)
    vp = detector.compute_volume_profile(df)

    assert isinstance(vp, VolumeProfileResult)
    assert vp.poc_price > 0
    assert vp.val_price <= vp.poc_price <= vp.vah_price
    assert vp.total_volume > 0
    assert len(vp.nodes) == 40
    assert vp.profile_shape in ["D_SHAPE", "P_SHAPE", "B_SHAPE", "THIN_PROFILE"]

    # Verify POC is marked on exactly one node
    poc_nodes = [n for n in vp.nodes if n.is_poc]
    assert len(poc_nodes) == 1
    assert poc_nodes[0].price == vp.poc_price


def test_volume_profile_morphologies():
    """Test classification of P-shape (buying), b-shape (selling), and D-shape (balance)."""
    detector = VolumeProfileDetector(num_bins=30)
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    
    # 1. Bullish P-profile: thin impulse at bottom (low volume), dense accumulation at top (high volume)
    closes_p = [100.0 + i * 0.8 for i in range(10)] + [108.0 + (i % 3) * 0.5 for i in range(20)]
    vols_p = [100000] * 10 + [3000000] * 20
    df_p = pd.DataFrame({
        "Open": [c - 0.2 for c in closes_p],
        "High": [c + 0.5 for c in closes_p],
        "Low": [c - 0.5 for c in closes_p],
        "Close": closes_p,
        "Volume": vols_p
    }, index=dates)
    
    vp_p = detector.compute_volume_profile(df_p)
    assert vp_p.profile_shape == "P_SHAPE"

    # 2. Bearish b-profile: thin drop at top (low volume), dense distribution at bottom (high volume)
    closes_b = [110.0 - i * 0.8 for i in range(10)] + [102.0 - (i % 3) * 0.5 for i in range(20)]
    vols_b = [100000] * 10 + [3000000] * 20
    df_b = pd.DataFrame({
        "Open": [c + 0.2 for c in closes_b],
        "High": [c + 0.5 for c in closes_b],
        "Low": [c - 0.5 for c in closes_b],
        "Close": closes_b,
        "Volume": vols_b
    }, index=dates)

    vp_b = detector.compute_volume_profile(df_b)
    assert vp_b.profile_shape == "B_SHAPE"



def test_trader_dale_setups_detection():
    """Test Trader Dale's Accumulation (#1), Trend (#2), and Rejection (#3) setups."""
    df = generate_synthetic_candles(n_bars=50, base_price=120.0)
    detector = VolumeProfileDetector()
    vp = detector.compute_volume_profile(df)

    assert isinstance(vp.detected_setups, list)
    for s in vp.detected_setups:
        assert isinstance(s, VolumeSetupSignal)
        assert s.setup_type in ["ACCUMULATION", "TREND", "REJECTION", "REVERSAL"]
        assert s.bias in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert s.entry_level > 0
        assert s.stop_loss_level > 0
        assert s.target_level > 0
        assert 0 <= s.confidence <= 100


def test_volume_profile_features_in_pipeline():
    """Test that vectorized Volume Profile features are computed in technical features and pipeline."""
    df = generate_synthetic_candles(n_bars=50, base_price=200.0)
    feats = calculate_technical_features(df)

    assert "vp_poc_dist_pct" in feats.columns
    assert "vp_in_value_area" in feats.columns
    assert "vp_val_dist_pct" in feats.columns
    assert "vp_vah_dist_pct" in feats.columns
    assert "vp_shape_skew" in feats.columns
    assert "vp_accumulation_index" in feats.columns

    # Check non-empty values
    assert not feats["vp_poc_dist_pct"].isna().all()
    assert not feats["vp_in_value_area"].isna().all()

    pipeline = FeaturePipeline(DEFAULT_CONFIG)
    dataset = pipeline.prepare_dataset(df, horizon=5)
    assert "vp_poc_dist_pct" in dataset["X"].columns
    assert "vp_in_value_area" in dataset["X"].columns


def test_trade_planner_volume_based_stops():
    """Test that AlgorithmicTradePlanner incorporates Volume-Based Stop-Loss and Targets."""
    df = generate_synthetic_candles(n_bars=40, base_price=100.0)
    vp_detector = VolumeProfileDetector()
    vp = vp_detector.compute_volume_profile(df)
    
    sr_engine = SupportResistanceEngine()
    levels = sr_engine.calculate_levels(df, volume_profile=vp)
    
    regime_detector = MarketRegimeDetector()
    regime = regime_detector.detect_regime(df)

    planner = AlgorithmicTradePlanner()
    plan = planner.generate_plan(
        current_price=100.0,
        predicted_price=104.0,
        predicted_return_pct=4.0,
        direction_prob=0.72,
        lower_bound_price=98.0,
        upper_bound_price=106.0,
        levels=levels,
        regime=regime,
        recent_df=df,
        volume_profile=vp
    )

    assert plan.action in ["ACCUMULATE", "BUY LIMIT", "BREAKOUT BUY"]
    assert plan.stop_loss < 100.0
    assert plan.target_1 > 100.0
    assert plan.risk_reward_ratio >= 0.8
    assert 0.0 <= plan.kelly_size_pct <= 25.0


def test_ai_analyst_volume_profile_synthesis():
    """Test AIStockAnalyst synthesizes Volume Profile insights."""
    df = generate_synthetic_candles(n_bars=40, base_price=100.0)
    vp_detector = VolumeProfileDetector()
    vp = vp_detector.compute_volume_profile(df)
    
    sr_engine = SupportResistanceEngine()
    levels = sr_engine.calculate_levels(df, volume_profile=vp)
    
    regime_detector = MarketRegimeDetector()
    regime = regime_detector.detect_regime(df)
    
    factor_scorer = QuantitativeFactorScorer()
    factors = factor_scorer.compute_factor_scores(df, volume_profile=vp)
    
    planner = AlgorithmicTradePlanner()
    plan = planner.generate_plan(
        current_price=100.0,
        predicted_price=103.0,
        predicted_return_pct=3.0,
        direction_prob=0.68,
        lower_bound_price=98.5,
        upper_bound_price=105.0,
        levels=levels,
        regime=regime,
        recent_df=df,
        volume_profile=vp
    )

    analyst = AIStockAnalyst()
    report = analyst.synthesize_report(
        ticker="AAPL",
        timeframe="1w",
        current_price=100.0,
        predicted_price=103.0,
        predicted_return_pct=3.0,
        direction_prob=0.68,
        signal="BUY",
        confidence_score=78.0,
        lower_bound_price=98.5,
        upper_bound_price=105.0,
        levels=levels,
        regime=regime,
        factors=factors,
        patterns=[],
        trade_plan=plan,
        volume_profile=vp
    )

    assert report.ticker == "AAPL"
    assert report.volume_profile_summary is not None
    assert "Point of Control" in report.volume_profile_summary
    assert "Value Area" in report.volume_profile_summary


def test_ensemble_predictor_forecast_with_volume_profile():
    """Test end-to-end Ensemble forecast generation returns Volume Profile data."""
    df = generate_synthetic_candles(n_bars=50, base_price=100.0)
    pipeline = FeaturePipeline(DEFAULT_CONFIG)
    dataset = pipeline.prepare_dataset(df, horizon=5)
    splits = pipeline.train_val_test_split(dataset)

    ensemble = EnsembleStockPredictor(config=DEFAULT_CONFIG)
    ensemble.fit(splits["train"]["X"], splits["train"]["y_return"], splits["train"]["y_dir"])

    forecast = ensemble.generate_forecast(
        ticker="TEST",
        X_latest=dataset["X_latest"],
        current_price=dataset["latest_price"],
        latest_date=dataset["latest_date"],
        horizon_days=5,
        timeframe="1w",
        raw_df=df
    )

    assert forecast.volume_profile is not None
    assert forecast.volume_profile.poc_price > 0
    assert forecast.volume_profile.val_price <= forecast.volume_profile.vah_price
    assert isinstance(forecast.volume_setups, list)
