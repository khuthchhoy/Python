"""Unit tests for Quantitative Analytics (Regimes, S/R, Patterns, Factors, Trade Planner)."""

import pytest
import pandas as pd
import numpy as np

from stock_predictor.data.synthetic import generate_synthetic_stock_data
from stock_predictor.analytics.regime import MarketRegimeDetector
from stock_predictor.analytics.support_resistance import SupportResistanceEngine
from stock_predictor.analytics.patterns import PatternDetector
from stock_predictor.analytics.factors import QuantitativeFactorScorer
from stock_predictor.analytics.trade_planner import AlgorithmicTradePlanner


@pytest.fixture
def sample_stock_data():
    return generate_synthetic_stock_data("TEST", "2023-01-01", "2023-08-01", initial_price=150.0)


def test_market_regime_detection(sample_stock_data):
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(sample_stock_data)
    
    assert regime.trend_regime in [
        "STRONG_BULLISH", "MODERATE_BULLISH", "SIDEWAYS_CHOP", "MODERATE_BEARISH", "STRONG_BEARISH"
    ]
    assert regime.volatility_regime in [
        "LOW_VOLATILITY_COMPRESSION", "NORMAL_VOLATILITY", "HIGH_VOLATILITY_EXPANSION", "EXTREME_VOLATILITY_PANIC"
    ]
    assert 0.0 <= regime.adx_proxy <= 100.0
    assert 0.0 <= regime.volatility_percentile <= 100.0
    assert len(regime.regime_summary) > 20


def test_support_resistance_levels(sample_stock_data):
    engine = SupportResistanceEngine()
    levels = engine.calculate_levels(sample_stock_data)
    
    p = levels.current_price
    assert levels.support_1 < p
    assert levels.support_2 <= levels.support_1
    assert levels.resistance_1 > p
    assert levels.resistance_2 >= levels.resistance_1
    assert levels.breakout_level >= levels.resistance_1
    assert levels.breakdown_level <= levels.support_1
    assert levels.nearest_level_type in ["SUPPORT", "RESISTANCE"]
    assert levels.nearest_level_distance_pct >= 0.0


def test_pattern_detector(sample_stock_data):
    detector = PatternDetector()
    patterns = detector.detect_patterns(sample_stock_data)
    
    assert isinstance(patterns, list)
    for p in patterns:
        assert p.category in ["MOMENTUM", "VOLATILITY", "CANDLESTICK", "TREND"]
        assert p.bias in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert 0.0 <= p.confidence <= 100.0


def test_factor_scorer(sample_stock_data):
    scorer = QuantitativeFactorScorer()
    scores = scorer.compute_factor_scores(sample_stock_data)
    
    assert 0.0 <= scores.momentum_score <= 100.0
    assert 0.0 <= scores.trend_score <= 100.0
    assert 0.0 <= scores.volatility_score <= 100.0
    assert 0.0 <= scores.flow_score <= 100.0
    assert 0.0 <= scores.composite_score <= 100.0
    assert scores.verdict in ["EXCEPTIONAL", "FAVORABLE", "NEUTRAL", "UNFAVORABLE", "EXTREME_RISK"]


def test_trade_planner(sample_stock_data):
    engine_sr = SupportResistanceEngine()
    regime_det = MarketRegimeDetector()
    planner = AlgorithmicTradePlanner()
    
    levels = engine_sr.calculate_levels(sample_stock_data)
    regime = regime_det.detect_regime(sample_stock_data)
    cur_p = levels.current_price
    
    plan = planner.generate_plan(
        current_price=cur_p,
        predicted_price=cur_p * 1.03,
        predicted_return_pct=3.0,
        direction_prob=0.70,
        lower_bound_price=cur_p * 0.98,
        upper_bound_price=cur_p * 1.05,
        levels=levels,
        regime=regime,
        recent_df=sample_stock_data
    )
    
    assert plan.action in ["ACCUMULATE", "BUY LIMIT", "BREAKOUT BUY", "HOLD / MONITOR", "SCALE OUT", "SELL / SHORT"]
    assert plan.stop_loss < cur_p
    assert plan.stop_loss_pct < 0.0
    assert plan.target_1 > cur_p
    assert plan.risk_reward_ratio > 0.0
    assert plan.var_95_pct > 0.0
    assert 0.0 <= plan.kelly_size_pct <= 25.0
