"""Unit tests for Autonomous AI Stock Analyst."""

import pytest
import pandas as pd

from stock_predictor.analytics.analyst import AIStockAnalyst
from stock_predictor.analytics.regime import MarketRegimeDetector
from stock_predictor.analytics.support_resistance import SupportResistanceEngine
from stock_predictor.analytics.patterns import PatternDetector
from stock_predictor.analytics.factors import QuantitativeFactorScorer
from stock_predictor.analytics.trade_planner import AlgorithmicTradePlanner
from stock_predictor.data.synthetic import generate_synthetic_stock_data


def test_ai_stock_analyst_synthesis():
    df = generate_synthetic_stock_data("NVDA", "2023-01-01", "2023-08-01", initial_price=220.0)
    
    regime = MarketRegimeDetector().detect_regime(df)
    levels = SupportResistanceEngine().calculate_levels(df)
    patterns = PatternDetector().detect_patterns(df)
    factors = QuantitativeFactorScorer().compute_factor_scores(df)
    
    cur_p = levels.current_price
    trade_plan = AlgorithmicTradePlanner().generate_plan(
        current_price=cur_p,
        predicted_price=cur_p * 1.04,
        predicted_return_pct=4.0,
        direction_prob=0.74,
        lower_bound_price=cur_p * 0.98,
        upper_bound_price=cur_p * 1.08,
        levels=levels,
        regime=regime,
        recent_df=df
    )
    
    analyst = AIStockAnalyst()
    report = analyst.synthesize_report(
        ticker="NVDA",
        timeframe="1w",
        current_price=cur_p,
        predicted_price=cur_p * 1.04,
        predicted_return_pct=4.0,
        direction_prob=0.74,
        signal="STRONG_BUY",
        confidence_score=82.0,
        lower_bound_price=cur_p * 0.98,
        upper_bound_price=cur_p * 1.08,
        levels=levels,
        regime=regime,
        factors=factors,
        patterns=patterns,
        trade_plan=trade_plan,
        learning_telemetry={"total_predictions": 25, "directional_accuracy_pct": 76.0, "calibration_score": 0.90}
    )
    
    assert report.ticker == "NVDA"
    assert report.timeframe == "1w"
    assert len(report.executive_summary) > 50
    assert len(report.primary_catalysts) >= 3
    assert len(report.contrarian_risks) >= 2
    assert 0.0 <= report.conviction_score <= 100.0
    assert report.trade_plan.action in ["ACCUMULATE", "BUY LIMIT", "BREAKOUT BUY", "HOLD / MONITOR", "SCALE OUT", "SELL / SHORT"]
