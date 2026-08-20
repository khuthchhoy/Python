"""Quantitative analytics, regime detection, support/resistance, Volume Profile, pattern recognition, factor scoring, and autonomous AI analyst."""

from stock_predictor.analytics.regime import MarketRegimeDetector, MarketRegimeInfo
from stock_predictor.analytics.support_resistance import SupportResistanceEngine, SupportResistanceLevels
from stock_predictor.analytics.volume_profile import (
    VolumeProfileDetector,
    VolumeProfileResult,
    VolumeProfileNode,
    VolumeSetupSignal
)
from stock_predictor.analytics.patterns import PatternDetector, DetectedPattern
from stock_predictor.analytics.factors import QuantitativeFactorScorer, FactorScores
from stock_predictor.analytics.trade_planner import AlgorithmicTradePlanner, TradePlan
from stock_predictor.analytics.analyst import AIStockAnalyst, AnalystReport

__all__ = [
    "MarketRegimeDetector",
    "MarketRegimeInfo",
    "SupportResistanceEngine",
    "SupportResistanceLevels",
    "VolumeProfileDetector",
    "VolumeProfileResult",
    "VolumeProfileNode",
    "VolumeSetupSignal",
    "PatternDetector",
    "DetectedPattern",
    "QuantitativeFactorScorer",
    "FactorScores",
    "AlgorithmicTradePlanner",
    "TradePlan",
    "AIStockAnalyst",
    "AnalystReport",
]
