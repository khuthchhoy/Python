"""Base class and output data structures for stock predictors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class HorizonPoint:
    timeframe: str
    minutes_ahead: int
    predicted_price: float
    predicted_return_pct: float
    lower_bound_price: float
    upper_bound_price: float
    direction: str
    target_time: str


@dataclass
class ForecastResult:
    ticker: str
    current_price: float
    predicted_price: float
    predicted_return_pct: float
    lower_bound_price: float
    upper_bound_price: float
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    direction_prob: float  # Probability of positive return [0, 1]
    signal: str  # "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
    confidence_score: float  # [0, 100]
    forecast_horizon_days: int
    forecast_date: str
    target_date: str
    model_name: str
    timeframe: str = "1w"
    feature_importances: Optional[Dict[str, float]] = None
    is_synthetic: bool = False
    temporal_attention_weights: Optional[List[float]] = None
    multi_horizon_path: Optional[List[HorizonPoint]] = None


class BaseStockModel(ABC):
    """Abstract interface for all stock price prediction models."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y_return: pd.Series,
        y_dir: Optional[pd.Series] = None,
        val_X: Optional[pd.DataFrame] = None,
        val_y: Optional[pd.Series] = None
    ) -> "BaseStockModel":
        """Train model on historical features and forward return targets."""
        pass

    @abstractmethod
    def predict_returns(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected H-bar forward log returns."""
        pass

    @abstractmethod
    def predict_intervals(self, X: pd.DataFrame, quantiles: Tuple[float, float] = (0.10, 0.90)) -> Tuple[np.ndarray, np.ndarray]:
        """Predict lower and upper return quantiles for uncertainty estimation."""
        pass

    @abstractmethod
    def predict_direction_prob(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of positive return [0, 1]."""
        pass

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importance dictionary sorted descending."""
        return {}
