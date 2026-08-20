"""Gradient Boosted Decision Trees Model for Return & Uncertainty Forecasting."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from stock_predictor.config import DEFAULT_CONFIG, PredictionConfig
from stock_predictor.models.base import BaseStockModel

logger = logging.getLogger(__name__)


class GBMStockModel(BaseStockModel):
    """
    Gradient Boosted Model predicting:
    1. Point estimate for forward log-return
    2. Quantile estimates (10th and 90th percentiles) with strict non-crossing guarantees
    3. Calibrated directional win probability P(Up)
    """

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.model_point = None
        self.model_lower = None
        self.model_upper = None
        self.model_classifier = None
        self._importances: Dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y_return: pd.Series,
        y_dir: Optional[pd.Series] = None,
        val_X: Optional[pd.DataFrame] = None,
        val_y: Optional[pd.Series] = None
    ) -> "GBMStockModel":
        self.feature_names = list(X.columns)
        
        # Fit scaler on training data only
        X_scaled = self.scaler.fit_transform(X)
        y_arr = y_return.values.astype(float)
        
        if y_dir is None:
            y_dir = (y_return > 0).astype(int)
        y_dir_arr = y_dir.values.astype(int)
        
        # 1. Train Point Forecaster
        self.model_point = HistGradientBoostingRegressor(
            max_iter=self.config.xgb_n_estimators,
            learning_rate=self.config.xgb_learning_rate,
            max_depth=self.config.xgb_max_depth,
            random_state=self.config.random_state
        )
        self.model_point.fit(X_scaled, y_arr)
        
        # 2. Train Quantile Regressors for Confidence Intervals
        # Lower bound (10th percentile)
        self.model_lower = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=self.config.quantiles[0],
            max_iter=max(40, self.config.xgb_n_estimators // 2),
            learning_rate=0.04,
            max_depth=3,
            random_state=self.config.random_state
        )
        self.model_lower.fit(X_scaled, y_arr)
        
        # Upper bound (90th percentile)
        self.model_upper = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=self.config.quantiles[2],
            max_iter=max(40, self.config.xgb_n_estimators // 2),
            learning_rate=0.04,
            max_depth=3,
            random_state=self.config.random_state
        )
        self.model_upper.fit(X_scaled, y_arr)
        
        # 3. Train Directional Classifier P(Up)
        self.model_classifier = HistGradientBoostingClassifier(
            max_iter=self.config.xgb_n_estimators,
            learning_rate=0.03,
            max_depth=3,
            random_state=self.config.random_state
        )
        self.model_classifier.fit(X_scaled, y_dir_arr)
        
        # Calculate feature importances
        corrs = np.array([
            abs(np.corrcoef(X_scaled[:, i], y_arr)[0, 1]) if np.std(X_scaled[:, i]) > 1e-6 else 0.0
            for i in range(X_scaled.shape[1])
        ])
        corrs = np.nan_to_num(corrs)
        total_imp = np.sum(corrs) + 1e-8
        norm_imp = corrs / total_imp
        self._importances = {
            feat: float(score) for feat, score in sorted(
                zip(self.feature_names, norm_imp),
                key=lambda item: item[1],
                reverse=True
            )
        }
        
        return self

    def predict_returns(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model_point.predict(X_scaled)

    def predict_intervals(
        self,
        X: pd.DataFrame,
        quantiles: Tuple[float, float] = (0.10, 0.90)
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        point = self.model_point.predict(X_scaled)
        lower = self.model_lower.predict(X_scaled)
        upper = self.model_upper.predict(X_scaled)
        
        # Enforce strict non-crossing monotonicity: lower <= point <= upper
        lower = np.minimum(lower, point - 0.002)
        upper = np.maximum(upper, point + 0.002)
        return lower, upper

    def predict_direction_prob(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        probs = self.model_classifier.predict_proba(X_scaled)
        # Probability of class 1 (Up)
        if probs.shape[1] == 2:
            return np.clip(probs[:, 1], 0.05, 0.95)
        return np.full(len(X), 0.5)

    def get_feature_importances(self) -> Dict[str, float]:
        return self._importances
