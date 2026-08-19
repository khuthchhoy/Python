"""End-to-end Feature Pipeline with target formulation, feature cleaning, and leakage prevention."""

from typing import Tuple, Dict, Optional, List, Any
import numpy as np
import pandas as pd

from stock_predictor.config import DEFAULT_CONFIG, PredictionConfig
from stock_predictor.features.technical import calculate_technical_features
from stock_predictor.features.market_context import calculate_market_context_features
from stock_predictor.features.calendar import calculate_calendar_features


class FeaturePipeline:
    """Orchestrates feature extraction, cleaning, multi-horizon target alignment, and train/val/test splits."""

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.feature_names: List[str] = []

    def build_feature_matrix(
        self,
        target_df: pd.DataFrame,
        benchmarks: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """Combine technical, market context, and calendar features into a single DataFrame."""
        benchmarks = benchmarks or {}
        
        tech_feats = calculate_technical_features(target_df)
        mkt_feats = calculate_market_context_features(target_df, benchmarks)
        cal_feats = calculate_calendar_features(target_df.index)
        
        all_feats = pd.concat([tech_feats, mkt_feats, cal_feats], axis=1)
        
        # Replace infinite values and clip extreme outliers to prevent gradient explosions
        all_feats = all_feats.replace([np.inf, -np.inf], np.nan)
        all_feats = all_feats.ffill().bfill()
        return all_feats

    def prepare_dataset(
        self,
        target_df: pd.DataFrame,
        benchmarks: Optional[Dict[str, pd.DataFrame]] = None,
        horizon: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Builds feature matrix, constructs H-bar forward targets, and handles edge data.
        Returns:
            - X: Historical feature matrix (aligned with targets)
            - y_return: H-bar forward log returns
            - y_dir: H-bar forward binary direction (1 = Up, 0 = Down)
            - y_price: H-bar forward target price
            - current_prices: Price at time t corresponding to X
            - dates: Datetime index of X
            - X_latest: Feature vector for the most recent trading bar
            - latest_price: Most recent close price
            - latest_date: Timestamp of the most recent bar
            - feature_names: List of feature names
            - full_df: Full raw dataframe
        """
        h = horizon or self.config.forecast_horizon
        feats = self.build_feature_matrix(target_df, benchmarks)
        
        # Forward return target: ln(Close_{t+h} / Close_t)
        forward_close = target_df["Close"].shift(-h)
        target_return = np.log(forward_close / (target_df["Close"] + 1e-8))
        target_dir = (target_return > 0.0).astype(float)
        
        # Latest observation for forward forecast (t -> t+h)
        valid_feat_idx = feats.dropna().index
        latest_valid_idx = valid_feat_idx[-1] if len(valid_feat_idx) > 0 else target_df.index[-1]
        
        X_latest = feats.loc[[latest_valid_idx]].copy()
        latest_price = float(target_df.loc[latest_valid_idx, "Close"])
        latest_date = latest_valid_idx
        
        # Training dataset: drop rows where features or forward target are NaN
        valid_mask = feats.notna().all(axis=1) & target_return.notna()
        
        X = feats.loc[valid_mask].copy()
        y_return = target_return.loc[valid_mask].copy()
        y_dir = target_dir.loc[valid_mask].copy()
        y_price = forward_close.loc[valid_mask].copy()
        current_prices = target_df.loc[valid_mask, "Close"].copy()
        dates = X.index
        
        self.feature_names = list(X.columns)
        
        return {
            "X": X,
            "y_return": y_return,
            "y_dir": y_dir,
            "y_price": y_price,
            "current_prices": current_prices,
            "dates": dates,
            "X_latest": X_latest,
            "latest_price": latest_price,
            "latest_date": latest_date,
            "feature_names": self.feature_names,
            "full_df": target_df
        }

    def train_val_test_split(
        self,
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Temporal Walk-Forward split with embargo buffer to prevent target overlap leakage.
        """
        X = dataset["X"]
        n_samples = len(X)
        embargo = max(1, min(self.config.embargo_days, max(1, n_samples // 20)))
        
        train_end = int(n_samples * self.config.train_ratio)
        val_start = min(train_end + embargo, max(train_end, n_samples - 2))
        val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))
        test_start = min(val_end + embargo, max(val_end, n_samples - 1))
        
        if test_start >= n_samples - 5:
            # Fallback for smaller datasets
            train_end = int(n_samples * 0.75)
            val_start = train_end
            val_end = int(n_samples * 0.85)
            test_start = val_end
            
        splits = {
            "train": {
                "X": X.iloc[:train_end],
                "y_return": dataset["y_return"].iloc[:train_end],
                "y_dir": dataset["y_dir"].iloc[:train_end],
                "y_price": dataset["y_price"].iloc[:train_end],
                "prices": dataset["current_prices"].iloc[:train_end],
                "dates": dataset["dates"][:train_end]
            },
            "val": {
                "X": X.iloc[val_start:val_end],
                "y_return": dataset["y_return"].iloc[val_start:val_end],
                "y_dir": dataset["y_dir"].iloc[val_start:val_end],
                "y_price": dataset["y_price"].iloc[val_start:val_end],
                "prices": dataset["current_prices"].iloc[val_start:val_end],
                "dates": dataset["dates"][val_start:val_end]
            },
            "test": {
                "X": X.iloc[test_start:],
                "y_return": dataset["y_return"].iloc[test_start:],
                "y_dir": dataset["y_dir"].iloc[test_start:],
                "y_price": dataset["y_price"].iloc[test_start:],
                "prices": dataset["current_prices"].iloc[test_start:],
                "dates": dataset["dates"][test_start:]
            }
        }
        return splits
