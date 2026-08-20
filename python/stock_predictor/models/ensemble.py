"""Meta-Ensemble Stock Predictor combining Gradient Boosted Trees, Deep Learning Sequences, and Autonomous AI Analyst."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from stock_predictor.config import DEFAULT_CONFIG, PredictionConfig, parse_timeframe
from stock_predictor.models.base import BaseStockModel, ForecastResult, HorizonPoint
from stock_predictor.models.gbm_model import GBMStockModel
from stock_predictor.models.lstm_model import LSTMStockModel
from stock_predictor.models.classifier import generate_trading_signal
from stock_predictor.analytics.regime import MarketRegimeDetector
from stock_predictor.analytics.support_resistance import SupportResistanceEngine
from stock_predictor.analytics.patterns import PatternDetector
from stock_predictor.analytics.factors import QuantitativeFactorScorer
from stock_predictor.analytics.trade_planner import AlgorithmicTradePlanner
from stock_predictor.analytics.analyst import AIStockAnalyst
from stock_predictor.learning.engine import get_global_learning_engine

logger = logging.getLogger(__name__)


class EnsembleStockPredictor(BaseStockModel):
    """
    Ensemble Meta-Learner that synergizes:
    - Tabular Gradient Boosted Regressor & Classifier (capturing non-linear cross-sectional indicators)
    - PyTorch Temporal Sequence Model with Attention (capturing sequential temporal patterns)
    - Quantile Regressors (estimating 10th - 90th percentile bounds)
    - Continuous Self-Learning Engine (adaptive dynamic weighting & empirical feedback)
    - Autonomous AI Stock Analyst (institutional trade thesis, factor scores & execution plan)
    """

    def __init__(
        self,
        config: Optional[PredictionConfig] = None,
        gbm_weight: Optional[float] = None,
        lstm_weight: Optional[float] = None
    ):
        self.config = config or DEFAULT_CONFIG
        self.learning_engine = get_global_learning_engine()
        
        # If weights are not explicitly passed, adaptive defaults are used
        self.gbm_weight = gbm_weight if gbm_weight is not None else 0.65
        self.lstm_weight = lstm_weight if lstm_weight is not None else 0.35
        
        self.gbm_model = GBMStockModel(self.config)
        self.lstm_model = LSTMStockModel(self.config)
        self.is_fitted = False

        # Analytics components
        self.regime_detector = MarketRegimeDetector()
        self.sr_engine = SupportResistanceEngine()
        self.pattern_detector = PatternDetector()
        self.factor_scorer = QuantitativeFactorScorer()
        self.trade_planner = AlgorithmicTradePlanner()
        self.ai_analyst = AIStockAnalyst()

    def fit(
        self,
        X: pd.DataFrame,
        y_return: pd.Series,
        y_dir: Optional[pd.Series] = None,
        val_X: Optional[pd.DataFrame] = None,
        val_y: Optional[pd.Series] = None
    ) -> "EnsembleStockPredictor":
        logger.info(f"Fitting Ensemble Model on {len(X)} historical samples...")
        
        # Fit GBM
        self.gbm_model.fit(X, y_return, y_dir, val_X, val_y)
        
        # Fit LSTM / Temporal Sequence Model
        try:
            self.lstm_model.fit(X, y_return, y_dir, val_X, val_y)
        except Exception as e:
            logger.warning(f"Temporal model training issue: {e}. Falling back to 100% GBM weight.")
            self.gbm_weight = 1.0
            self.lstm_weight = 0.0

        self.is_fitted = True
        return self

    def predict_returns(self, X: pd.DataFrame) -> np.ndarray:
        gbm_ret = self.gbm_model.predict_returns(X)
        if self.lstm_weight > 0:
            lstm_ret = self.lstm_model.predict_returns(X)
            return self.gbm_weight * gbm_ret + self.lstm_weight * lstm_ret
        return gbm_ret

    def predict_intervals(
        self,
        X: pd.DataFrame,
        quantiles: Tuple[float, float] = (0.10, 0.90)
    ) -> Tuple[np.ndarray, np.ndarray]:
        gbm_low, gbm_high = self.gbm_model.predict_intervals(X, quantiles)
        point = self.predict_returns(X)
        
        if self.lstm_weight > 0:
            lstm_low, lstm_high = self.lstm_model.predict_intervals(X, quantiles)
            low = self.gbm_weight * gbm_low + self.lstm_weight * lstm_low
            high = self.gbm_weight * gbm_high + self.lstm_weight * lstm_high
        else:
            low, high = gbm_low, gbm_high
            
        low = np.minimum(low, point - 0.002)
        high = np.maximum(high, point + 0.002)
        return low, high

    def predict_direction_prob(self, X: pd.DataFrame) -> np.ndarray:
        gbm_prob = self.gbm_model.predict_direction_prob(X)
        if self.lstm_weight > 0:
            lstm_prob = self.lstm_model.predict_direction_prob(X)
            return np.clip(self.gbm_weight * gbm_prob + self.lstm_weight * lstm_prob, 0.05, 0.95)
        return gbm_prob

    def get_feature_importances(self) -> Dict[str, float]:
        return self.gbm_model.get_feature_importances()

    def generate_forecast(
        self,
        ticker: str,
        X_latest: pd.DataFrame,
        current_price: float,
        latest_date: pd.Timestamp,
        horizon_days: int = 5,
        timeframe: str = "1w",
        minutes_ahead: int = 7200,
        is_synthetic: bool = False,
        raw_df: Optional[pd.DataFrame] = None,
        benchmarks: Optional[Dict[str, pd.DataFrame]] = None
    ) -> ForecastResult:
        """
        Generate comprehensive forecast, quantitative analytics, adaptive self-learning tracking,
        and autonomous AI analyst synthesis report.
        """
        t_clean = ticker.upper().strip()

        # 1. Check & Apply Adaptive Dynamic Learning Weights
        adaptive_gbm_w, adaptive_lstm_w = self.learning_engine.get_adaptive_weights(t_clean)
        # Blend config/user weights with learned weights if default
        if self.gbm_weight == 0.65 and self.lstm_weight == 0.35:
            self.gbm_weight = adaptive_gbm_w
            self.lstm_weight = adaptive_lstm_w

        # 2. Predicted log return + residual drift calibration
        pred_return_raw = float(self.predict_returns(X_latest)[0])
        drift_corr_pct = self.learning_engine.get_residual_drift(t_clean)
        drift_log_ret = np.log(1.0 + (drift_corr_pct / 100.0)) if abs(drift_corr_pct) < 10.0 else 0.0
        
        pred_return = pred_return_raw + (drift_log_ret * 0.5) # Apply smooth calibration
        pred_return_pct = float((np.exp(pred_return) - 1.0) * 100.0)
        
        # Predicted price = P_t * exp(r)
        pred_price = float(current_price * np.exp(pred_return))
        
        # Predicted intervals
        low_ret, high_ret = self.predict_intervals(X_latest)
        low_ret_val = float(low_ret[0])
        high_ret_val = float(high_ret[0])
        
        lower_price = float(current_price * np.exp(low_ret_val))
        upper_price = float(current_price * np.exp(high_ret_val))
        
        low_ret_pct = float((np.exp(low_ret_val) - 1.0) * 100.0)
        high_ret_pct = float((np.exp(high_ret_val) - 1.0) * 100.0)
        
        # Directional probability
        dir_prob = float(self.predict_direction_prob(X_latest)[0])
        
        # Trading signal and confidence
        signal, direction, confidence = generate_trading_signal(
            expected_return_pct=pred_return_pct,
            up_probability=dir_prob,
            lower_bound_return_pct=low_ret_pct,
            upper_bound_return_pct=high_ret_pct
        )
        
        # Format target calendar date / time
        is_intraday = minutes_ahead < 1440
        if is_intraday:
            target_dt = latest_date + pd.Timedelta(minutes=minutes_ahead)
            target_date_str = target_dt.strftime("%Y-%m-%d %H:%M")
            forecast_date_str = latest_date.strftime("%Y-%m-%d %H:%M")
        else:
            days_add = max(1, horizon_days * 7 // 5)
            target_dt = latest_date + pd.Timedelta(days=days_add)
            target_date_str = target_dt.strftime("%Y-%m-%d")
            forecast_date_str = latest_date.strftime("%Y-%m-%d")
            
        attn_weights = self.lstm_model.get_temporal_attention_weights()
        attn_list = [float(w) for w in attn_weights] if attn_weights is not None else None
        
        # Build Multi-Horizon Term Structure Path
        if is_intraday:
            milestones = [("10m", 10), ("20m", 20), ("30m", 30), ("1h", 60), ("2h", 120), ("4h", 240)]
        else:
            milestones = [("1d", 1440), ("3d", 4320), ("1w", 7200), ("2w", 14400)]
            
        base_mins = max(1, minutes_ahead)
        multi_horizon_path: List[HorizonPoint] = []
        
        for m_tf, m_mins in milestones:
            time_ratio = float(m_mins) / float(base_mins)
            scaled_ret = pred_return * (time_ratio ** 0.75)
            scaled_ret_pct = float((np.exp(scaled_ret) - 1.0) * 100.0)
            
            p_m = float(current_price * np.exp(scaled_ret))
            half_width = max(0.005, (high_ret_val - low_ret_val) * 0.5 * np.sqrt(time_ratio))
            p_m_low = float(current_price * np.exp(scaled_ret - half_width))
            p_m_high = float(current_price * np.exp(scaled_ret + half_width))
            
            m_dir = "BULLISH" if scaled_ret_pct > 0.1 else ("BEARISH" if scaled_ret_pct < -0.1 else "NEUTRAL")
            
            if m_mins < 1440:
                m_target_str = (latest_date + pd.Timedelta(minutes=m_mins)).strftime("%H:%M")
            else:
                m_days = max(1, m_mins // 1440)
                m_target_str = (latest_date + pd.Timedelta(days=m_days * 7 // 5)).strftime("%Y-%m-%d")
                
            multi_horizon_path.append(HorizonPoint(
                timeframe=m_tf,
                minutes_ahead=m_mins,
                predicted_price=round(p_m, 2),
                predicted_return_pct=round(scaled_ret_pct, 2),
                lower_bound_price=round(p_m_low, 2),
                upper_bound_price=round(p_m_high, 2),
                direction=m_dir,
                target_time=m_target_str
            ))

        # 3. Analytics & Intelligence Pipeline
        eval_df = raw_df if raw_df is not None and len(raw_df) > 5 else pd.DataFrame({
            "Open": [current_price * 0.99, current_price * 0.995, current_price],
            "High": [current_price * 1.01, current_price * 1.005, current_price * 1.01],
            "Low": [current_price * 0.985, current_price * 0.99, current_price * 0.995],
            "Close": [current_price * 0.995, current_price, current_price],
            "Volume": [1000000, 1200000, 1100000]
        })

        regime_info = self.regime_detector.detect_regime(eval_df, benchmarks=benchmarks)
        sr_levels = self.sr_engine.calculate_levels(eval_df)
        detected_patterns = self.pattern_detector.detect_patterns(eval_df)
        factor_scores = self.factor_scorer.compute_factor_scores(eval_df)
        
        trade_plan = self.trade_planner.generate_plan(
            current_price=current_price,
            predicted_price=pred_price,
            predicted_return_pct=pred_return_pct,
            direction_prob=dir_prob,
            lower_bound_price=lower_price,
            upper_bound_price=upper_price,
            levels=sr_levels,
            regime=regime_info,
            recent_df=eval_df
        )

        # 4. Continuous Self-Learning Journaling
        self.learning_engine.record_forecast(
            ticker=t_clean,
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=pred_price,
            predicted_return_pct=pred_return_pct,
            direction=direction,
            direction_prob=dir_prob,
            lower_bound_price=lower_price,
            upper_bound_price=upper_price,
            gbm_weight=self.gbm_weight,
            lstm_weight=self.lstm_weight,
            minutes_ahead=minutes_ahead
        )

        learning_telemetry = self.learning_engine.get_learning_telemetry(t_clean)

        # 5. Autonomous AI Analyst Report Synthesis
        analyst_report = self.ai_analyst.synthesize_report(
            ticker=t_clean,
            timeframe=timeframe,
            current_price=current_price,
            predicted_price=pred_price,
            predicted_return_pct=pred_return_pct,
            direction_prob=dir_prob,
            signal=signal,
            confidence_score=confidence,
            lower_bound_price=lower_price,
            upper_bound_price=upper_price,
            levels=sr_levels,
            regime=regime_info,
            factors=factor_scores,
            patterns=detected_patterns,
            trade_plan=trade_plan,
            learning_telemetry=learning_telemetry.__dict__ if hasattr(learning_telemetry, "__dict__") else None
        )

        pattern_names = [p.name for p in detected_patterns]
            
        return ForecastResult(
            ticker=t_clean,
            current_price=round(current_price, 2),
            predicted_price=round(pred_price, 2),
            predicted_return_pct=round(pred_return_pct, 2),
            lower_bound_price=round(lower_price, 2),
            upper_bound_price=round(upper_price, 2),
            direction=direction,
            direction_prob=round(dir_prob, 4),
            signal=signal,
            confidence_score=round(confidence, 1),
            forecast_horizon_days=horizon_days,
            forecast_date=forecast_date_str,
            target_date=target_date_str,
            model_name="Ensemble (Adaptive Quantile Trees + PyTorch Sequence Attention)",
            timeframe=timeframe,
            feature_importances=self.get_feature_importances(),
            is_synthetic=is_synthetic,
            temporal_attention_weights=attn_list,
            multi_horizon_path=multi_horizon_path,
            trade_plan=trade_plan,
            support_resistance=sr_levels,
            factor_scores=factor_scores,
            market_regime=regime_info,
            analyst_report=analyst_report,
            learning_telemetry=learning_telemetry,
            detected_patterns=pattern_names
        )
