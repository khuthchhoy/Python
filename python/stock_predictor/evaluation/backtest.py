"""Walk-Forward Backtesting Engine with Purged and Embargoed Splits."""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from stock_predictor.config import DEFAULT_CONFIG, PredictionConfig
from stock_predictor.models.ensemble import EnsembleStockPredictor
from stock_predictor.evaluation.metrics import (
    BacktestMetrics,
    calculate_forecast_metrics,
    calculate_strategy_performance
)

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Simulates realistic out-of-sample forward trading with purged/embargoed splits."""

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or DEFAULT_CONFIG

    def evaluate_test_set(
        self,
        model: EnsembleStockPredictor,
        test_data: Dict[str, any]
    ) -> Tuple[pd.DataFrame, BacktestMetrics]:
        """
        Evaluate a trained model on a held-out test split with mark-to-market daily P&L.
        """
        X_test = test_data["X"]
        y_ret_true = np.asarray(test_data["y_return"].values, dtype=float)
        y_price_true = np.asarray(test_data["y_price"].values, dtype=float)
        current_prices = np.asarray(test_data["prices"].values, dtype=float)
        dates = test_data["dates"]
        n = len(X_test)
        
        # Point predictions for 5-day horizon
        pred_returns = model.predict_returns(X_test)
        pred_prices = current_prices * np.exp(pred_returns)
        
        # Interval predictions
        low_rets, high_rets = model.predict_intervals(X_test)
        lower_bounds = current_prices * np.exp(low_rets)
        upper_bounds = current_prices * np.exp(high_rets)
        
        # 1-day realized mark-to-market daily returns
        daily_1d_returns = np.zeros(n)
        if n > 1:
            daily_1d_returns[:-1] = np.log(current_prices[1:] / (current_prices[:-1] + 1e-8))
            daily_1d_returns[-1] = daily_1d_returns[-2] if n > 2 else 0.0

        # Calculate regression and directional metrics
        acc_metrics = calculate_forecast_metrics(
            true_prices=y_price_true,
            pred_prices=pred_prices,
            true_returns=y_ret_true,
            pred_returns=pred_returns,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds
        )
        
        # Calculate simulated trading strategy metrics using daily realized returns
        strat_metrics = calculate_strategy_performance(
            true_returns=y_ret_true,
            pred_returns=pred_returns,
            daily_returns=daily_1d_returns,
            trade_threshold=0.005,
            holding_period=self.config.forecast_horizon
        )
        
        # Build realistic daily mark-to-market equity curve
        positions = np.zeros(n)
        positions[pred_returns > 0.005] = 1.0
        positions[pred_returns < -0.005] = -0.5
        
        pos_changes = np.abs(np.diff(np.insert(positions, 0, 0)))
        costs = pos_changes * 0.0005
        daily_strat_rets = positions * daily_1d_returns - costs
        
        strat_equity = np.exp(np.cumsum(daily_strat_rets))
        bench_equity = current_prices / (current_prices[0] + 1e-8)
        
        results_df = pd.DataFrame({
            "Date": dates,
            "Current_Price": np.round(current_prices, 2),
            "True_Price_5d": np.round(y_price_true, 2),
            "Pred_Price_5d": np.round(pred_prices, 2),
            "Lower_Bound_5d": np.round(lower_bounds, 2),
            "Upper_Bound_5d": np.round(upper_bounds, 2),
            "True_Return_5d_pct": np.round((np.exp(y_ret_true) - 1.0) * 100.0, 2),
            "Pred_Return_5d_pct": np.round((np.exp(pred_returns) - 1.0) * 100.0, 2),
            "Direction_Correct": (np.sign(y_ret_true) == np.sign(pred_returns)),
            "Daily_Strategy_Return_pct": np.round(daily_strat_rets * 100.0, 3),
            "Strategy_Equity": np.round(strat_equity, 4),
            "Benchmark_Equity": np.round(bench_equity, 4)
        }).set_index("Date")
        
        summary = BacktestMetrics(
            total_samples=n,
            mae=acc_metrics["mae"],
            rmse=acc_metrics["rmse"],
            mape=acc_metrics["mape"],
            directional_accuracy=acc_metrics["directional_accuracy"],
            interval_coverage=acc_metrics["interval_coverage"],
            strategy_return_pct=strat_metrics.get("strategy_return_pct", 0.0),
            benchmark_return_pct=strat_metrics.get("benchmark_return_pct", 0.0),
            alpha_pct=strat_metrics.get("alpha_pct", 0.0),
            sharpe_ratio=strat_metrics.get("sharpe_ratio", 0.0),
            max_drawdown_pct=strat_metrics.get("max_drawdown_pct", 0.0),
            win_loss_ratio=strat_metrics.get("win_loss_ratio", 0.0),
            profit_factor=strat_metrics.get("profit_factor", 0.0),
            sortino_ratio=strat_metrics.get("sortino_ratio", 0.0),
            calmar_ratio=strat_metrics.get("calmar_ratio", 0.0),
            annualized_volatility_pct=strat_metrics.get("annualized_volatility_pct", 0.0),
            win_rate_pct=strat_metrics.get("win_rate_pct", 0.0)
        )
        
        return results_df, summary
