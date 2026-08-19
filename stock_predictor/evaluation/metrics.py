"""Financial and statistical evaluation metrics for stock forecasting models."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    total_samples: int
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float  # Percentage of correct directional calls [0, 100]
    interval_coverage: float  # Percentage of true prices inside predicted 10-90% interval [0, 100]
    strategy_return_pct: float  # Cumulative return of model strategy (%)
    benchmark_return_pct: float  # Cumulative return of buy-and-hold (%)
    alpha_pct: float  # Strategy return - Benchmark return
    sharpe_ratio: float
    max_drawdown_pct: float
    win_loss_ratio: float
    profit_factor: float
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    annualized_volatility_pct: float = 0.0
    win_rate_pct: float = 0.0


def calculate_forecast_metrics(
    true_prices: np.ndarray,
    pred_prices: np.ndarray,
    true_returns: np.ndarray,
    pred_returns: np.ndarray,
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate core regression and directional accuracy metrics."""
    true_prices = np.asarray(true_prices, dtype=float)
    pred_prices = np.asarray(pred_prices, dtype=float)
    true_returns = np.asarray(true_returns, dtype=float)
    pred_returns = np.asarray(pred_returns, dtype=float)

    # Regression metrics
    mae = float(np.mean(np.abs(true_prices - pred_prices)))
    rmse = float(np.sqrt(np.mean((true_prices - pred_prices) ** 2)))
    mape = float(np.mean(np.abs((true_prices - pred_prices) / (true_prices + 1e-8))) * 100.0)
    
    # Directional Hit Rate (Did model correctly predict up/down?)
    dir_correct = (np.sign(true_returns) == np.sign(pred_returns))
    directional_accuracy = float(np.mean(dir_correct) * 100.0)
    
    # Interval Coverage (% inside 10th - 90th percentile)
    if lower_bounds is not None and upper_bounds is not None:
        lower_bounds = np.asarray(lower_bounds, dtype=float)
        upper_bounds = np.asarray(upper_bounds, dtype=float)
        inside = (true_prices >= lower_bounds) & (true_prices <= upper_bounds)
        coverage = float(np.mean(inside) * 100.0)
    else:
        coverage = 0.0
        
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "directional_accuracy": round(directional_accuracy, 2),
        "interval_coverage": round(coverage, 2)
    }


def calculate_strategy_performance(
    true_returns: np.ndarray,
    pred_returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    daily_returns: Optional[np.ndarray] = None,
    trade_threshold: float = 0.005,  # 0.5% expected return threshold
    transaction_cost: float = 0.0005,  # 5 bps cost per trade
    holding_period: int = 5
) -> Dict[str, float]:
    """
    Simulates realistic trading performance based on model predictions.
    
    If daily_returns is provided, performs true daily mark-to-market rebalancing.
    Otherwise, steps through discrete non-overlapping holding blocks (stride=holding_period)
    or computes period returns to avoid multi-counting overlapping returns.
    """
    true_returns = np.asarray(true_returns, dtype=float)
    pred_returns = np.asarray(pred_returns, dtype=float)
    n = len(true_returns)
    if n == 0:
        return {}

    if benchmark_returns is None:
        benchmark_returns = true_returns

    benchmark_returns = np.asarray(benchmark_returns, dtype=float)

    # Determine desired position based on predicted return
    positions = np.zeros(n)
    positions[pred_returns > trade_threshold] = 1.0
    positions[pred_returns < -trade_threshold] = -0.5  # Modest short / hedge

    if daily_returns is not None and len(daily_returns) == n:
        # Mark-to-market daily returns simulation
        d_rets = np.asarray(daily_returns, dtype=float)
        pos_changes = np.abs(np.diff(np.insert(positions, 0, 0)))
        cost_deduction = pos_changes * transaction_cost
        strategy_rets = positions * d_rets - cost_deduction
        bench_rets = d_rets
        annual_factor = np.sqrt(252)
    else:
        # Step through non-overlapping holding periods (stride = holding_period)
        # or scale returns by holding period length to prevent leverage distortion
        stride = max(1, min(holding_period, n))
        eval_indices = np.arange(0, n, stride)
        
        pos_eval = positions[eval_indices]
        ret_eval = true_returns[eval_indices]
        bench_eval = benchmark_returns[eval_indices]
        
        pos_changes = np.abs(np.diff(np.insert(pos_eval, 0, 0)))
        cost_deduction = pos_changes * transaction_cost
        strategy_rets = pos_eval * ret_eval - cost_deduction
        bench_rets = bench_eval
        annual_factor = np.sqrt(252 / holding_period)

    # Cumulative returns
    strat_cum = np.exp(np.cumsum(strategy_rets)) - 1.0
    bench_cum = np.exp(np.cumsum(bench_rets)) - 1.0
    
    strategy_total_return = float(strat_cum[-1] * 100.0) if len(strat_cum) > 0 else 0.0
    benchmark_total_return = float(bench_cum[-1] * 100.0) if len(bench_cum) > 0 else 0.0
    alpha = strategy_total_return - benchmark_total_return
    
    # Volatility and Sharpe Ratio
    mean_ret = float(np.mean(strategy_rets)) if len(strategy_rets) > 0 else 0.0
    std_ret = float(np.std(strategy_rets)) + 1e-8
    annualized_vol = float(std_ret * annual_factor * 100.0)
    sharpe = float((mean_ret / std_ret) * annual_factor) if std_ret > 1e-6 else 0.0
    
    # Downside Volatility and Sortino Ratio
    downside_returns = strategy_rets[strategy_rets < 0]
    downside_std = float(np.std(downside_returns)) + 1e-8 if len(downside_returns) > 0 else std_ret
    sortino = float((mean_ret / downside_std) * annual_factor) if downside_std > 1e-6 else 0.0
    
    # Max Drawdown
    equity_curve = np.exp(np.cumsum(strategy_rets))
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-8)
    max_dd = float(np.min(drawdown) * 100.0) if len(drawdown) > 0 else 0.0
    
    # Calmar Ratio
    calmar = float(abs(strategy_total_return / max_dd)) if abs(max_dd) > 0.1 else 0.0
    
    # Win / Loss Ratio, Win Rate & Profit Factor
    gains = strategy_rets[strategy_rets > 0]
    losses = strategy_rets[strategy_rets < 0]
    
    win_rate = float((len(gains) / len(strategy_rets)) * 100.0) if len(strategy_rets) > 0 else 0.0
    win_loss_ratio = float(len(gains) / (len(losses) + 1e-8))
    total_gain = float(np.sum(gains)) if len(gains) > 0 else 0.0
    total_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-8
    profit_factor = float(total_gain / total_loss)
    
    return {
        "strategy_return_pct": round(strategy_total_return, 2),
        "benchmark_return_pct": round(benchmark_total_return, 2),
        "alpha_pct": round(alpha, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "annualized_volatility_pct": round(annualized_vol, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 2),
        "win_loss_ratio": round(win_loss_ratio, 2),
        "profit_factor": round(profit_factor, 2)
    }
