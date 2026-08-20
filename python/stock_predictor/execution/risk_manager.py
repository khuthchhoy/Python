import math
import numpy as np
from typing import Dict, Any

class RiskManager:
    """
    Institutional-grade Risk Management Engine.
    Handles dynamic position sizing, max drawdown enforcement, and volatility scaling.
    """
    
    def __init__(
        self, 
        initial_capital: float = 100000.0,
        max_position_pct: float = 0.20,
        max_drawdown_pct: float = 0.10,
        kelly_fraction: float = 0.5  # Half-Kelly for safety
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.kelly_fraction = kelly_fraction
        
        # Win rate and reward metrics for Kelly (defaults until calibrated)
        self.historical_win_rate = 0.55
        self.historical_reward_to_risk = 1.5

    def update_capital(self, new_capital: float):
        """Update current capital and recalculate peak for drawdown limits."""
        self.current_capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

    def is_trading_halted(self) -> bool:
        """Check if max drawdown has been breached."""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        return drawdown >= self.max_drawdown_pct

    def calculate_kelly_size(self, win_prob: float, reward_risk_ratio: float) -> float:
        """
        Calculate the optimal bet size fraction using the Kelly Criterion.
        f = p - (q / b)
        where:
        f = fraction of bankroll
        p = probability of win
        q = probability of loss (1 - p)
        b = proportion of the bet gained with a win (reward:risk ratio)
        """
        if reward_risk_ratio <= 0:
            return 0.0
            
        p = win_prob
        q = 1.0 - p
        
        kelly_f = p - (q / reward_risk_ratio)
        
        if kelly_f <= 0:
            return 0.0
            
        # Apply fractional Kelly for safer compounding
        return kelly_f * self.kelly_fraction

    def get_position_size(
        self, 
        ticker: str, 
        confidence_score: float, 
        predicted_return_pct: float,
        volatility_proxy: float = 1.0
    ) -> float:
        """
        Returns the dollar amount to allocate to this trade.
        Returns 0.0 if trading is halted or risk metrics are unfavorable.
        """
        if self.is_trading_halted():
            return 0.0
            
        # Convert confidence (0-100) to win probability (0-1)
        win_prob = max(0.0, min(1.0, confidence_score / 100.0))
        
        if win_prob < 0.50:
            return 0.0  # Do not take negative expectancy trades
            
        # Use provided return or fallback to historical
        reward_risk = max(0.1, abs(predicted_return_pct) / 2.0)  # Assume a 2% stop loss scale
        
        kelly_pct = self.calculate_kelly_size(win_prob, reward_risk)
        
        # Volatility Scaling: Reduce size in high volatility
        vol_scaler = 1.0 / max(1.0, volatility_proxy)
        scaled_pct = kelly_pct * vol_scaler
        
        # Enforce max position limits
        final_pct = min(scaled_pct, self.max_position_pct)
        
        return self.current_capital * final_pct
