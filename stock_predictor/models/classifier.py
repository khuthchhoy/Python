"""Trading signal generation and confidence scoring engine."""

from typing import Tuple
import numpy as np


def generate_trading_signal(
    expected_return_pct: float,
    up_probability: float,
    lower_bound_return_pct: float,
    upper_bound_return_pct: float,
    recent_volatility_pct: float = 2.0
) -> Tuple[str, str, float]:
    """
    Generate actionable trading signal, directional sentiment, and confidence score.
    
    Returns:
        signal: "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL"
        direction: "BULLISH" | "BEARISH" | "NEUTRAL"
        confidence_score: float in range [0, 100]
    """
    prob_margin = abs(up_probability - 0.5) * 2.0  # [0, 1]
    
    # Confidence is higher when probability is decisive, expected return is non-trivial, and interval is tight
    interval_width = max(upper_bound_return_pct - lower_bound_return_pct, 0.1)
    certainty_factor = np.clip(1.0 / (1.0 + interval_width / 12.0), 0.3, 1.0)
    
    base_confidence = (prob_margin * 60.0 + 40.0 * certainty_factor)
    confidence_score = float(np.clip(base_confidence, 20.0, 95.0))
    
    # Determine directional bias
    if up_probability >= 0.52 and expected_return_pct > 0.15:
        direction = "BULLISH"
    elif up_probability <= 0.48 and expected_return_pct < -0.15:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"
        
    # Determine actionable trading signal
    if expected_return_pct >= 2.0 and up_probability >= 0.60 and lower_bound_return_pct > -3.0:
        signal = "STRONG_BUY"
    elif expected_return_pct >= 0.6 and up_probability >= 0.53:
        signal = "BUY"
    elif expected_return_pct <= -2.0 and up_probability <= 0.40:
        signal = "STRONG_SELL"
    elif expected_return_pct <= -0.6 and up_probability <= 0.47:
        signal = "SELL"
    else:
        signal = "HOLD"
        
    return signal, direction, confidence_score
