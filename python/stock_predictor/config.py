"""Global configuration parameters and timeframe parsing for the Stock Price Prediction system."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path


@dataclass
class TimeframeSpec:
    timeframe_id: str          # e.g. "10m", "20m", "30m", "1h", "2h", "4h", "1d", "1w"
    data_interval: str         # yfinance interval: "1m", "5m", "15m", "1h", "1d"
    horizon_bars: int          # Number of bars forward
    minutes_ahead: int         # Total minutes ahead
    human_label: str           # e.g. "10 Minutes", "1 Hour", "1 Week"
    default_period: str        # yfinance period: "5d", "30d", "60d", "2y"


TIMEFRAME_MAP: Dict[str, TimeframeSpec] = {
    "10m": TimeframeSpec("10m", "5m", 2, 10, "10 Minutes", "5d"),
    "20m": TimeframeSpec("20m", "5m", 4, 20, "20 Minutes", "5d"),
    "30m": TimeframeSpec("30m", "5m", 6, 30, "30 Minutes", "1mo"),
    "1h": TimeframeSpec("1h", "5m", 12, 60, "1 Hour", "1mo"),
    "2h": TimeframeSpec("2h", "15m", 8, 120, "2 Hours", "1mo"),
    "4h": TimeframeSpec("4h", "15m", 16, 240, "4 Hours", "2mo"),
    "1d": TimeframeSpec("1d", "1d", 1, 1440, "1 Day", "2y"),
    "1w": TimeframeSpec("1w", "1d", 5, 7200, "1 Week (5D)", "2y"),
    "5d": TimeframeSpec("5d", "1d", 5, 7200, "1 Week (5D)", "2y"),
    "2w": TimeframeSpec("2w", "1d", 10, 14400, "2 Weeks (10D)", "2y"),
}


def parse_timeframe(timeframe_str: str) -> TimeframeSpec:
    """Parse timeframe string like '10m', '30m', '1h', '1d', '5d' into TimeframeSpec."""
    tf_clean = str(timeframe_str).lower().strip()
    if tf_clean in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[tf_clean]
    
    # Custom integer day fallback (e.g. 5 -> 5d)
    if tf_clean.isdigit():
        days = int(tf_clean)
        if days == 1:
            return TIMEFRAME_MAP["1d"]
        return TimeframeSpec(f"{days}d", "1d", days, days * 1440, f"{days} Days", "2y")
    
    # Custom minute fallback (e.g. 15m)
    if tf_clean.endswith("m") and tf_clean[:-1].isdigit():
        mins = int(tf_clean[:-1])
        return TimeframeSpec(tf_clean, "5m", max(1, mins // 5), mins, f"{mins} Minutes", "5d")
        
    # Custom hour fallback (e.g. 3h)
    if tf_clean.endswith("h") and tf_clean[:-1].isdigit():
        hrs = int(tf_clean[:-1])
        return TimeframeSpec(tf_clean, "15m", hrs * 4, hrs * 60, f"{hrs} Hours", "1mo")
        
    return TIMEFRAME_MAP["1w"]


@dataclass
class PredictionConfig:
    # Target Horizon in bars
    forecast_horizon: int = 5
    
    # Active data interval ("1m", "5m", "15m", "1h", "1d")
    data_interval: str = "1d"
    
    # Active timeframe identifier ("10m", "20m", "30m", "1h", "1d", "1w")
    timeframe: str = "1w"
    
    # Sequence length for temporal sequence models (e.g. LSTM)
    sequence_length: int = 20
    
    # Train / Validation / Test split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Embargo / purge window (bars) to prevent overlap leakage
    embargo_days: int = 5
    
    # Benchmarks to download for market regime context
    benchmark_tickers: List[str] = field(default_factory=lambda: ["SPY", "^VIX"])
    
    # Quantiles for interval estimation
    quantiles: Tuple[float, float, float] = (0.10, 0.50, 0.90)
    
    # Cache settings
    cache_dir: Path = Path.home() / ".cache" / "stock_predictor"
    cache_expiry_hours: int = 2
    
    # Default model training hyperparameters
    xgb_n_estimators: int = 80
    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 4
    
    # Neural model hyperparameters
    lstm_hidden_dim: int = 48
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.1
    lstm_epochs: int = 20
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 1e-3
    
    # Random seed
    random_state: int = 42


DEFAULT_CONFIG = PredictionConfig()
