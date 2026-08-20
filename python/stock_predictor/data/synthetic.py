"""Synthetic Market Data Generator with realistic ticker price anchors."""

from typing import Tuple, Optional, Dict
import hashlib
import numpy as np
import pandas as pd

# Known anchor prices for major market assets (fallback when offline/sandboxed)
KNOWN_TICKER_PRICES: Dict[str, float] = {
    "DELL": 138.50,
    "NVDA": 218.50,
    "AAPL": 316.50,
    "MSFT": 485.00,
    "TSLA": 298.00,
    "AMZN": 224.50,
    "GOOGL": 195.00,
    "GOOG": 196.00,
    "META": 645.00,
    "SPY": 770.00,
    "QQQ": 560.00,
    "AMD": 168.00,
    "AVGO": 182.00,
    "NFLX": 820.00,
    "PLTR": 62.50,
    "COIN": 265.00,
    "MSTR": 380.00,
    "INTC": 24.50,
    "SMCI": 48.00,
    "LLY": 890.00,
    "JPM": 242.00,
    "BAC": 44.50,
    "WMT": 92.00,
    "COST": 965.00,
    "BABA": 98.00,
    "^VIX": 15.30,
    "^GSPC": 6050.00,
    "^DJI": 44200.00,
}


def get_anchor_price_for_ticker(ticker: str, custom_price: Optional[float] = None) -> float:
    """Return realistic anchor market price for any ticker symbol."""
    if custom_price is not None and custom_price > 0:
        return float(custom_price)
    
    t_clean = ticker.upper().strip()
    if t_clean in KNOWN_TICKER_PRICES:
        return KNOWN_TICKER_PRICES[t_clean]
    
    # Deterministic pseudo-random price for unknown tickers based on name hash
    hash_val = int(hashlib.md5(t_clean.encode()).hexdigest()[:6], 16)
    return round(45.0 + (hash_val % 220) + (hash_val % 100) / 100.0, 2)


def generate_synthetic_stock_data(
    ticker: str = "SYNTH",
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    n_days: int = 1200,
    initial_price: Optional[float] = None,
    annual_drift: float = 0.12,
    annual_volatility: float = 0.28,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic daily OHLCV data ending at the ticker's real-world price anchor.
    """
    rng = np.random.default_rng(seed)
    anchor_price = get_anchor_price_for_ticker(ticker, initial_price)
    
    if end_date is None:
        dates = pd.bdate_range(start=start_date, periods=n_days)
    else:
        dates = pd.bdate_range(start=start_date, end=end_date)
        n_days = len(dates)
        
    dt = 1.0 / 252.0  # Daily time-step
    
    # Stochastic volatility
    vol = np.zeros(n_days)
    vol[0] = annual_volatility
    kappa = 2.0
    theta = annual_volatility
    xi = 0.15
    
    for t in range(1, n_days):
        dW = rng.normal(0, np.sqrt(dt))
        vol[t] = np.clip(vol[t-1] + kappa * (theta - vol[t-1]) * dt + xi * np.sqrt(max(vol[t-1], 0.05)) * dW, 0.05, 0.80)
        
    # Generate relative price path
    log_returns = np.zeros(n_days)
    jump_intensity = 0.04
    
    for t in range(1, n_days):
        z = rng.normal(0, 1)
        jump = rng.normal(-0.01, 0.05) if rng.random() < jump_intensity else 0.0
        drift_term = (annual_drift - 0.5 * vol[t]**2) * dt
        diffusion_term = vol[t] * np.sqrt(dt) * z
        log_returns[t] = drift_term + diffusion_term + jump
        
    cum_returns = np.exp(np.cumsum(log_returns))
    
    # Scale series so the final closing price exactly matches anchor_price
    closes = anchor_price * (cum_returns / cum_returns[-1])
    
    opens = np.zeros(n_days)
    highs = np.zeros(n_days)
    lows = np.zeros(n_days)
    volumes = np.zeros(n_days, dtype=int)
    
    opens[0] = closes[0] * (1 + rng.normal(0, 0.002))
    highs[0] = max(opens[0], closes[0]) * (1 + abs(rng.normal(0, 0.008)))
    lows[0] = min(opens[0], closes[0]) * (1 - abs(rng.normal(0, 0.008)))
    volumes[0] = int(rng.lognormal(mean=15.0, sigma=0.5))
    
    for t in range(1, n_days):
        gap = rng.normal(0, 0.003 * (vol[t] / annual_volatility))
        opens[t] = closes[t-1] * (1 + gap)
        
        intraday_vol = vol[t] * np.sqrt(dt) * 0.7
        high_bump = abs(rng.normal(0, intraday_vol))
        low_bump = abs(rng.normal(0, intraday_vol))
        
        highs[t] = max(opens[t], closes[t]) * (1 + high_bump)
        lows[t] = min(opens[t], closes[t]) * (1 - low_bump)
        
        ret = abs((closes[t] - closes[t-1]) / closes[t-1])
        vol_multiplier = 1.0 + 15.0 * ret + 2.0 * (vol[t] / annual_volatility)
        base_vol = rng.lognormal(mean=14.5, sigma=0.4)
        volumes[t] = int(base_vol * vol_multiplier)
        
    df = pd.DataFrame({
        "Open": np.round(opens, 2),
        "High": np.round(highs, 2),
        "Low": np.round(lows, 2),
        "Close": np.round(closes, 2),
        "Adj Close": np.round(closes, 2),
        "Volume": volumes
    }, index=dates)
    df.index.name = "Date"
    return df


def generate_synthetic_intraday_data(
    ticker: str = "SYNTH",
    interval: str = "5m",
    n_bars: int = 600,
    initial_price: Optional[float] = None,
    annual_volatility: float = 0.35,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate high-frequency intraday stock bars ending at the ticker's real-world price anchor.
    """
    rng = np.random.default_rng(seed)
    anchor_price = get_anchor_price_for_ticker(ticker, initial_price)
    
    if interval == "1m":
        step_min = 1
    elif interval == "5m":
        step_min = 5
    elif interval == "15m":
        step_min = 15
    elif interval == "30m":
        step_min = 30
    elif interval in ["1h", "60m"]:
        step_min = 60
    else:
        step_min = 5
        
    now = pd.Timestamp.now().floor(f"{step_min}min")
    timestamps = [now - pd.Timedelta(minutes=(n_bars - 1 - i) * step_min) for i in range(n_bars)]
    
    dt = (step_min / 390.0) / 252.0
    bar_vol = annual_volatility * np.sqrt(dt)
    
    log_returns = np.zeros(n_bars)
    for i in range(1, n_bars):
        ts = timestamps[i]
        minute_of_day = ts.hour * 60 + ts.minute
        dist_from_open = abs(minute_of_day - 570)
        dist_from_close = abs(minute_of_day - 960)
        edge_dist = min(dist_from_open, dist_from_close)
        u_factor = np.clip(1.8 - (edge_dist / 180.0), 0.6, 2.5)
        
        effective_vol = bar_vol * u_factor
        log_returns[i] = rng.normal(0, effective_vol)
        
    cum_returns = np.exp(np.cumsum(log_returns))
    # Scale series so the final closing price exactly matches anchor_price
    closes = anchor_price * (cum_returns / cum_returns[-1])
    
    opens = np.zeros(n_bars)
    highs = np.zeros(n_bars)
    lows = np.zeros(n_bars)
    volumes = np.zeros(n_bars, dtype=int)
    
    for i in range(n_bars):
        ts = timestamps[i]
        minute_of_day = ts.hour * 60 + ts.minute
        edge_dist = min(abs(minute_of_day - 570), abs(minute_of_day - 960))
        u_factor = np.clip(1.8 - (edge_dist / 180.0), 0.6, 2.5)
        effective_vol = bar_vol * u_factor
        
        if i == 0:
            open_p = closes[0]
        else:
            open_p = closes[i-1] * (1.0 + rng.normal(0, effective_vol * 0.2))
            
        close_p = closes[i]
        high_spread = abs(rng.normal(0, effective_vol * 0.8))
        low_spread = abs(rng.normal(0, effective_vol * 0.8))
        high_p = max(open_p, close_p) * (1.0 + high_spread)
        low_p = min(open_p, close_p) * (1.0 - low_spread)
        
        base_vol = rng.lognormal(mean=11.0, sigma=0.5)
        ret = abs((close_p - open_p) / open_p) if open_p > 0 else 0.0
        vol_val = int(base_vol * u_factor * (1.0 + 8.0 * ret))
        
        opens[i] = open_p
        highs[i] = high_p
        lows[i] = low_p
        volumes[i] = max(100, vol_val)
        
    df = pd.DataFrame({
        "Open": np.round(opens, 2),
        "High": np.round(highs, 2),
        "Low": np.round(lows, 2),
        "Close": np.round(closes, 2),
        "Adj Close": np.round(closes, 2),
        "Volume": volumes
    }, index=pd.DatetimeIndex(timestamps))
    df.index.name = "Date"
    return df
