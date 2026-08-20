"""Synthetic Market Data Generator with realistic ticker price anchors and current timestamps."""

from typing import Tuple, Optional, Dict
import hashlib
import time
import numpy as np
import pandas as pd

# Real-world calibrated anchor prices for major market assets
KNOWN_TICKER_PRICES: Dict[str, float] = {
    "DELL": 138.50,
    "NVDA": 128.50,
    "AAPL": 224.50,
    "MSFT": 448.00,
    "TSLA": 215.00,
    "AMZN": 186.00,
    "GOOGL": 165.00,
    "GOOG": 166.00,
    "META": 530.00,
    "SPY": 560.00,
    "QQQ": 480.00,
    "AMD": 145.00,
    "AVGO": 160.00,
    "NFLX": 680.00,
    "PLTR": 32.50,
    "COIN": 210.00,
    "MSTR": 140.00,
    "INTC": 20.50,
    "SMCI": 45.00,
    "LLY": 920.00,
    "JPM": 215.00,
    "BAC": 39.50,
    "WMT": 74.00,
    "COST": 880.00,
    "BABA": 84.00,
    "^VIX": 15.30,
    "^GSPC": 5600.00,
    "^DJI": 40800.00,
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
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    n_days: int = 1200,
    initial_price: Optional[float] = None,
    annual_drift: float = 0.12,
    annual_volatility: float = 0.28,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic daily OHLCV data ending at TODAY's date and the ticker's real-world price anchor.
    """
    rng = np.random.default_rng(seed)
    anchor_price = get_anchor_price_for_ticker(ticker, initial_price)
    
    # Anchor to today's date if end_date is None
    if end_date is None:
        today = pd.Timestamp.now().normalize()
        dates = pd.bdate_range(end=today, periods=n_days)
    else:
        if start_date is None:
            dates = pd.bdate_range(end=pd.Timestamp(end_date), periods=n_days)
        else:
            dates = pd.bdate_range(start=start_date, end=end_date)
            n_days = len(dates)
        
    dt = 1.0 / 252.0  # Daily time-step
    
    # Stochastic volatility (Heston-style process)
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
    Generate high-frequency synthetic OHLCV intraday bars ending at the current time.
    """
    rng = np.random.default_rng(seed)
    anchor_price = get_anchor_price_for_ticker(ticker, initial_price)
    
    now = pd.Timestamp.now().floor("min")
    freq_map = {
        "1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min",
        "30m": "30min", "1h": "1h", "4h": "4h"
    }
    freq = freq_map.get(interval, "5min")
    dates = pd.date_range(end=now, periods=n_bars, freq=freq)
    
    mins_per_bar = {
        "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240
    }.get(interval, 5)
    
    dt = (mins_per_bar / (252.0 * 390.0))
    bar_vol = annual_volatility * np.sqrt(dt)
    
    log_returns = rng.normal(0.00005, bar_vol, size=n_bars)
    
    # Add intraday U-shape volatility curve (higher volatility at market open/close)
    hours = dates.hour + dates.minute / 60.0
    u_shape = 1.0 + 0.8 * np.exp(-((hours - 9.5) / 1.0)**2) + 0.6 * np.exp(-((hours - 16.0) / 1.0)**2)
    log_returns *= u_shape
    
    cum_returns = np.exp(np.cumsum(log_returns))
    closes = anchor_price * (cum_returns / cum_returns[-1])
    
    opens = np.zeros(n_bars)
    highs = np.zeros(n_bars)
    lows = np.zeros(n_bars)
    volumes = np.zeros(n_bars, dtype=int)
    
    opens[0] = closes[0]
    highs[0] = closes[0] * 1.001
    lows[0] = closes[0] * 0.999
    volumes[0] = 5000
    
    for t in range(1, n_bars):
        opens[t] = closes[t-1]
        noise = abs(rng.normal(0, bar_vol * 0.5))
        highs[t] = max(opens[t], closes[t]) * (1.0 + noise)
        lows[t] = min(opens[t], closes[t]) * (1.0 - noise)
        
        base_v = int(rng.lognormal(mean=9.0, sigma=0.6))
        volumes[t] = int(base_v * u_shape[t])
        
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
