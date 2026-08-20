"""Calendar and intraday seasonality features."""

import numpy as np
import pandas as pd


def calculate_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Extract cyclical calendar and intraday time-of-day features.
    Supports daily and intraday (1m, 5m, 15m, 1h) DatetimeIndex.
    """
    feats = pd.DataFrame(index=index)
    
    # 1. Day of week (0=Mon, 4=Fri)
    day_of_week = index.dayofweek
    feats["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 5.0)
    feats["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 5.0)
    
    # 2. Month of year (1-12)
    month = index.month
    feats["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    feats["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    
    # 3. Day of month (1-31)
    day = index.day
    feats["day_of_month_sin"] = np.sin(2 * np.pi * (day - 1) / 31.0)
    feats["day_of_month_cos"] = np.cos(2 * np.pi * (day - 1) / 31.0)
    
    # 4. Intraday Time-of-Day features (Minutes & Hours)
    # Minute of day in [0, 1440]
    minute_of_day = index.hour * 60 + index.minute
    feats["time_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    feats["time_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    
    # Hour of day (0-23)
    feats["hour_sin"] = np.sin(2 * np.pi * index.hour / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * index.hour / 24.0)
    
    # Intraday market regimes (US Market Hours: Open 9:30-10:30 AM, Close 3:00-4:00 PM)
    feats["is_market_open_hour"] = ((index.hour == 9) | (index.hour == 10)).astype(float)
    feats["is_power_hour"] = (index.hour == 15).astype(float)
    
    # Turn of month effect
    feats["is_month_end"] = index.is_month_end.astype(float)
    feats["is_month_start"] = index.is_month_start.astype(float)
    
    return feats
