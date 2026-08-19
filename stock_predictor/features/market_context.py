"""Market context and cross-asset features (SPY benchmark, VIX volatility regime)."""

from typing import Dict
import numpy as np
import pandas as pd


def calculate_market_context_features(
    target_df: pd.DataFrame,
    benchmarks: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate cross-asset and market regime features.
    - Beta to SPY
    - Relative strength vs SPY
    - Rolling correlation with SPY
    - Market trend regime (SPY > 20d SMA)
    - VIX volatility regime and changes
    """
    feats = pd.DataFrame(index=target_df.index)
    target_ret = np.log(target_df["Close"] / (target_df["Close"].shift(1) + 1e-8))
    
    # Process SPY (S&P 500 Market Benchmark)
    spy_key = None
    for k in benchmarks:
        if "SPY" in k.upper() or "GSPC" in k.upper():
            spy_key = k
            break
            
    if spy_key and spy_key in benchmarks and len(benchmarks[spy_key]) > 5:
        spy_raw = benchmarks[spy_key]
        spy_df = spy_raw.reindex(target_df.index).ffill().bfill()
        spy_close = spy_df["Close"]
        spy_ret = np.log(spy_close / (spy_close.shift(1) + 1e-8))
        
        # 1-day, 5-day, 20-day relative strength vs market
        feats["spy_ret_1d"] = spy_ret.fillna(0.0)
        feats["spy_ret_5d"] = np.log(spy_close / (spy_close.shift(5) + 1e-8)).fillna(0.0)
        feats["rel_strength_5d"] = (np.log(target_df["Close"] / (target_df["Close"].shift(5) + 1e-8)) - feats["spy_ret_5d"]).fillna(0.0)
        feats["rel_strength_20d"] = (np.log(target_df["Close"] / (target_df["Close"].shift(20) + 1e-8)) - np.log(spy_close / (spy_close.shift(20) + 1e-8))).fillna(0.0)
        
        # Rolling 60-day Beta to SPY
        cov_60 = target_ret.rolling(60, min_periods=15).cov(spy_ret)
        var_spy_60 = spy_ret.rolling(60, min_periods=15).var()
        feats["beta_spy_60d"] = (cov_60 / (var_spy_60 + 1e-8)).fillna(1.0).clip(-3.0, 4.0)
        
        # Rolling correlation with SPY
        feats["corr_spy_60d"] = target_ret.rolling(60, min_periods=15).corr(spy_ret).fillna(0.5).clip(-1.0, 1.0)
        
        # SPY Market Trend Regime (Is SPY in an uptrend above its 20d SMA?)
        spy_sma_20 = spy_close.rolling(20, min_periods=5).mean()
        feats["spy_trend_regime"] = ((spy_close - spy_sma_20) / (spy_sma_20 + 1e-8)).fillna(0.0)
    else:
        # Default placeholders if SPY not supplied or single asset
        feats["spy_ret_1d"] = 0.0
        feats["spy_ret_5d"] = 0.0
        feats["rel_strength_5d"] = 0.0
        feats["rel_strength_20d"] = 0.0
        feats["beta_spy_60d"] = 1.0
        feats["corr_spy_60d"] = 0.5
        feats["spy_trend_regime"] = 0.0
        
    # Process VIX (Volatility Index)
    vix_key = None
    for k in benchmarks:
        if "VIX" in k.upper():
            vix_key = k
            break
            
    if vix_key and vix_key in benchmarks and len(benchmarks[vix_key]) > 5:
        vix_raw = benchmarks[vix_key]
        vix_df = vix_raw.reindex(target_df.index).ffill().bfill()
        vix_close = vix_df["Close"]
        
        # VIX level normalized (e.g. 20 -> 0.20)
        feats["vix_level"] = (vix_close / 100.0).fillna(0.18)
        feats["vix_change_5d"] = ((vix_close - vix_close.shift(5)) / (vix_close.shift(5) + 1e-8)).fillna(0.0)
        
        # VIX Regime: high volatility flag (>25)
        feats["vix_high_regime"] = (vix_close > 25.0).astype(float).fillna(0.0)
    else:
        feats["vix_level"] = 0.18
        feats["vix_change_5d"] = 0.0
        feats["vix_high_regime"] = 0.0
        
    return feats
