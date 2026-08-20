"""Vectorized Technical Indicators and Price-Derived Quantitative Signals (Daily and Intraday)."""

import numpy as np
import pandas as pd


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive suite of technical indicators and statistical signals.
    Supports both daily and intraday (1m, 5m, 15m, 1h) OHLCV series.
    """
    feats = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_p = df["Open"]
    vol = df["Volume"].astype(float)
    
    # 1. Historical Log Returns (Stationary across multiple bar lags)
    for lag in [1, 2, 3, 5, 10, 20]:
        ret_val = np.log(close / (close.shift(lag) + 1e-8))
        feats[f"return_{lag}d"] = ret_val
        feats[f"return_{lag}b"] = ret_val

    # 2. Overnight Gap vs Intraday Return
    overnight_gap = np.log(open_p / (close.shift(1) + 1e-8))
    intraday_ret = np.log(close / (open_p + 1e-8))
    feats["overnight_gap"] = overnight_gap
    feats["intraday_return"] = intraday_ret
    feats["gap_momentum_interaction"] = overnight_gap * intraday_ret.shift(1)
        
    # 3. Moving Averages & Moving Average Ratios
    for span in [5, 10, 20, 50]:
        sma = close.rolling(window=span, min_periods=3).mean()
        feats[f"sma_{span}"] = (close - sma) / (sma + 1e-8)
        
    for span in [5, 9, 13, 21, 50]:
        ema = close.ewm(span=span, adjust=False).mean()
        feats[f"ema_{span}"] = (close - ema) / (ema + 1e-8)
        
    # Moving Average Crosses (Short / Medium)
    sma_20 = close.rolling(window=20, min_periods=5).mean()
    sma_50 = close.rolling(window=50, min_periods=10).mean()
    feats["ma_20_50_ratio"] = (sma_20 - sma_50) / (sma_50 + 1e-8)
    
    # 4. Multi-Period Wilder's RSI (7, 14, 21) & Stochastic RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    for rsi_period in [7, 14, 21]:
        avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi_val = 100.0 - (100.0 / (1.0 + rs))
        feats[f"rsi_{rsi_period}"] = rsi_val / 100.0  # Normalize to [0, 1]
        
    # Stochastic RSI (14-period)
    rsi_14 = feats["rsi_14"]
    rsi_14_low = rsi_14.rolling(window=14, min_periods=5).min()
    rsi_14_high = rsi_14.rolling(window=14, min_periods=5).max()
    feats["stoch_rsi"] = (rsi_14 - rsi_14_low) / ((rsi_14_high - rsi_14_low) + 1e-8)
    
    # 5. MACD (Moving Average Convergence Divergence)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    feats["macd_line_norm"] = macd_line / (close + 1e-8)
    feats["macd_signal_norm"] = signal_line / (close + 1e-8)
    feats["macd_hist_norm"] = macd_hist / (close + 1e-8)
    
    # 6. Bollinger Bands (20 period, 2 std)
    bb_mid = close.rolling(window=20, min_periods=5).mean()
    bb_std = close.rolling(window=20, min_periods=5).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    
    feats["bb_pct_b"] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-8)
    feats["bb_bandwidth"] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
    
    # 7. Volatility, ATR, Garman-Klass & Parkinson Extreme-Value Estimators
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = true_range.rolling(window=14, min_periods=5).mean()
    
    feats["natr_14"] = atr_14 / (close + 1e-8)  # Normalized ATR (%)
    feats["hist_vol_10d"] = feats["return_1d"].rolling(window=10, min_periods=5).std() * np.sqrt(252)
    feats["hist_vol_30d"] = feats["return_1d"].rolling(window=30, min_periods=5).std() * np.sqrt(252)
    
    # Garman-Klass volatility estimator
    log_hl = np.log((high + 1e-8) / (low + 1e-8))
    log_co = np.log((close + 1e-8) / (open_p + 1e-8))
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    feats["garman_klass_vol_20"] = np.sqrt(np.maximum(0, gk_var.rolling(window=20, min_periods=5).mean())) * np.sqrt(252)
    
    # Parkinson volatility estimator
    park_var = (log_hl ** 2) / (4.0 * np.log(2.0))
    feats["parkinson_vol_20"] = np.sqrt(np.maximum(0, park_var.rolling(window=20, min_periods=5).mean())) * np.sqrt(252)
    
    # 8. Stochastic Oscillator (%K and %D) & Williams %R
    low_14 = low.rolling(window=14, min_periods=5).min()
    high_14 = high.rolling(window=14, min_periods=5).max()
    stoch_k = 100.0 * (close - low_14) / ((high_14 - low_14) + 1e-8)
    stoch_d = stoch_k.rolling(window=3, min_periods=1).mean()
    feats["stoch_k"] = stoch_k / 100.0
    feats["stoch_d"] = stoch_d / 100.0
    
    # Williams %R
    feats["williams_r_14"] = ((high_14 - close) / ((high_14 - low_14) + 1e-8)) * -1.0
    
    # 9. Rate of Change (ROC) and Micro-Momentum
    feats["roc_3"] = (close - close.shift(3)) / (close.shift(3) + 1e-8)
    feats["roc_5"] = (close - close.shift(5)) / (close.shift(5) + 1e-8)
    feats["roc_6"] = (close - close.shift(6)) / (close.shift(6) + 1e-8)
    feats["roc_10"] = (close - close.shift(10)) / (close.shift(10) + 1e-8)
    feats["roc_12"] = (close - close.shift(12)) / (close.shift(12) + 1e-8)
    
    # 10. Intraday VWAP (Volume-Weighted Average Price)
    typical_price = (high + low + close) / 3.0
    cum_vol = vol.cumsum() + 1e-8
    cum_tp_vol = (typical_price * vol).cumsum()
    vwap = cum_tp_vol / cum_vol
    feats["vwap_ratio"] = (close - vwap) / (vwap + 1e-8)
    
    # 11. Volume Dynamics, CMF & VPT
    vol_sma_20 = vol.rolling(window=20, min_periods=5).mean()
    feats["volume_ratio_20"] = vol / (vol_sma_20 + 1e-8)
    
    obv_direction = np.sign(close.diff().fillna(0))
    obv = (obv_direction * vol).cumsum()
    obv_sma = obv.rolling(window=20, min_periods=5).mean()
    feats["obv_trend"] = (obv - obv_sma) / (obv.abs().rolling(window=20, min_periods=5).mean() + 1e-8)
    
    # Chaikin Money Flow (CMF 20)
    clv = ((close - low) - (high - close)) / ((high - low) + 1e-8)
    vol_sum_20 = vol.rolling(window=20, min_periods=5).sum() + 1e-8
    feats["cmf_20"] = (clv * vol).rolling(window=20, min_periods=5).sum() / vol_sum_20
    
    # 12. High-Low-Close Intra-bar Price Dynamics
    feats["high_low_spread"] = (high - low) / (open_p + 1e-8)
    feats["close_open_spread"] = (close - open_p) / (open_p + 1e-8)
    feats["upper_shadow"] = (high - np.maximum(close, open_p)) / (open_p + 1e-8)
    feats["lower_shadow"] = (np.minimum(close, open_p) - low) / (open_p + 1e-8)
    
    # 13. Rolling Z-Scores and Higher Statistical Moments
    roll_mean_20 = close.rolling(window=20, min_periods=5).mean()
    roll_std_20 = close.rolling(window=20, min_periods=5).std()
    feats["zscore_20d"] = (close - roll_mean_20) / (roll_std_20 + 1e-8)
    
    # Rolling skewness of 1-day returns
    feats["skew_20d"] = feats["return_1d"].rolling(window=20, min_periods=10).skew().fillna(0.0)
    
    # 14. SuperTrend Volatility & Regime Filter (ATR 10, Multiplier 3.0)
    atr_10 = true_range.rolling(window=10, min_periods=3).mean()
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + (3.0 * atr_10)
    basic_lower = hl2 - (3.0 * atr_10)
    supertrend = pd.Series(index=df.index, dtype=float)
    supertrend_dir = pd.Series(index=df.index, dtype=float)
    
    # Vectorized iterative SuperTrend computation
    st_val = close.iloc[0] if len(close) > 0 else 0.0
    st_dir = 1.0
    for idx, (c_val, bu_val, bl_val) in enumerate(zip(close, basic_upper, basic_lower)):
        if np.isnan(bu_val) or np.isnan(bl_val):
            supertrend.iloc[idx] = c_val
            supertrend_dir.iloc[idx] = 1.0
            continue
        if st_dir == 1.0:
            st_val = max(bl_val, st_val) if c_val >= st_val else bu_val
            st_dir = 1.0 if c_val >= st_val else -1.0
        else:
            st_val = min(bu_val, st_val) if c_val <= st_val else bl_val
            st_dir = -1.0 if c_val <= st_val else 1.0
        supertrend.iloc[idx] = st_val
        supertrend_dir.iloc[idx] = st_dir
        
    feats["supertrend_dir"] = supertrend_dir
    feats["supertrend_dist"] = (close - supertrend) / (close + 1e-8)
    
    # 15. TTM Volatility Squeeze (Bollinger Bands inside Keltner Channel)
    kc_mid = close.rolling(window=20, min_periods=5).mean()
    kc_range = atr_14.rolling(window=20, min_periods=5).mean()
    kc_upper = kc_mid + (1.5 * kc_range)
    kc_lower = kc_mid - (1.5 * kc_range)
    
    # Squeeze is ON when BB is completely inside KC (volatility compression)
    squeeze_on = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(float)
    feats["ttm_squeeze_on"] = squeeze_on
    
    # Squeeze momentum (delta from mid price)
    hl_mid = (high.rolling(window=20, min_periods=5).max() + low.rolling(window=20, min_periods=5).min()) / 2.0
    squeeze_mid = (hl_mid + kc_mid) / 2.0
    feats["ttm_squeeze_momentum"] = (close - squeeze_mid) / (close + 1e-8)
    
    # 16. Multi-Period EMA Ribbon (9, 21, 34, 55, 89)
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_34 = close.ewm(span=34, adjust=False).mean()
    ema_55 = close.ewm(span=55, adjust=False).mean()
    ema_89 = close.ewm(span=89, adjust=False).mean()
    
    ribbon_bullish = ((ema_9 > ema_21) & (ema_21 > ema_34) & (ema_34 > ema_55) & (ema_55 > ema_89)).astype(float)
    ribbon_bearish = ((ema_9 < ema_21) & (ema_21 < ema_34) & (ema_34 < ema_55) & (ema_55 < ema_89)).astype(float)
    feats["ema_ribbon_alignment"] = ribbon_bullish - ribbon_bearish
    feats["ema_ribbon_spread"] = (ema_9 - ema_89) / (close + 1e-8)
    
    # 17. Donchian Breakout Channels (Turtle 20 & 55 Channels)
    donchian_20_high = high.rolling(window=20, min_periods=5).max()
    donchian_20_low = low.rolling(window=20, min_periods=5).min()
    donchian_55_high = high.rolling(window=55, min_periods=10).max()
    donchian_55_low = low.rolling(window=55, min_periods=10).min()
    
    feats["donchian_20_pos"] = (close - donchian_20_low) / ((donchian_20_high - donchian_20_low) + 1e-8)
    feats["donchian_55_pos"] = (close - donchian_55_low) / ((donchian_55_high - donchian_55_low) + 1e-8)
    feats["donchian_breakout_20"] = ((close >= donchian_20_high.shift(1)).astype(float) - (close <= donchian_20_low.shift(1)).astype(float))
    
    # 18. Volume Price Trend (VPT) & Relative Volume (RVOL)
    price_pct_change = close.pct_change().fillna(0.0)
    vpt = (price_pct_change * vol).cumsum()
    vpt_sma = vpt.rolling(window=20, min_periods=5).mean()
    feats["vpt_norm"] = (vpt - vpt_sma) / (vpt.abs().rolling(window=20, min_periods=5).mean() + 1e-8)
    feats["rvol_20"] = vol / (vol_sma_20 + 1e-8)
    
    # 19. Hull Moving Average (HMA 14 - low lag trend tracking)
    half_period = 7
    full_period = 14
    sqrt_period = int(np.sqrt(full_period))
    wma_half = close.ewm(span=half_period, adjust=False).mean()
    wma_full = close.ewm(span=full_period, adjust=False).mean()
    diff_wma = (2.0 * wma_half) - wma_full
    hma_14 = diff_wma.ewm(span=sqrt_period, adjust=False).mean()
    feats["hma_14_ratio"] = (close - hma_14) / (close + 1e-8)
    
    # 20. Vectorized Volume Profile Features (Trader Dale Methodology)
    # Rolling Volume-Weighted Distribution & Point of Control (POC) Proxy
    roll_w = 20
    vp_vwap = (typical_price * vol).rolling(window=roll_w, min_periods=5).sum() / (vol.rolling(window=roll_w, min_periods=5).sum() + 1e-8)
    vp_vwsd = np.sqrt(
        (((typical_price - vp_vwap) ** 2) * vol).rolling(window=roll_w, min_periods=5).sum() / 
        (vol.rolling(window=roll_w, min_periods=5).sum() + 1e-8)
    )
    
    # Value Area approximation (70% normal distribution ~ 1.04 std deviations)
    vp_vah = vp_vwap + (1.04 * vp_vwsd)
    vp_val = vp_vwap - (1.04 * vp_vwsd)
    
    feats["vp_poc_dist_pct"] = (close - vp_vwap) / (close + 1e-8)
    feats["vp_in_value_area"] = ((close >= vp_val) & (close <= vp_vah)).astype(float)
    feats["vp_val_dist_pct"] = (close - vp_val) / (close + 1e-8)
    feats["vp_vah_dist_pct"] = (close - vp_vah) / (close + 1e-8)
    
    # Volume Profile Shape Proxy: > 0 for P-Shape (Buyer Aggression), < 0 for b-Shape (Seller Aggression)
    vp_range = (vp_vah - vp_val).replace(0, 1e-8)
    feats["vp_shape_skew"] = (vp_vwap - ((vp_vah + vp_val) / 2.0)) / vp_range
    
    # Institutional Volume Accumulation / Absorption Index
    candle_body = (close - open_p)
    candle_range = (high - low).replace(0, 1e-8)
    feats["vp_accumulation_index"] = (candle_body / candle_range) * (vol / (vol_sma_20 + 1e-8))
    
    return feats

