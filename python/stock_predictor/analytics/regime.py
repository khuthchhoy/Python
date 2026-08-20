"""Market & Asset Regime Detection Engine for trend structure, volatility states, and macro beta."""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd


@dataclass
class MarketRegimeInfo:
    trend_regime: str           # "STRONG_BULLISH", "MODERATE_BULLISH", "SIDEWAYS_CHOP", "MODERATE_BEARISH", "STRONG_BEARISH"
    volatility_regime: str      # "LOW_VOLATILITY_COMPRESSION", "NORMAL_VOLATILITY", "HIGH_VOLATILITY_EXPANSION", "EXTREME_VOLATILITY_PANIC"
    relative_strength_regime: str # "MARKET_OUTPERFORMER", "MARKET_IN_LINE", "MARKET_UNDERPERFORMER"
    regime_summary: str
    risk_multiplier: float      # e.g., 0.75 for panic, 1.25 for clear bull trend
    adx_proxy: float            # Trend strength [0, 100]
    trend_direction: str        # "UPTREND", "DOWNTREND", "RANGE"
    volatility_percentile: float # [0, 100]


class MarketRegimeDetector:
    """Detects multi-dimensional market and asset regimes from price and cross-asset dynamics."""

    def detect_regime(
        self,
        df: pd.DataFrame,
        benchmarks: Optional[Dict[str, pd.DataFrame]] = None
    ) -> MarketRegimeInfo:
        if len(df) < 15:
            return MarketRegimeInfo(
                trend_regime="SIDEWAYS_CHOP",
                volatility_regime="NORMAL_VOLATILITY",
                relative_strength_regime="MARKET_IN_LINE",
                regime_summary="Insufficient historical bars for deep regime classification.",
                risk_multiplier=1.0,
                adx_proxy=20.0,
                trend_direction="RANGE",
                volatility_percentile=50.0
            )

        close = df["Close"]
        cur_price = float(close.iloc[-1])
        
        # 1. Moving Averages & Trend
        span_short = min(10, max(3, len(df) // 4))
        span_med = min(20, max(5, len(df) // 2))
        span_long = min(50, len(df) - 1)

        ema_fast = close.ewm(span=span_short, adjust=False).mean()
        ema_med = close.ewm(span=span_med, adjust=False).mean()
        sma_long = close.rolling(span_long, min_periods=max(5, span_long // 2)).mean()

        cur_fast = float(ema_fast.iloc[-1])
        cur_med = float(ema_med.iloc[-1])
        cur_long = float(sma_long.iloc[-1])

        # Trend slope & ADX proxy (normalized difference of moving averages + directional movement)
        pct_from_med = ((cur_price - cur_med) / (cur_med + 1e-8)) * 100.0
        pct_from_long = ((cur_price - cur_long) / (cur_long + 1e-8)) * 100.0
        ma_alignment = 1 if (cur_price > cur_fast > cur_med > cur_long) else (-1 if (cur_price < cur_fast < cur_med < cur_long) else 0)

        # 20-period Directional Movement proxy
        lookback_regime = min(30, len(df))
        rolling_ret = np.log(close / close.shift(1)).tail(lookback_regime)
        pos_moves = rolling_ret[rolling_ret > 0].sum()
        neg_moves = abs(rolling_ret[rolling_ret < 0].sum()) + 1e-8
        trend_ratio = (pos_moves - neg_moves) / (pos_moves + neg_moves) # [-1, 1]
        adx_proxy = float(np.clip((abs(trend_ratio) * 60.0 + abs(pct_from_med) * 5.0), 10.0, 95.0))

        # Trend Regime Classification
        if ma_alignment == 1 and pct_from_med > 1.0:
            trend_regime = "STRONG_BULLISH"
            trend_dir = "UPTREND"
        elif cur_price > cur_med and pct_from_long >= 0:
            trend_regime = "MODERATE_BULLISH"
            trend_dir = "UPTREND"
        elif ma_alignment == -1 and pct_from_med < -1.0:
            trend_regime = "STRONG_BEARISH"
            trend_dir = "DOWNTREND"
        elif cur_price < cur_med and pct_from_long < 0:
            trend_regime = "MODERATE_BEARISH"
            trend_dir = "DOWNTREND"
        else:
            trend_regime = "SIDEWAYS_CHOP"
            trend_dir = "RANGE"

        # 2. Volatility Regime Classification
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - close.shift(1)).abs()
        tr3 = (df["Low"] - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        natr = (tr / (close + 1e-8)) * 100.0
        cur_natr = float(natr.iloc[-1])
        
        hist_natr = natr.tail(min(90, len(natr)))
        vol_pctile = float(np.clip((cur_natr - hist_natr.min()) / ((hist_natr.max() - hist_natr.min()) + 1e-8) * 100.0, 0.0, 100.0))

        # Bollinger Band squeeze check
        bb_mid = close.rolling(20, min_periods=5).mean()
        bb_std = close.rolling(20, min_periods=5).std()
        bandwidth = (bb_std * 2.0) / (bb_mid + 1e-8)
        cur_bandwidth = float(bandwidth.iloc[-1])
        hist_bandwidth = bandwidth.tail(min(60, len(bandwidth)))
        is_squeeze = cur_bandwidth <= hist_bandwidth.quantile(0.20)

        if is_squeeze or vol_pctile < 25.0:
            vol_regime = "LOW_VOLATILITY_COMPRESSION"
            vol_multiplier = 1.15  # Good entry environment
        elif vol_pctile > 85.0 or cur_natr > 4.5:
            vol_regime = "EXTREME_VOLATILITY_PANIC"
            vol_multiplier = 0.65  # Reduce sizing
        elif vol_pctile > 65.0:
            vol_regime = "HIGH_VOLATILITY_EXPANSION"
            vol_multiplier = 0.85
        else:
            vol_regime = "NORMAL_VOLATILITY"
            vol_multiplier = 1.0

        # 3. Relative Strength vs Market Benchmark (SPY)
        rel_regime = "MARKET_IN_LINE"
        if benchmarks:
            spy_key = next((k for k in benchmarks if "SPY" in k.upper() or "GSPC" in k.upper()), None)
            if spy_key and len(benchmarks[spy_key]) > 5:
                spy_df = benchmarks[spy_key].reindex(df.index).ffill().bfill()
                stock_ret_20 = (cur_price / float(close.iloc[-min(20, len(close))]) - 1.0) * 100.0
                spy_close = spy_df["Close"]
                spy_ret_20 = (float(spy_close.iloc[-1]) / float(spy_close.iloc[-min(20, len(spy_close))]) - 1.0) * 100.0
                alpha_20 = stock_ret_20 - spy_ret_20
                if alpha_20 > 3.0:
                    rel_regime = "MARKET_OUTPERFORMER"
                elif alpha_20 < -3.0:
                    rel_regime = "MARKET_UNDERPERFORMER"

        # Construct Narrative Summary
        summary = (
            f"Asset is currently exhibiting a {trend_regime.replace('_', ' ').title()} trend structure "
            f"with {vol_regime.replace('_', ' ').lower()} (volatility at {vol_pctile:.0f}th percentile). "
            f"Relative strength vs broad market is classified as {rel_regime.replace('_', ' ').lower()}."
        )

        return MarketRegimeInfo(
            trend_regime=trend_regime,
            volatility_regime=vol_regime,
            relative_strength_regime=rel_regime,
            regime_summary=summary,
            risk_multiplier=round(vol_multiplier, 2),
            adx_proxy=round(adx_proxy, 1),
            trend_direction=trend_dir,
            volatility_percentile=round(vol_pctile, 1)
        )
