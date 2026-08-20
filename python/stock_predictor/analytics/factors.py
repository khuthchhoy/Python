"""Multi-Factor Quantitative Scoring Engine (Momentum, Trend, Volatility, Flow, Composite)."""

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd


@dataclass
class FactorScores:
    momentum_score: float       # [0, 100]
    trend_score: float          # [0, 100]
    volatility_score: float     # [0, 100] (Higher = more stable/favorable)
    flow_score: float           # [0, 100] (Institutional volume & money flow)
    composite_score: float      # [0, 100] (Overall quantitative score)
    verdict: str                # "EXCEPTIONAL", "FAVORABLE", "NEUTRAL", "UNFAVORABLE", "EXTREME_RISK"


class QuantitativeFactorScorer:
    """Calculates multi-factor quantitative scores based on vectorized indicators and price action."""

    def compute_factor_scores(self, df: pd.DataFrame) -> FactorScores:
        if len(df) < 10:
            return FactorScores(
                momentum_score=50.0,
                trend_score=50.0,
                volatility_score=50.0,
                flow_score=50.0,
                composite_score=50.0,
                verdict="NEUTRAL"
            )

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        vol = df["Volume"].astype(float)
        cur_p = float(close.iloc[-1])

        # 1. Momentum Factor Score (Wilder's RSI + Multi-period returns)
        ret_5 = (cur_p / float(close.iloc[-min(5, len(close))]) - 1.0) * 100.0
        ret_20 = (cur_p / float(close.iloc[-min(20, len(close))]) - 1.0) * 100.0
        
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # Wilder's Exponential Smoothing
        avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        cur_rsi = float(rsi.iloc[-1])

        raw_mom = (ret_5 * 2.0 + ret_20 + (cur_rsi - 50.0))
        mom_score = float(np.clip(50.0 + raw_mom * 2.2, 5.0, 95.0))

        # 2. Trend Quality Factor Score
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        sma_50 = close.rolling(min(50, len(close)), min_periods=5).mean()

        cur_e9 = float(ema_9.iloc[-1])
        cur_e21 = float(ema_21.iloc[-1])
        cur_s50 = float(sma_50.iloc[-1])

        trend_pts = 50.0
        if cur_p > cur_e9: trend_pts += 12.0
        if cur_e9 > cur_e21: trend_pts += 15.0
        if cur_e21 > cur_s50: trend_pts += 15.0
        if cur_p < cur_e9: trend_pts -= 12.0
        if cur_e9 < cur_e21: trend_pts -= 15.0
        if cur_e21 < cur_s50: trend_pts -= 15.0
        trend_score = float(np.clip(trend_pts, 5.0, 95.0))

        # 3. Volatility / Risk Factor Score (Higher = cleaner volatility, lower downside tail risk)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        natr_pct = float(((tr / (close + 1e-8)) * 100.0).iloc[-1])
        
        # Lower NATR -> Higher stability score
        vol_score = float(np.clip(95.0 - (natr_pct * 12.0), 10.0, 95.0))

        # 4. Money Flow & Liquidity Factor Score
        hl_spread = (high - low).replace(0, 1e-8)
        clv = ((close - low) - (high - close)) / hl_spread
        clv = np.nan_to_num(clv, nan=0.0)
        vol_window = min(20, len(df))
        cmf_denom = vol.rolling(vol_window, min_periods=3).sum() + 1e-8
        cmf_num = (pd.Series(clv, index=df.index) * vol).rolling(vol_window, min_periods=3).sum()
        cmf_20 = float((cmf_num / cmf_denom).iloc[-1])
        
        flow_pts = 50.0 + (cmf_20 * 120.0)
        flow_score = float(np.clip(flow_pts, 5.0, 95.0))

        # 5. Composite Quantitative Score
        composite = (
            mom_score * 0.35 +
            trend_score * 0.30 +
            vol_score * 0.15 +
            flow_score * 0.20
        )
        composite_score = float(np.clip(composite, 5.0, 95.0))

        if composite_score >= 80.0:
            verdict = "EXCEPTIONAL"
        elif composite_score >= 62.0:
            verdict = "FAVORABLE"
        elif composite_score >= 42.0:
            verdict = "NEUTRAL"
        elif composite_score >= 25.0:
            verdict = "UNFAVORABLE"
        else:
            verdict = "EXTREME_RISK"

        return FactorScores(
            momentum_score=round(mom_score, 1),
            trend_score=round(trend_score, 1),
            volatility_score=round(vol_score, 1),
            flow_score=round(flow_score, 1),
            composite_score=round(composite_score, 1),
            verdict=verdict
        )
