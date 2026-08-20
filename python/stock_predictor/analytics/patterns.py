"""Vectorized Candlestick & Technical Indicator Pattern Recognition."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class DetectedPattern:
    name: str                   # e.g., "RSI Bullish Divergence"
    category: str               # "MOMENTUM", "VOLATILITY", "CANDLESTICK", "TREND"
    bias: str                   # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float           # [0, 100]
    description: str


class PatternDetector:
    """Detects multi-timeframe structural chart patterns and technical indicator divergences."""

    def detect_patterns(self, df: pd.DataFrame) -> List[DetectedPattern]:
        if len(df) < 15:
            return []

        patterns: List[DetectedPattern] = []
        close = df["Close"]
        open_p = df["Open"]
        high = df["High"]
        low = df["Low"]
        vol = df["Volume"].astype(float)

        cur_close = float(close.iloc[-1])
        cur_open = float(open_p.iloc[-1])
        cur_high = float(high.iloc[-1])
        cur_low = float(low.iloc[-1])
        cur_vol = float(vol.iloc[-1])

        # 1. Multi-Period RSI Calculation for Divergence
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=3).mean()
        avg_loss = loss.rolling(14, min_periods=3).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        cur_rsi = float(rsi.iloc[-1])

        # RSI Divergence Detection over last 20 bars
        if len(df) >= 20:
            sub_p = close.tail(20)
            sub_rsi = rsi.tail(20)
            p_min_idx = sub_p.idxmin()
            rsi_min_idx = sub_rsi.idxmin()

            # Bullish Divergence: Recent price lower than past low, but RSI is higher
            if cur_close <= sub_p.min() * 1.01 and cur_rsi > sub_rsi.iloc[0] + 4.0 and cur_rsi < 45.0:
                patterns.append(DetectedPattern(
                    name="RSI Bullish Divergence",
                    category="MOMENTUM",
                    bias="BULLISH",
                    confidence=78.0,
                    description=f"Price is testing local lows while RSI ({cur_rsi:.1f}) is forming higher lows, signaling exhausting selling pressure."
                ))

            # Bearish Divergence: Price higher than past high, but RSI lower
            if cur_close >= sub_p.max() * 0.99 and cur_rsi < sub_rsi.max() - 4.0 and cur_rsi > 60.0:
                patterns.append(DetectedPattern(
                    name="RSI Bearish Divergence",
                    category="MOMENTUM",
                    bias="BEARISH",
                    confidence=76.0,
                    description=f"Price pushed higher while RSI ({cur_rsi:.1f}) printed a lower high, indicating momentum deceleration."
                ))

        # 2. Moving Average Crosses (EMA 20 & EMA 50)
        if len(df) >= 30:
            ema_20 = close.ewm(span=20, adjust=False).mean()
            ema_50 = close.ewm(span=50, adjust=False).mean()
            diff = ema_20 - ema_50
            if diff.iloc[-1] > 0 and diff.iloc[-2] <= 0:
                patterns.append(DetectedPattern(
                    name="Bullish EMA Golden Cross (20/50)",
                    category="TREND",
                    bias="BULLISH",
                    confidence=82.0,
                    description="20 EMA has crossed above the 50 EMA, initiating a medium-term upward trend acceleration."
                ))
            elif diff.iloc[-1] < 0 and diff.iloc[-2] >= 0:
                patterns.append(DetectedPattern(
                    name="Bearish EMA Death Cross (20/50)",
                    category="TREND",
                    bias="BEARISH",
                    confidence=80.0,
                    description="20 EMA has crossed below the 50 EMA, indicating increasing downward structural pressure."
                ))

        # 3. Bollinger Band Squeeze / Breakout
        if len(df) >= 20:
            bb_mid = close.rolling(20, min_periods=5).mean()
            bb_std = close.rolling(20, min_periods=5).std()
            bb_upper = bb_mid + 2.0 * bb_std
            bb_lower = bb_mid - 2.0 * bb_std
            bandwidth = (bb_upper - bb_lower) / (bb_mid + 1e-8)
            cur_bw = float(bandwidth.iloc[-1])
            hist_bw = bandwidth.tail(min(50, len(bandwidth)))

            if cur_bw <= hist_bw.quantile(0.15):
                patterns.append(DetectedPattern(
                    name="Bollinger Volatility Squeeze",
                    category="VOLATILITY",
                    bias="NEUTRAL",
                    confidence=85.0,
                    description="Bandwidth is in the lowest 15th percentile of historical range, signaling an imminent explosive directional expansion."
                ))
            elif cur_close > float(bb_upper.iloc[-1]) and cur_vol > float(vol.tail(20).mean()) * 1.3:
                patterns.append(DetectedPattern(
                    name="Bollinger Upper Band Breakout",
                    category="VOLATILITY",
                    bias="BULLISH",
                    confidence=79.0,
                    description="Price broke above upper Bollinger Band on elevated volume, confirming momentum continuation."
                ))

        # 4. Candlestick Formations
        body = abs(cur_close - cur_open)
        bar_range = cur_high - cur_low + 1e-8
        lower_shadow = min(cur_close, cur_open) - cur_low
        upper_shadow = cur_high - max(cur_close, cur_open)

        # Hammer / Bullish Pin Bar
        if lower_shadow >= (2.0 * body) and upper_shadow <= (0.25 * bar_range) and cur_close > cur_low:
            patterns.append(DetectedPattern(
                name="Bullish Hammer / Pin Bar",
                category="CANDLESTICK",
                bias="BULLISH",
                confidence=72.0,
                description=f"Long lower shadow ({(lower_shadow/bar_range)*100:.0f}% of candle) demonstrates aggressive intra-period buyer rejection at lows."
            ))

        # Shooting Star / Bearish Pin Bar
        if upper_shadow >= (2.0 * body) and lower_shadow <= (0.25 * bar_range):
            patterns.append(DetectedPattern(
                name="Bearish Shooting Star / Pin Bar",
                category="CANDLESTICK",
                bias="BEARISH",
                confidence=70.0,
                description="Long upper shadow indicates sellers strongly rejected higher prices into the close."
            ))

        # Bullish Engulfing
        if len(df) >= 2:
            prev_close = float(close.iloc[-2])
            prev_open = float(open_p.iloc[-2])
            if prev_close < prev_open and cur_close > cur_open and cur_close >= prev_open and cur_open <= prev_close:
                patterns.append(DetectedPattern(
                    name="Bullish Engulfing Pattern",
                    category="CANDLESTICK",
                    bias="BULLISH",
                    confidence=75.0,
                    description="Current green candle completely engulfs previous bearish candle, signaling a shift in control to buyers."
                ))

        # 5. Volume Climax / Surge
        vol_mean_20 = float(vol.tail(20).mean()) if len(df) >= 20 else cur_vol
        if cur_vol >= vol_mean_20 * 2.2:
            patterns.append(DetectedPattern(
                name="Institutional Volume Surge",
                category="MOMENTUM",
                bias="BULLISH" if cur_close > cur_open else "BEARISH",
                confidence=84.0,
                description=f"Trading volume spiked {cur_vol/vol_mean_20:.1f}x above the 20-period average, indicating high institutional participation."
            ))

        return patterns
