"""Dynamic Support and Resistance level calculation combining Volume Profile (POC/VAH/VAL/HVNs), Kernel Density Estimation (KDE), Classical Pivots, and Fibonacci retracements."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class SupportResistanceLevels:
    current_price: float
    support_1: float
    support_2: float
    resistance_1: float
    resistance_2: float
    pivot_point: float
    breakout_level: float
    breakdown_level: float
    fib_382: float
    fib_500: float
    fib_618: float
    nearest_level_distance_pct: float
    nearest_level_type: str    # "SUPPORT" or "RESISTANCE"
    # Volume Profile Confluence Levels
    poc_level: Optional[float] = None
    vah_level: Optional[float] = None
    val_level: Optional[float] = None
    volume_nodes: Optional[List[float]] = None


class SupportResistanceEngine:
    """Calculates quantitative structural price levels, Volume Profile confluence, pivot points, and Fibonacci retracements."""

    def calculate_levels(
        self,
        df: pd.DataFrame,
        volume_profile: Optional[Any] = None
    ) -> SupportResistanceLevels:
        if len(df) < 5:
            p = float(df["Close"].iloc[-1]) if len(df) > 0 else 100.0
            return SupportResistanceLevels(
                current_price=p,
                support_1=round(p * 0.97, 2),
                support_2=round(p * 0.94, 2),
                resistance_1=round(p * 1.03, 2),
                resistance_2=round(p * 1.06, 2),
                pivot_point=p,
                breakout_level=round(p * 1.035, 2),
                breakdown_level=round(p * 0.965, 2),
                fib_382=round(p * 0.98, 2),
                fib_500=p,
                fib_618=round(p * 1.02, 2),
                nearest_level_distance_pct=3.0,
                nearest_level_type="SUPPORT",
                poc_level=p,
                vah_level=round(p * 1.02, 2),
                val_level=round(p * 0.98, 2),
                volume_nodes=[p]
            )

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        cur_price = float(close.iloc[-1])

        # 1. Swing High & Swing Low over recent window
        lookback = min(60, len(df))
        recent_high = float(high.tail(lookback).max())
        recent_low = float(low.tail(lookback).min())
        recent_range = max(recent_high - recent_low, cur_price * 0.02)

        # 2. Classical Pivot Points
        last_h = float(high.iloc[-1])
        last_l = float(low.iloc[-1])
        last_c = float(close.iloc[-1])
        
        pivot = (last_h + last_l + last_c) / 3.0
        r1_raw = (2.0 * pivot) - last_l
        s1_raw = (2.0 * pivot) - last_h
        r2_raw = pivot + (last_h - last_l)
        s2_raw = pivot - (last_h - last_l)

        # 3. Fibonacci Retracement Levels
        fib_382 = recent_high - (recent_range * 0.382)
        fib_500 = recent_high - (recent_range * 0.500)
        fib_618 = recent_high - (recent_range * 0.618)

        # 4. Kernel Density Estimation / Local Extremes Clustering
        price_nodes = pd.concat([high.tail(lookback), low.tail(lookback), close.tail(lookback)]).values
        hist, bin_edges = np.histogram(price_nodes, bins=min(15, len(price_nodes) // 3))
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        dense_levels = list(centers[hist >= np.percentile(hist, 60)])

        # 5. Volume Profile Confluence Integration
        poc_val = None
        vah_val = None
        val_val = None
        vp_nodes_list: List[float] = []

        if volume_profile is not None:
            poc_val = float(volume_profile.poc_price)
            vah_val = float(volume_profile.vah_price)
            val_val = float(volume_profile.val_price)
            vp_nodes_list = list(volume_profile.hvns)
            # Add POC, VAH, VAL and HVNs into structural density candidates
            dense_levels.extend([poc_val, vah_val, val_val])
            dense_levels.extend(volume_profile.hvns)

        dense_levels = np.array(dense_levels)
        supports = [lvl for lvl in dense_levels if lvl < cur_price * 0.998]
        resistances = [lvl for lvl in dense_levels if lvl > cur_price * 1.002]

        s1 = max(supports) if supports else min(s1_raw, cur_price * 0.98)
        s2 = min(supports) if len(supports) > 1 else min(s2_raw, s1 * 0.97)

        r1 = min(resistances) if resistances else max(r1_raw, cur_price * 1.02)
        r2 = max(resistances) if len(resistances) > 1 else max(r2_raw, r1 * 1.03)

        # Enforce strict hierarchy: s2 < s1 < cur_price < r1 < r2
        s1 = min(s1, cur_price * 0.995)
        s2 = min(s2, s1 * 0.985)
        r1 = max(r1, cur_price * 1.005)
        r2 = max(r2, r1 * 1.015)

        # Dynamic Breakout / Breakdown triggers
        tr = high.iloc[-1] - low.iloc[-1]
        atr_buffer = max(tr * 0.5, cur_price * 0.005)
        breakout = r1 + atr_buffer
        breakdown = s1 - atr_buffer

        # Distance to nearest level
        dist_s = abs((cur_price - s1) / cur_price) * 100.0
        dist_r = abs((r1 - cur_price) / cur_price) * 100.0

        if dist_s <= dist_r:
            nearest_dist = dist_s
            nearest_type = "SUPPORT"
        else:
            nearest_dist = dist_r
            nearest_type = "RESISTANCE"

        return SupportResistanceLevels(
            current_price=round(cur_price, 2),
            support_1=round(s1, 2),
            support_2=round(s2, 2),
            resistance_1=round(r1, 2),
            resistance_2=round(r2, 2),
            pivot_point=round(pivot, 2),
            breakout_level=round(breakout, 2),
            breakdown_level=round(breakdown, 2),
            fib_382=round(fib_382, 2),
            fib_500=round(fib_500, 2),
            fib_618=round(fib_618, 2),
            nearest_level_distance_pct=round(nearest_dist, 2),
            nearest_level_type=nearest_type,
            poc_level=round(poc_val, 2) if poc_val is not None else None,
            vah_level=round(vah_val, 2) if vah_val is not None else None,
            val_level=round(val_val, 2) if val_val is not None else None,
            volume_nodes=[round(n, 2) for n in vp_nodes_list] if vp_nodes_list else None
        )
