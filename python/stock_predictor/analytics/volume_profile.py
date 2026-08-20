"""Institutional Volume Profile Engine based on Trader Dale's Volume Profile Methodology.

Calculates:
- Point of Control (POC): Price level with highest traded volume.
- Value Area High (VAH) & Value Area Low (VAL): 70% of total volume distribution.
- High Volume Nodes (HVN) & Low Volume Nodes (LVN).
- Profile Morphology: D-profile (balance), P-profile (aggressive buying), b-profile (aggressive selling), Thin profile (trend imbalance).
- Trader Dale's Setups:
    1. Setup #1: Volume Accumulation Setup
    2. Setup #2: Trend Setup
    3. Setup #3: Rejection Setup
    4. Reversal Setup (Support/Resistance Polarity Flip)
- Tested vs. Virgin Level Tracking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd


@dataclass
class VolumeProfileNode:
    price: float
    volume: float
    relative_volume: float   # [0, 1] relative to POC
    is_poc: bool = False
    in_value_area: bool = False
    is_hvn: bool = False
    is_lvn: bool = False


@dataclass
class VolumeSetupSignal:
    name: str                 # e.g., "Volume Accumulation Setup #1 (Bullish)"
    setup_type: str           # "ACCUMULATION", "TREND", "REJECTION", "REVERSAL"
    bias: str                 # "BULLISH", "BEARISH", "NEUTRAL"
    entry_level: float        # Retest entry price (POC / Volume Node)
    stop_loss_level: float    # Volume-based stop loss
    target_level: float       # Volume-based profit target
    confidence: float         # [0, 100]
    description: str
    is_tested: bool = False   # False = Virgin / Untested level (Highest Probability)
    touch_count: int = 0


@dataclass
class VolumeProfileResult:
    poc_price: float
    vah_price: float
    val_price: float
    total_volume: float
    profile_shape: str        # "D_SHAPE", "P_SHAPE", "B_SHAPE", "THIN_PROFILE"
    shape_description: str
    nodes: List[VolumeProfileNode]
    hvns: List[float]         # High volume node price levels
    lvns: List[float]         # Low volume node price levels
    detected_setups: List[VolumeSetupSignal] = field(default_factory=list)
    poc_distance_pct: float = 0.0
    is_in_value_area: bool = True
    virgin_poc_count: int = 0


class VolumeProfileDetector:
    """Calculates quantitative Volume Profile metrics, Value Area distribution, and Trader Dale's setups."""

    def __init__(self, num_bins: int = 40, value_area_pct: float = 0.70):
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct

    def compute_volume_profile(
        self,
        df: pd.DataFrame,
        lookback_bars: Optional[int] = None
    ) -> VolumeProfileResult:
        """Computes Volume Profile, POC, Value Area (70%), and classifies profile morphology."""
        if len(df) < 5:
            p = float(df["Close"].iloc[-1]) if len(df) > 0 else 100.0
            return VolumeProfileResult(
                poc_price=p,
                vah_price=round(p * 1.02, 2),
                val_price=round(p * 0.98, 2),
                total_volume=1000000.0,
                profile_shape="D_SHAPE",
                shape_description="Balanced Normal Distribution",
                nodes=[],
                hvns=[p],
                lvns=[],
                detected_setups=[],
                poc_distance_pct=0.0,
                is_in_value_area=True
            )

        sub_df = df.tail(lookback_bars) if lookback_bars else df
        close = sub_df["Close"].values
        high = sub_df["High"].values
        low = sub_df["Low"].values
        open_p = sub_df["Open"].values
        volume = sub_df["Volume"].astype(float).values

        cur_price = float(close[-1])
        min_price = float(np.min(low))
        max_price = float(np.max(high))
        
        # Ensure non-zero price range
        if max_price <= min_price:
            max_price = min_price * 1.01

        # 1. Price Bin Discretization
        bin_edges = np.linspace(min_price, max_price, self.num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_volumes = np.zeros(self.num_bins, dtype=float)

        # 2. Volume-at-Price Distribution (distribute bar volume across overlapping price bins)
        for h, l, c, o, v in zip(high, low, close, open_p, volume):
            if v <= 0:
                continue
            bar_range = max(h - l, 1e-6)
            # Find bins overlapping with [l, h]
            overlap_mask = (bin_edges[1:] >= l) & (bin_edges[:-1] <= h)
            if not np.any(overlap_mask):
                idx = np.clip(np.searchsorted(bin_edges, c) - 1, 0, self.num_bins - 1)
                bin_volumes[idx] += v
            else:
                overlap_indices = np.where(overlap_mask)[0]
                # Weight volume towards close and open for realistic micro-structure
                weights = np.ones(len(overlap_indices))
                for i_idx, b_idx in enumerate(overlap_indices):
                    b_center = bin_centers[b_idx]
                    # Higher density near intra-bar trading body
                    dist_to_body = min(abs(b_center - c), abs(b_center - o))
                    weights[i_idx] = 1.0 / (1.0 + (dist_to_body / bar_range))
                weights /= np.sum(weights)
                for i_idx, b_idx in enumerate(overlap_indices):
                    bin_volumes[b_idx] += v * weights[i_idx]

        total_vol = float(np.sum(bin_volumes))
        if total_vol <= 0:
            total_vol = 1.0

        # 3. Point of Control (POC) Calculation
        poc_idx = int(np.argmax(bin_volumes))
        poc_price = float(bin_centers[poc_idx])
        max_bin_vol = float(bin_volumes[poc_idx])

        # 4. Value Area Calculation (70% standard expansion outward from POC)
        target_va_vol = total_vol * self.value_area_pct
        va_vol = max_bin_vol
        va_low_idx = poc_idx
        va_high_idx = poc_idx

        while va_vol < target_va_vol and (va_low_idx > 0 or va_high_idx < self.num_bins - 1):
            next_low_vol = bin_volumes[va_low_idx - 1] if va_low_idx > 0 else -1.0
            next_high_vol = bin_volumes[va_high_idx + 1] if va_high_idx < self.num_bins - 1 else -1.0

            if next_high_vol >= next_low_vol and next_high_vol >= 0:
                va_high_idx += 1
                va_vol += next_high_vol
            elif next_low_vol >= 0:
                va_low_idx -= 1
                va_vol += next_low_vol
            else:
                break

        val_price = float(bin_centers[va_low_idx])
        vah_price = float(bin_centers[va_high_idx])

        # 5. Identify High Volume Nodes (HVN) and Low Volume Nodes (LVN)
        mean_vol = np.mean(bin_volumes)
        hvns: List[float] = []
        lvns: List[float] = []

        for i in range(1, self.num_bins - 1):
            v_curr = bin_volumes[i]
            v_prev = bin_volumes[i - 1]
            v_next = bin_volumes[i + 1]
            
            # Local peak with volume > 1.25x mean
            if v_curr > v_prev and v_curr > v_next and v_curr > mean_vol * 1.2:
                hvns.append(round(float(bin_centers[i]), 2))
            # Local trough with volume < 0.65x mean
            elif v_curr < v_prev and v_curr < v_next and v_curr < mean_vol * 0.7:
                lvns.append(round(float(bin_centers[i]), 2))

        if not hvns:
            hvns.append(round(poc_price, 2))

        # 6. Profile Morphology Classification (Trader Dale's 4 Shapes)
        poc_relative_pos = (poc_idx / (self.num_bins - 1))  # 0.0 at low, 1.0 at high
        upper_vol = np.sum(bin_volumes[self.num_bins // 2:])
        lower_vol = np.sum(bin_volumes[:self.num_bins // 2])
        vol_skew = (upper_vol - lower_vol) / total_vol

        # Check for thin profile (trend imbalance across bins)
        zero_or_thin_bins = np.sum(bin_volumes < mean_vol * 0.4)
        is_thin = zero_or_thin_bins >= (self.num_bins * 0.55)

        if poc_relative_pos >= 0.55 and vol_skew > 0.10:
            shape = "P_SHAPE"
            shape_desc = "P-Profile: Aggressive buyers / institutional accumulation at highs or short covering."
        elif poc_relative_pos <= 0.45 and vol_skew < -0.10:
            shape = "B_SHAPE"
            shape_desc = "b-Profile: Aggressive sellers / institutional distribution at lows or long liquidation."
        elif is_thin:
            shape = "THIN_PROFILE"
            shape_desc = "Thin Profile: Strong trend day / market imbalance with rapid price transit."
        else:
            shape = "D_SHAPE"
            shape_desc = "D-Profile: Balanced bell-curve market rotation and range accumulation."


        # 7. Construct Nodes
        nodes: List[VolumeProfileNode] = []
        for i in range(self.num_bins):
            p_node = float(bin_centers[i])
            v_node = float(bin_volumes[i])
            rel_v = float(v_node / max_bin_vol) if max_bin_vol > 0 else 0.0
            
            nodes.append(VolumeProfileNode(
                price=round(p_node, 2),
                volume=round(v_node, 2),
                relative_volume=round(rel_v, 3),
                is_poc=(i == poc_idx),
                in_value_area=(va_low_idx <= i <= va_high_idx),
                is_hvn=(round(p_node, 2) in hvns),
                is_lvn=(round(p_node, 2) in lvns)
            ))

        poc_dist_pct = round(((cur_price - poc_price) / poc_price) * 100.0, 2)
        in_va = (val_price <= cur_price <= vah_price)

        # 8. Detect Trader Dale's Volume Setups
        setups = self.detect_volume_setups(sub_df, poc_price, vah_price, val_price, hvns, lvns, shape)

        return VolumeProfileResult(
            poc_price=round(poc_price, 2),
            vah_price=round(vah_price, 2),
            val_price=round(val_price, 2),
            total_volume=round(total_vol, 0),
            profile_shape=shape,
            shape_description=shape_desc,
            nodes=nodes,
            hvns=hvns,
            lvns=lvns,
            detected_setups=setups,
            poc_distance_pct=poc_dist_pct,
            is_in_value_area=in_va,
            virgin_poc_count=sum(1 for s in setups if not s.is_tested)
        )

    def detect_volume_setups(
        self,
        df: pd.DataFrame,
        poc_price: float,
        vah_price: float,
        val_price: float,
        hvns: List[float],
        lvns: List[float],
        profile_shape: str
    ) -> List[VolumeSetupSignal]:
        """
        Detects Trader Dale's core Volume Profile setups:
        1. Volume Accumulation Setup (#1)
        2. Trend Setup (#2)
        3. Rejection Setup (#3)
        4. Reversal Setup (Support/Resistance Polarity Flip)
        """
        if len(df) < 10:
            return []

        setups: List[VolumeSetupSignal] = []
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        open_p = df["Open"]
        vol = df["Volume"].astype(float)

        cur_price = float(close.iloc[-1])
        cur_high = float(high.iloc[-1])
        cur_low = float(low.iloc[-1])
        cur_open = float(open_p.iloc[-1])

        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = float(tr.tail(14).mean()) if len(tr) >= 14 else cur_price * 0.015

        # =========================================================================
        # SETUP #1: VOLUME ACCUMULATION SETUP (Trader Dale Setup #1)
        # Consolidation/Accumulation Box -> Impulsive Breakout -> Retest of POC
        # =========================================================================
        if len(df) >= 15:
            past_window = df.iloc[-25:-3] if len(df) >= 25 else df.iloc[:-3]
            if len(past_window) >= 8:
                breakout_up = cur_price > float(past_window["High"].max()) and float(close.iloc[-2]) >= float(past_window["High"].max())
                breakout_down = cur_price < float(past_window["Low"].min()) and float(close.iloc[-2]) <= float(past_window["Low"].min())
                
                dist_to_poc_pct = abs((cur_price - poc_price) / cur_price) * 100.0
                
                if (breakout_up or profile_shape == "P_SHAPE") and dist_to_poc_pct <= 2.2 and cur_price >= poc_price * 0.995:
                    sl_level = round(min(poc_price - (atr * 0.8), val_price - (atr * 0.3)), 2)
                    pt_level = round(max(cur_price + (atr * 2.0), vah_price + (atr * 1.0)), 2)
                    
                    touches = int(np.sum((low.iloc[-5:] <= poc_price * 1.002) & (high.iloc[-5:] >= poc_price * 0.998)))
                    is_virgin = (touches <= 1)
                    
                    setups.append(VolumeSetupSignal(
                        name="Volume Accumulation Setup #1 (Bullish)",
                        setup_type="ACCUMULATION",
                        bias="BULLISH",
                        entry_level=round(poc_price, 2),
                        stop_loss_level=sl_level,
                        target_level=pt_level,
                        confidence=88.0 if is_virgin else 74.0,
                        description=(
                            f"Institutions accumulated heavy volume at POC (${poc_price:.2f}). "
                            f"Price broke out upward and is currently retesting the accumulation cluster. "
                            f"{'Virgin untested level: High probability first touch.' if is_virgin else f'Tested {touches}x.'}"
                        ),
                        is_tested=not is_virgin,
                        touch_count=touches
                    ))

                elif (breakout_down or profile_shape == "B_SHAPE") and dist_to_poc_pct <= 2.2 and cur_price <= poc_price * 1.005:
                    sl_level = round(max(poc_price + (atr * 0.8), vah_price + (atr * 0.3)), 2)
                    pt_level = round(min(cur_price - (atr * 2.0), val_price - (atr * 1.0)), 2)
                    
                    touches = int(np.sum((low.iloc[-5:] <= poc_price * 1.002) & (high.iloc[-5:] >= poc_price * 0.998)))
                    is_virgin = (touches <= 1)
                    
                    setups.append(VolumeSetupSignal(
                        name="Volume Accumulation Setup #1 (Bearish)",
                        setup_type="ACCUMULATION",
                        bias="BEARISH",
                        entry_level=round(poc_price, 2),
                        stop_loss_level=sl_level,
                        target_level=pt_level,
                        confidence=87.0 if is_virgin else 73.0,
                        description=(
                            f"Institutions distributed heavy volume at POC (${poc_price:.2f}). "
                            f"Price broke out downward and is retesting the institutional selling zone. "
                            f"{'Virgin untested level: High probability first touch.' if is_virgin else f'Tested {touches}x.'}"
                        ),
                        is_tested=not is_virgin,
                        touch_count=touches
                    ))

        # =========================================================================
        # SETUP #2: TREND SETUP (Trader Dale Setup #2)
        # Strong Trend -> In-trend HVN Volume Cluster -> Retest of in-trend POC
        # =========================================================================
        if len(df) >= 12:
            ema_20 = close.ewm(span=20, adjust=False).mean()
            ema_50 = close.ewm(span=50, adjust=False).mean()
            is_uptrend = float(ema_20.iloc[-1]) > float(ema_50.iloc[-1]) and cur_price > float(ema_50.iloc[-1])
            is_downtrend = float(ema_20.iloc[-1]) < float(ema_50.iloc[-1]) and cur_price < float(ema_50.iloc[-1])
            
            for hvn in hvns:
                hvn_dist_pct = abs((cur_price - hvn) / cur_price) * 100.0
                if hvn_dist_pct <= 1.8:
                    if is_uptrend and cur_price >= hvn * 0.996:
                        sl = round(hvn - (atr * 0.7), 2)
                        pt = round(cur_price + (atr * 1.8), 2)
                        setups.append(VolumeSetupSignal(
                            name="Trend Setup #2 (In-Trend Long Pullback)",
                            setup_type="TREND",
                            bias="BULLISH",
                            entry_level=round(hvn, 2),
                            stop_loss_level=sl,
                            target_level=pt,
                            confidence=84.0,
                            description=(
                                f"Active uptrend with strong in-trend volume cluster at ${hvn:.2f}. "
                                f"Institutions added to long positions during the trend. Retest of this HVN provides low-risk continuation entry."
                            ),
                            is_tested=False,
                            touch_count=1
                        ))
                        break
                    elif is_downtrend and cur_price <= hvn * 1.004:
                        sl = round(hvn + (atr * 0.7), 2)
                        pt = round(cur_price - (atr * 1.8), 2)
                        setups.append(VolumeSetupSignal(
                            name="Trend Setup #2 (In-Trend Short Pullback)",
                            setup_type="TREND",
                            bias="BEARISH",
                            entry_level=round(hvn, 2),
                            stop_loss_level=sl,
                            target_level=pt,
                            confidence=83.0,
                            description=(
                                f"Active downtrend with strong in-trend selling volume cluster at ${hvn:.2f}. "
                                f"Institutions added to short positions during trend pause. Retest offers high-conviction continuation."
                            ),
                            is_tested=False,
                            touch_count=1
                        ))
                        break

        # =========================================================================
        # SETUP #3: REJECTION SETUP (Trader Dale Setup #3)
        # Aggressive Price Wick Rejection + Heavy Volume Climax Node
        # =========================================================================
        if len(df) >= 5:
            for i in range(-1, -min(4, len(df)), -1):
                b_high = float(high.iloc[i])
                b_low = float(low.iloc[i])
                b_close = float(close.iloc[i])
                b_open = float(open_p.iloc[i])
                b_vol = float(vol.iloc[i])
                avg_vol = float(vol.tail(15).mean())
                
                b_range = max(b_high - b_low, 1e-6)
                lower_wick = min(b_close, b_open) - b_low
                upper_wick = b_high - max(b_close, b_open)
                
                # Bullish Rejection: Lower wick >= 50% of candle + high volume
                if lower_wick >= (0.50 * b_range) and b_vol >= (avg_vol * 1.15):
                    rejection_zone = round(b_low + (lower_wick * 0.5), 2)
                    dist_to_rej = abs((cur_price - rejection_zone) / cur_price) * 100.0
                    if dist_to_rej <= 2.5:
                        sl = round(b_low - (atr * 0.4), 2)
                        pt = round(cur_price + (atr * 2.2), 2)
                        setups.append(VolumeSetupSignal(
                            name="Rejection Setup #3 (Aggressive Buyer Absorption)",
                            setup_type="REJECTION",
                            bias="BULLISH",
                            entry_level=rejection_zone,
                            stop_loss_level=sl,
                            target_level=pt,
                            confidence=86.0,
                            description=(
                                f"Strong price rejection at ${b_low:.2f} on {b_vol/avg_vol:.1f}x average volume. "
                                f"Aggressive buyers absorbed selling pressure and established institutional defense at ${rejection_zone:.2f}."
                            ),
                            is_tested=False,
                            touch_count=1
                        ))
                        break

                # Bearish Rejection: Upper wick >= 50% of candle + high volume
                elif upper_wick >= (0.50 * b_range) and b_vol >= (avg_vol * 1.15):
                    rejection_zone = round(b_high - (upper_wick * 0.5), 2)
                    dist_to_rej = abs((cur_price - rejection_zone) / cur_price) * 100.0
                    if dist_to_rej <= 2.5:
                        sl = round(b_high + (atr * 0.4), 2)
                        pt = round(cur_price - (atr * 2.2), 2)
                        setups.append(VolumeSetupSignal(
                            name="Rejection Setup #3 (Aggressive Seller Absorption)",
                            setup_type="REJECTION",
                            bias="BEARISH",
                            entry_level=rejection_zone,
                            stop_loss_level=sl,
                            target_level=pt,
                            confidence=85.0,
                            description=(
                                f"Violent rejection of higher prices at ${b_high:.2f} on {b_vol/avg_vol:.1f}x volume. "
                                f"Aggressive institutional sellers pushed price back, forming heavy resistance at ${rejection_zone:.2f}."
                            ),
                            is_tested=False,
                            touch_count=1
                        ))
                        break

        # =========================================================================
        # REVERSAL SETUP: SUPPORT/RESISTANCE POLARITY FLIP AT VOLUME NODES
        # Breached heavy volume zone flips polarity
        # =========================================================================
        if len(df) >= 10:
            for lvn in lvns:
                if abs((cur_price - lvn) / cur_price) * 100.0 <= 1.2:
                    setups.append(VolumeSetupSignal(
                        name="Volume Reversal / Polarity Flip",
                        setup_type="REVERSAL",
                        bias="BULLISH" if cur_price > lvn else "BEARISH",
                        entry_level=round(lvn, 2),
                        stop_loss_level=round(lvn - (atr * 0.8) if cur_price > lvn else lvn + (atr * 0.8), 2),
                        target_level=round(cur_price + (atr * 1.5) if cur_price > lvn else cur_price - (atr * 1.5), 2),
                        confidence=78.0,
                        description=(
                            f"Low Volume Node at ${lvn:.2f} marks structural transition boundary. "
                            f"Decisive breakout turns prior resistance into active support."
                        ),
                        is_tested=False,
                        touch_count=1
                    ))
                    break

        return setups
