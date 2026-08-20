"""Algorithmic Trade Execution Planner with Volume-Based Stop-Loss, Volume-Based Profit Targets, and Kelly sizing based on Trader Dale's Volume Profile Methodology."""

from dataclasses import dataclass
from typing import Optional, Any
import numpy as np
import pandas as pd

from stock_predictor.analytics.support_resistance import SupportResistanceLevels
from stock_predictor.analytics.regime import MarketRegimeInfo


@dataclass
class TradePlan:
    action: str                 # "ACCUMULATE", "BUY LIMIT", "BREAKOUT BUY", "HOLD / MONITOR", "SCALE OUT", "SELL / SHORT"
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    stop_loss_pct: float        # Negative value (e.g., -2.85%)
    target_1: float
    target_1_return_pct: float
    target_2: float
    target_2_return_pct: float
    risk_reward_ratio: float    # e.g., 2.40 (Reward per unit Risk)
    var_95_pct: float           # 95% Value at Risk
    var_99_pct: float           # 99% Value at Risk
    kelly_size_pct: float       # Suggested portfolio allocation % (0% - 25%)
    execution_strategy: str
    volume_setup_name: Optional[str] = None
    is_virgin_level: bool = True


class AlgorithmicTradePlanner:
    """Computes actionable quantitative trade execution parameters and risk envelopes with Volume Profile confluence."""

    def generate_plan(
        self,
        current_price: float,
        predicted_price: float,
        predicted_return_pct: float,
        direction_prob: float,
        lower_bound_price: float,
        upper_bound_price: float,
        levels: SupportResistanceLevels,
        regime: MarketRegimeInfo,
        recent_df: Optional[pd.DataFrame] = None,
        volume_profile: Optional[Any] = None
    ) -> TradePlan:
        # 1. Estimate True Range / Volatility
        if recent_df is not None and len(recent_df) >= 5:
            tr = pd.concat([
                recent_df["High"] - recent_df["Low"],
                (recent_df["High"] - recent_df["Close"].shift(1)).abs(),
                (recent_df["Low"] - recent_df["Close"].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = float(tr.tail(14).mean())
        else:
            atr = current_price * 0.02

        atr_pct = (atr / current_price) * 100.0

        # 2. Value at Risk (VaR)
        daily_vol_pct = atr_pct * 0.70
        var_95 = round(daily_vol_pct * 1.645 * np.sqrt(max(1.0, regime.risk_multiplier)), 2)
        var_99 = round(daily_vol_pct * 2.326 * np.sqrt(max(1.0, regime.risk_multiplier)), 2)

        # 3. Check for Active Trader Dale Volume Profile Setup
        active_setup = None
        if volume_profile and volume_profile.detected_setups:
            active_setup = volume_profile.detected_setups[0]

        # 4. Determine Directional Bias & Action
        is_bullish = predicted_return_pct > 0.4 and direction_prob >= 0.52
        is_bearish = predicted_return_pct < -0.4 and direction_prob <= 0.48

        # Volume Profile Reference Levels
        poc = volume_profile.poc_price if volume_profile else levels.pivot_point
        vah = volume_profile.vah_price if volume_profile else levels.resistance_1
        val = volume_profile.val_price if volume_profile else levels.support_1
        hvns = volume_profile.hvns if volume_profile else [poc]

        setup_name = active_setup.name if active_setup else None
        is_virgin = not active_setup.is_tested if active_setup else True

        if is_bullish:
            if active_setup and active_setup.bias == "BULLISH":
                action = "ACCUMULATE" if active_setup.confidence >= 85 else "BUY LIMIT"
                exec_strat = (
                    f"Trader Dale {active_setup.name}: Place limit bids at volume cluster (${active_setup.entry_level:.2f}). "
                    f"Volume-based Stop-Loss positioned behind institutional defense zone (${active_setup.stop_loss_level:.2f})."
                )
                entry_low = round(min(current_price * 0.995, active_setup.entry_level * 0.996), 2)
                entry_high = round(max(current_price, active_setup.entry_level * 1.004), 2)
                raw_sl = active_setup.stop_loss_level
                t1 = active_setup.target_level
            elif predicted_return_pct >= 2.5 and direction_prob >= 0.65:
                action = "ACCUMULATE"
                exec_strat = "Scale in with multi-tranche limit orders within the designated Volume Value Area."
                entry_low = round(min(current_price * 0.992, max(levels.support_1, val)), 2)
                entry_high = round(max(current_price, entry_low + (current_price * 0.005)), 2)
                raw_sl = min(entry_low * 0.99, val - (atr * 0.5), levels.support_1 - (atr * 0.5))
                t1 = round(max(levels.resistance_1, vah, current_price * 1.015), 2)
            elif current_price >= levels.resistance_1 * 0.995:
                action = "BREAKOUT BUY"
                exec_strat = "Enter on confirmed volume breakout above resistance/VAH with trailing stop."
                entry_low = round(min(current_price, levels.resistance_1), 2)
                entry_high = round(max(current_price * 1.005, entry_low + (current_price * 0.005)), 2)
                raw_sl = min(entry_low * 0.99, levels.support_1 - (atr * 0.5))
                t1 = round(max(levels.resistance_2, predicted_price * 0.98), 2)
            else:
                action = "BUY LIMIT"
                exec_strat = "Place patient limit bids near immediate POC/Support volume node."
                entry_low = round(min(current_price * 0.992, max(levels.support_1, poc * 0.995)), 2)
                entry_high = round(max(current_price, entry_low + (current_price * 0.005)), 2)
                raw_sl = min(entry_low * 0.99, levels.support_1 - (atr * 0.5))
                t1 = round(max(levels.resistance_1, current_price * 1.015, predicted_price * 0.95), 2)

            # Enforce Volume-Based Dynamic Stop-Loss Boundaries
            sl = round(min(current_price * 0.992, max(raw_sl, current_price * 0.88)), 2)
            sl_pct = round(((sl - current_price) / current_price) * 100.0, 2)

            t2 = round(max(levels.resistance_2, upper_bound_price, t1 * 1.02), 2)
            t1_pct = round(((t1 - current_price) / current_price) * 100.0, 2)
            t2_pct = round(((t2 - current_price) / current_price) * 100.0, 2)

            # Risk-Reward Ratio
            downside_risk = max(0.5, abs(sl_pct))
            upside_reward = max(0.5, t1_pct)
            rrr = round(upside_reward / downside_risk, 2)

            # Half-Kelly Position Sizing: f* = 0.5 * (p - q/b)
            p_win = np.clip(direction_prob, 0.40, 0.85)
            q_lose = 1.0 - p_win
            b_ratio = max(0.8, rrr)
            kelly_raw = max(0.0, 0.5 * (p_win - (q_lose / b_ratio)))
            kelly_pct = round(float(np.clip(kelly_raw * 100.0 * regime.risk_multiplier, 0.0, 25.0)), 1)

        elif is_bearish:
            if active_setup and active_setup.bias == "BEARISH":
                action = "SELL / SHORT" if active_setup.confidence >= 85 else "SCALE OUT"
                exec_strat = (
                    f"Trader Dale {active_setup.name}: Position tactical short/hedge near volume node (${active_setup.entry_level:.2f}). "
                    f"Volume-based Stop-Loss placed above institutional selling barrier (${active_setup.stop_loss_level:.2f})."
                )
                entry_high = round(max(current_price * 1.005, active_setup.entry_level * 1.004), 2)
                entry_low = round(min(current_price, active_setup.entry_level * 0.996), 2)
                raw_sl = active_setup.stop_loss_level
                t1 = active_setup.target_level
            else:
                action = "SCALE OUT / SHORT" if predicted_return_pct <= -2.0 else "REDUCE / HEDGE"
                exec_strat = "Trim long exposure or establish tactical short/put hedges on relief bounces near VAH/Resistance."
                entry_high = round(max(current_price * 1.008, min(levels.resistance_1, vah)), 2)
                entry_low = round(min(current_price, entry_high - (current_price * 0.005)), 2)
                raw_sl = max(entry_high * 1.01, levels.resistance_1 + (atr * 0.5), vah + (atr * 0.5))
                t1 = round(min(levels.support_1, val, current_price * 0.985), 2)

            sl = round(max(current_price * 1.008, min(raw_sl, current_price * 1.12)), 2)
            sl_pct = round(((sl - current_price) / current_price) * 100.0, 2)

            t2 = round(min(levels.support_2, lower_bound_price, t1 * 0.98), 2)
            t1_pct = round(((t1 - current_price) / current_price) * 100.0, 2)
            t2_pct = round(((t2 - current_price) / current_price) * 100.0, 2)

            downside_risk = max(0.5, abs(sl_pct))
            upside_reward = max(0.5, abs(t1_pct))
            rrr = round(upside_reward / downside_risk, 2)

            p_win_short = np.clip(1.0 - direction_prob, 0.40, 0.85)
            q_lose_short = 1.0 - p_win_short
            b_ratio_short = max(0.8, rrr)
            kelly_raw_short = max(0.0, 0.5 * (p_win_short - (q_lose_short / b_ratio_short)))
            kelly_pct = round(float(np.clip(kelly_raw_short * 100.0 * regime.risk_multiplier, 0.0, 15.0)), 1)

        else:
            action = "HOLD / MONITOR"
            exec_strat = "Market is rotating within Volume Value Area [VAL, VAH]; await directional breakout before deploying risk."
            entry_low = round(current_price * 0.995, 2)
            entry_high = round(current_price * 1.005, 2)
            sl = round(min(current_price * 0.985, val), 2)
            sl_pct = round(((sl - current_price) / current_price) * 100.0, 2)
            t1 = round(max(current_price * 1.015, vah), 2)
            t2 = round(max(current_price * 1.03, levels.resistance_2), 2)
            t1_pct = round(((t1 - current_price) / current_price) * 100.0, 2)
            t2_pct = round(((t2 - current_price) / current_price) * 100.0, 2)
            rrr = 1.0
            kelly_pct = 0.0

        return TradePlan(
            action=action,
            entry_zone_low=entry_low,
            entry_zone_high=entry_high,
            stop_loss=sl,
            stop_loss_pct=sl_pct,
            target_1=t1,
            target_1_return_pct=t1_pct,
            target_2=t2,
            target_2_return_pct=t2_pct,
            risk_reward_ratio=rrr,
            var_95_pct=var_95,
            var_99_pct=var_99,
            kelly_size_pct=kelly_pct,
            execution_strategy=exec_strat,
            volume_setup_name=setup_name,
            is_virgin_level=is_virgin
        )
