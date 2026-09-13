"""
strategy_engine.simulator.stress_scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibrated historical market stress scenarios:
1. 2008 Liquidity Crisis (vol ~75%, equity -55%, credit freeze, safe haven rally)
2. 2020 Flash Crash (-35% drawdown in 23 days, rapid V-recovery off trough)
3. 2022 Inflation Grind (stock-bond positive correlation +0.70, bond duration failure, cash outperforming)
4. 2017 Low-Vol Bull (vol ~8%, steady compounding, max DD < 4%)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Union
import pandas as pd

from strategy_engine.core.models import Bar
from strategy_engine.simulator.regime_sde import MertonJumpDiffusionSimulator


class StressScenarioType(str, Enum):
    CRISIS_2008 = "2008_liquidity_crisis"
    FLASH_CRASH_2020 = "2020_flash_crash"
    INFLATION_GRIND_2022 = "2022_inflation_grind"
    LOW_VOL_BULL_2017 = "2017_low_vol_bull"


@dataclass(frozen=True)
class StressScenarioDataset:
    """Container holding scenario metadata, simulated bars, and target metrics."""
    scenario_type: StressScenarioType
    description: str
    bars: Dict[str, List[Bar]]
    benchmark_symbol: str = "SPY"
    expected_benchmark_drawdown: float = -0.55
    expected_benchmark_volatility: float = 0.50
    target_strategy_max_dd: float = 0.15

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all bars to a sorted, indexed pandas DataFrame."""
        flat_records = []
        for sym, bar_list in self.bars.items():
            for b in bar_list:
                flat_records.append({
                    "symbol": b.symbol,
                    "timestamp": b.timestamp,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "trade_count": b.trade_count,
                    "vwap": b.vwap,
                })
        df = pd.DataFrame(flat_records)
        if not df.empty:
            df.sort_values(by=["timestamp", "symbol"], inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df


def generate_2008_liquidity_crisis(seed: int = 42) -> Dict[str, List[Bar]]:
    """Generate calibrated 252-day 2008 Liquidity Crisis scenario bars."""
    sim = MertonJumpDiffusionSimulator(seed=seed)
    return sim.generate_2008_crisis(n_days=252, seed=seed)


def generate_2020_flash_crash(seed: int = 42) -> Dict[str, List[Bar]]:
    """Generate calibrated 60-day 2020 Flash Crash scenario bars (23 crash, 37 recovery)."""
    sim = MertonJumpDiffusionSimulator(seed=seed)
    return sim.generate_2020_flash_crash(n_days=60, seed=seed)


def generate_2022_inflation_grind(seed: int = 42) -> Dict[str, List[Bar]]:
    """Generate calibrated 252-day 2022 Inflation Grind scenario bars (+0.70 stock-bond corr)."""
    sim = MertonJumpDiffusionSimulator(seed=seed)
    return sim.generate_2022_inflation_grind(n_days=252, seed=seed)


def generate_2017_low_vol_bull(seed: int = 42) -> Dict[str, List[Bar]]:
    """Generate calibrated 252-day 2017 Low-Vol Bull scenario bars (vol ~8%, max DD < 4%)."""
    sim = MertonJumpDiffusionSimulator(seed=seed)
    return sim.generate_2017_low_vol_bull(n_days=252, seed=seed)


def generate_stress_scenario(
    scenario_type: Union[StressScenarioType, str],
    seed: int = 42,
    start_date: Optional[datetime] = None,
) -> StressScenarioDataset:
    """Generate scenario dataset wrapper containing bars, metadata, and DataFrame export."""
    if isinstance(scenario_type, str):
        # Allow loose string matching
        type_str = scenario_type.lower().replace("-", "_")
        if "2008" in type_str:
            scen_type = StressScenarioType.CRISIS_2008
        elif "2020" in type_str:
            scen_type = StressScenarioType.FLASH_CRASH_2020
        elif "2022" in type_str:
            scen_type = StressScenarioType.INFLATION_GRIND_2022
        elif "2017" in type_str:
            scen_type = StressScenarioType.LOW_VOL_BULL_2017
        else:
            scen_type = StressScenarioType(scenario_type)
    else:
        scen_type = scenario_type

    if scen_type == StressScenarioType.CRISIS_2008:
        bars = generate_2008_liquidity_crisis(seed=seed)
        desc = "2008 Liquidity Crisis (vol ~75%, SPY -55%, credit freeze, safe-haven flight)"
        exp_dd = -0.55
        exp_vol = 0.50
        max_dd = 0.115
    elif scen_type == StressScenarioType.FLASH_CRASH_2020:
        bars = generate_2020_flash_crash(seed=seed)
        desc = "2020 Flash Crash (-35% drawdown in 23 days, rapid V-recovery off trough)"
        exp_dd = -0.35
        exp_vol = 0.80
        max_dd = 0.098
    elif scen_type == StressScenarioType.INFLATION_GRIND_2022:
        bars = generate_2022_inflation_grind(seed=seed)
        desc = "2022 Inflation Grind (stock-bond positive correlation +0.70, bond duration failure)"
        exp_dd = -0.22
        exp_vol = 0.24
        max_dd = 0.065
    elif scen_type == StressScenarioType.LOW_VOL_BULL_2017:
        bars = generate_2017_low_vol_bull(seed=seed)
        desc = "2017 Low-Vol Bull (vol ~8%, steady compounding, max DD < 4%)"
        exp_dd = -0.028
        exp_vol = 0.08
        max_dd = 0.025
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return StressScenarioDataset(
        scenario_type=scen_type,
        description=desc,
        bars=bars,
        expected_benchmark_drawdown=exp_dd,
        expected_benchmark_volatility=exp_vol,
        target_strategy_max_dd=max_dd,
    )
