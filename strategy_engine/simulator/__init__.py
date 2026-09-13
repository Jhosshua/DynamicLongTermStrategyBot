"""
strategy_engine.simulator
~~~~~~~~~~~~~~~~~~~~~~~~~

Synthetic Market Regime Simulator and historical stress scenarios.
"""

from strategy_engine.simulator.regime_sde import (
    AssetJumpDiffusionParams,
    MertonJumpDiffusionSimulator,
    PiecewiseRegimeSchedule,
    RegimeParameters,
    make_positive_semidefinite,
)
from strategy_engine.simulator.stress_scenarios import (
    StressScenarioDataset,
    StressScenarioType,
    generate_2008_liquidity_crisis,
    generate_2017_low_vol_bull,
    generate_2020_flash_crash,
    generate_2022_inflation_grind,
    generate_stress_scenario,
)

__all__ = [
    "AssetJumpDiffusionParams",
    "MertonJumpDiffusionSimulator",
    "PiecewiseRegimeSchedule",
    "RegimeParameters",
    "StressScenarioDataset",
    "StressScenarioType",
    "generate_2008_liquidity_crisis",
    "generate_2017_low_vol_bull",
    "generate_2020_flash_crash",
    "generate_2022_inflation_grind",
    "generate_stress_scenario",
    "make_positive_semidefinite",
]
