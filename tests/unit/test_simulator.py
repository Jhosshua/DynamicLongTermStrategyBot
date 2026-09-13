"""
tests/unit/test_simulator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verification tests for the Synthetic Market Regime Simulator across the 4 calibrated
historical stress scenarios:
1. 2008 Liquidity Crisis
2. 2020 Flash Crash
3. 2022 Inflation Grind
4. 2017 Low-Vol Bull
"""

from datetime import datetime, timezone
import math
from typing import Dict, List
import numpy as np
import pytest

from strategy_engine.core.models import Bar
from strategy_engine.simulator.stress_scenarios import (
    StressScenarioType,
    generate_2008_liquidity_crisis,
    generate_2017_low_vol_bull,
    generate_2020_flash_crash,
    generate_2022_inflation_grind,
    generate_stress_scenario,
)


# --- 1. Determinism Verification ---

@pytest.mark.parametrize("generator_func", [
    generate_2008_liquidity_crisis,
    generate_2020_flash_crash,
    generate_2022_inflation_grind,
    generate_2017_low_vol_bull,
])
def test_simulator_determinism(generator_func):
    """Verify that identical seeds produce bit-for-bit identical bars."""
    run_1 = generator_func(seed=42)
    run_2 = generator_func(seed=42)

    assert set(run_1.keys()) == set(run_2.keys())

    for symbol in run_1:
        bars_1 = run_1[symbol]
        bars_2 = run_2[symbol]
        assert len(bars_1) == len(bars_2)

        for b1, b2 in zip(bars_1, bars_2):
            assert b1.timestamp == b2.timestamp
            assert math.isclose(b1.open, b2.open, abs_tol=1e-12)
            assert math.isclose(b1.high, b2.high, abs_tol=1e-12)
            assert math.isclose(b1.low, b2.low, abs_tol=1e-12)
            assert math.isclose(b1.close, b2.close, abs_tol=1e-12)
            assert b1.volume == b2.volume
            assert b1.trade_count == b2.trade_count


def test_simulator_seed_variation():
    """Verify that different seeds produce different price trajectories."""
    run_a = generate_2008_liquidity_crisis(seed=42)
    run_b = generate_2008_liquidity_crisis(seed=999)

    assert run_a["SPY"][50].close != run_b["SPY"][50].close


# --- 2. OHLC Consistency & Monotonicity ---

@pytest.mark.parametrize("generator_func,expected_days", [
    (generate_2008_liquidity_crisis, 252),
    (generate_2020_flash_crash, 60),
    (generate_2022_inflation_grind, 252),
    (generate_2017_low_vol_bull, 252),
])
def test_ohlc_consistency_and_timestamp_ordering(generator_func, expected_days: int):
    """Verify candlestick constraints and strictly ascending timestamps."""
    dataset = generator_func(seed=42)

    for symbol, bars in dataset.items():
        assert len(bars) == expected_days, f"Expected {expected_days} bars for {symbol}"

        for i, bar in enumerate(bars):
            # Monotonic timestamps
            if i > 0:
                assert bar.timestamp > bars[i - 1].timestamp, (
                    f"Timestamp out of order at index {i} for {symbol}"
                )

            # Candlestick geometry
            assert bar.high >= max(bar.open, bar.close) - 1e-5, (
                f"High violation at index {i} for {symbol}: high={bar.high}, open={bar.open}, close={bar.close}"
            )
            assert bar.low <= min(bar.open, bar.close) + 1e-5, (
                f"Low violation at index {i} for {symbol}: low={bar.low}, open={bar.open}, close={bar.close}"
            )
            if bar.vwap is not None:
                assert bar.low <= bar.vwap <= bar.high, (
                    f"VWAP out of bounds: low={bar.low}, vwap={bar.vwap}, high={bar.high}"
                )
            assert bar.low > 0.0, f"Non-positive low ({bar.low}) at index {i} for {symbol}"
            assert bar.open > 0.0, f"Non-positive open ({bar.open}) at index {i} for {symbol}"
            assert bar.close > 0.0, f"Non-positive close ({bar.close}) at index {i} for {symbol}"
            assert bar.volume > 0, f"Non-positive volume ({bar.volume}) at index {i} for {symbol}"


# --- 3. Scenario A: 2008 Liquidity Crisis Statistical Properties ---

def test_scenario_2008_liquidity_crisis_properties(scenario_2008_data: Dict[str, List[Bar]]):
    spy_bars = scenario_2008_data["SPY"]
    qqq_bars = scenario_2008_data["QQQ"]
    tlt_bars = scenario_2008_data["TLT"]
    shv_bars = scenario_2008_data["SHV"]

    spy_closes = np.array([b.close for b in spy_bars])
    tlt_closes = np.array([b.close for b in tlt_bars])
    shv_closes = np.array([b.close for b in shv_bars])

    # 1. SPY severe drawdown & collapse
    spy_return = (spy_closes[-1] / spy_bars[0].open) - 1.0
    assert -0.65 <= spy_return <= -0.45, f"SPY return {spy_return:.2%} outside expected [-65%, -45%]"

    running_max = np.maximum.accumulate(spy_closes)
    spy_max_dd = np.max((running_max - spy_closes) / running_max)
    assert 0.50 <= spy_max_dd <= 0.65, f"SPY max DD {spy_max_dd:.2%} outside expected [50%, 65%]"

    # 2. SPY extreme annualized volatility
    spy_log_ret = np.diff(np.log(spy_closes))
    spy_ann_vol = np.std(spy_log_ret, ddof=1) * np.sqrt(252)
    assert spy_ann_vol >= 0.40, f"SPY annualized vol {spy_ann_vol:.2%} below 40%"

    # 3. TLT flight-to-safety rally
    tlt_return = (tlt_closes[-1] / tlt_bars[0].open) - 1.0
    assert tlt_return >= 0.15, f"TLT return {tlt_return:.2%} below expected safe-haven rally >= 15%"

    # 4. SHV cash preservation
    shv_return = (shv_closes[-1] / shv_bars[0].open) - 1.0
    assert shv_return >= 0.0, f"SHV return {shv_return:.2%} negative in crisis"

    # 5. High equity correlation, negative stock-treasury correlation
    qqq_closes = np.array([b.close for b in qqq_bars])
    qqq_log_ret = np.diff(np.log(qqq_closes))
    tlt_log_ret = np.diff(np.log(tlt_closes))

    corr_spy_qqq = np.corrcoef(spy_log_ret, qqq_log_ret)[0, 1]
    assert corr_spy_qqq >= 0.85, f"SPY-QQQ correlation {corr_spy_qqq:.2f} below 0.85"

    corr_spy_tlt = np.corrcoef(spy_log_ret, tlt_log_ret)[0, 1]
    assert corr_spy_tlt <= 0.05, f"SPY-TLT correlation {corr_spy_tlt:.2f} unexpectedly positive"


# --- 4. Scenario B: 2020 Flash Crash Statistical Properties ---

def test_scenario_2020_flash_crash_properties(scenario_2020_data: Dict[str, List[Bar]]):
    spy_bars = scenario_2020_data["SPY"]
    assert len(spy_bars) == 60

    spy_closes = np.array([b.close for b in spy_bars])

    # Crash phase (first 23 days)
    crash_closes = spy_closes[:24]
    crash_peak = np.maximum.accumulate(crash_closes)
    crash_max_dd = np.max((crash_peak - crash_closes) / crash_peak)
    assert 0.28 <= crash_max_dd <= 0.45, f"Crash max DD {crash_max_dd:.2%} outside expected [28%, 45%]"

    # Volatility during crash phase
    crash_log_ret = np.diff(np.log(crash_closes))
    crash_vol = np.std(crash_log_ret, ddof=1) * np.sqrt(252)
    assert crash_vol >= 0.50, f"Crash phase annualized vol {crash_vol:.2%} below 50%"

    # Recovery phase: trough to end rebound
    trough_price = np.min(crash_closes)
    recovery_rebound = (spy_closes[-1] / trough_price) - 1.0
    assert recovery_rebound >= 0.20, f"V-rebound {recovery_rebound:.2%} below 20%"


# --- 5. Scenario C: 2022 Inflation Grind Statistical Properties ---

def test_scenario_2022_inflation_grind_properties(scenario_2022_data: Dict[str, List[Bar]]):
    spy_bars = scenario_2022_data["SPY"]
    tlt_bars = scenario_2022_data["TLT"]
    xle_bars = scenario_2022_data["XLE"]
    shv_bars = scenario_2022_data["SHV"]

    spy_closes = np.array([b.close for b in spy_bars])
    tlt_closes = np.array([b.close for b in tlt_bars])
    xle_closes = np.array([b.close for b in xle_bars])
    shv_closes = np.array([b.close for b in shv_bars])

    # 1. SPY declines
    spy_ret = (spy_closes[-1] / spy_bars[0].open) - 1.0
    assert -0.30 <= spy_ret <= -0.15, f"SPY return {spy_ret:.2%} outside [-30%, -15%]"

    # 2. TLT also crashes (bond duration failure)
    tlt_ret = (tlt_closes[-1] / tlt_bars[0].open) - 1.0
    assert -0.40 <= tlt_ret <= -0.20, f"TLT return {tlt_ret:.2%} outside [-40%, -20%]"

    # 3. Positive Stock-Bond correlation breakdown
    spy_log_ret = np.diff(np.log(spy_closes))
    tlt_log_ret = np.diff(np.log(tlt_closes))
    corr_spy_tlt = np.corrcoef(spy_log_ret, tlt_log_ret)[0, 1]
    assert corr_spy_tlt >= 0.30, f"Stock-bond correlation {corr_spy_tlt:.2f} failed to turn positive in inflation grind"
    assert corr_spy_tlt > 0.60, f"Stock-bond correlation {corr_spy_tlt:.2f} below target 0.60"

    # 4. Energy (XLE) surge
    xle_ret = (xle_closes[-1] / xle_bars[0].open) - 1.0
    assert xle_ret >= 0.35, f"XLE return {xle_ret:.2%} below expected energy surge >= 35%"

    # 5. Cash (SHV) positive
    shv_ret = (shv_closes[-1] / shv_bars[0].open) - 1.0
    assert shv_ret >= 0.01, f"SHV return {shv_ret:.2%} below 1%"


# --- 6. Scenario D: 2017 Low-Vol Bull Statistical Properties ---

def test_scenario_2017_low_vol_bull_properties(scenario_2017_data: Dict[str, List[Bar]]):
    spy_bars = scenario_2017_data["SPY"]
    qqq_bars = scenario_2017_data["QQQ"]

    spy_closes = np.array([b.close for b in spy_bars])
    qqq_closes = np.array([b.close for b in qqq_bars])

    # 1. Steady compounding return
    spy_ret = (spy_closes[-1] / spy_bars[0].open) - 1.0
    assert 0.15 <= spy_ret <= 0.30, f"SPY return {spy_ret:.2%} outside [15%, 30%]"

    # 2. Low annualized volatility
    spy_log_ret = np.diff(np.log(spy_closes))
    spy_ann_vol = np.std(spy_log_ret, ddof=1) * np.sqrt(252)
    assert 0.06 <= spy_ann_vol <= 0.12, f"SPY vol {spy_ann_vol:.2%} outside [6%, 12%]"

    # 3. Negligible max drawdown (< 4.0%)
    running_max = np.maximum.accumulate(spy_closes)
    spy_max_dd = np.max((running_max - spy_closes) / running_max)
    assert spy_max_dd <= 0.04, f"SPY max drawdown {spy_max_dd:.2%} exceeded 4%"

    # 4. Tech / QQQ outperformance
    qqq_ret = (qqq_closes[-1] / qqq_bars[0].open) - 1.0
    assert qqq_ret > spy_ret, f"QQQ return {qqq_ret:.2%} did not exceed SPY {spy_ret:.2%}"


# --- 7. Dataset Wrapper and DataFrame Export ---

def test_stress_scenario_dataset_dataframe():
    dataset = generate_stress_scenario(StressScenarioType.LOW_VOL_BULL_2017, seed=42)
    df = dataset.to_dataframe()
    expected_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
    assert list(df.columns) == expected_cols
    assert not df.empty
    assert not df.isnull().any().any()
    assert df["open"].min() > 0.0
    assert df["volume"].min() > 0
