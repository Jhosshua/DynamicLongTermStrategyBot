"""
tests.unit.test_momentum
~~~~~~~~~~~~~~~~~~~~~~~~

Unit test suite for 12-1 structural relative momentum and Antonacci safe-haven dual momentum:
- Lookback 252 days, skip 21 days exact math
- Skip window short-term reversal insulation
- Absolute momentum hurdle vs cash and 200 SMA
- Deterministic alphabetical tie-breaking
- 2022 duration shock defense (strict 0% TLT)
- Safe haven allocation splits (TLT, GLD, SHV)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import pytest

from strategy_engine.core.models import Bar
from strategy_engine.signals.momentum import (
    calculate_12_1_momentum,
    calculate_momentum_12_1,
    compute_safe_haven_weights,
    compute_universe_momentum,
    evaluate_absolute_momentum,
    evaluate_safe_haven_dual_momentum,
    evaluate_safe_haven_qualification,
    rank_momentum_candidates,
    select_top_growth_leaders,
    select_top_sectors,
)


def _create_bar(symbol: str, dt: datetime, close: float) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=dt,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=50_000,
    )


def test_12_1_momentum_exact_math():
    """Verify 12-1 structural momentum formula: P[-(skip+1)] / P[-(lookback+1)] - 1.0."""
    prices = [100.0 + i for i in range(260)]
    score = calculate_12_1_momentum(prices, lookback=252, skip=21)
    # p_base = prices[-(252+1)] = prices[260 - 253] = prices[7] = 107.0
    # p_skip = prices[-(21+1)] = prices[260 - 22] = prices[238] = 338.0
    expected = (338.0 / 107.0) - 1.0
    assert math.isclose(score, expected, rel_tol=1e-5)
    # Check alias
    assert math.isclose(calculate_momentum_12_1(prices), expected, rel_tol=1e-5)


def test_12_1_momentum_skip_insulation():
    """Verify that price action during recent 21 trading days has zero impact on 12-1 momentum."""
    base_prices = [100.0 + i * 0.5 for i in range(260)]
    score_normal = calculate_12_1_momentum(base_prices, lookback=252, skip=21)

    # Modify the most recent 21 days (extreme surge)
    surged_prices = base_prices.copy()
    for i in range(1, 22):
        surged_prices[-i] = 9999.0
    score_surged = calculate_12_1_momentum(surged_prices, lookback=252, skip=21)

    # Modify the most recent 21 days (extreme crash)
    crashed_prices = base_prices.copy()
    for i in range(1, 22):
        crashed_prices[-i] = 1.0
    score_crashed = calculate_12_1_momentum(crashed_prices, lookback=252, skip=21)

    assert score_normal == score_surged == score_crashed


def test_12_1_momentum_insufficient_history():
    """Verify graceful handling of series with < 253 prices or non-positive base price."""
    assert calculate_12_1_momentum([100.0] * 200) == 0.0
    assert calculate_12_1_momentum([]) == 0.0
    # Negative / zero base price
    bad_prices = [0.0] + [100.0] * 260
    assert calculate_12_1_momentum(bad_prices) == 0.0


def test_rank_momentum_candidates_descending_and_tie_breaking():
    """Verify candidates are ranked descending by score, with alphabetical secondary tie-breaking."""
    scores = {
        "XLY": 0.1500,
        "XLK": 0.2500,
        "XLE": 0.2500,
        "XLF": -0.0500,
        "XLV": 0.1000,
    }
    ranked = rank_momentum_candidates(scores)
    # Expected order:
    # 1. XLE: 0.2500 (alphabetically precedes XLK)
    # 2. XLK: 0.2500
    # 3. XLY: 0.1500
    # 4. XLV: 0.1000
    # 5. XLF: -0.0500
    assert ranked[0] == ("XLE", 0.2500)
    assert ranked[1] == ("XLK", 0.2500)
    assert ranked[2] == ("XLY", 0.1500)
    assert ranked[3] == ("XLV", 0.1000)
    assert ranked[4] == ("XLF", -0.0500)


def test_absolute_momentum_hurdle_evaluation():
    """Verify Gary Antonacci absolute momentum hurdle (Asset Mom > Cash Mom and Price > SMA200)."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market_data = {}

    # Asset 1 (Strong uptrend, Mom > cash, Price > SMA200) -> PASS
    market_data["WINNER"] = [_create_bar("WINNER", t0 + timedelta(days=i), 100.0 + i * 0.5) for i in range(260)]

    # Asset 2 (Downtrend, Mom < cash, Price < SMA200) -> FAIL
    market_data["LOSER"] = [_create_bar("LOSER", t0 + timedelta(days=i), 300.0 - i * 0.5) for i in range(260)]

    # Cash (SHV)
    market_data["SHV"] = [_create_bar("SHV", t0 + timedelta(days=i), 100.0 + i * 0.01) for i in range(260)]

    assert evaluate_absolute_momentum("WINNER", market_data, cash_symbol="SHV") is True
    assert evaluate_absolute_momentum("LOSER", market_data, cash_symbol="SHV") is False


def test_select_top_sectors():
    """Verify selection of top 2 sectors passing absolute hurdle."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market_data = {}

    # Provide all 11 sectors
    sectors = ["XLK", "XLC", "XLY", "XLI", "XLF", "XLV", "XLP", "XLU", "XLE", "XLB", "XLRE"]
    for idx, sym in enumerate(sectors):
        # Slope proportional to idx -> XLRE highest, XLK lowest
        slope = 0.1 + idx * 0.1
        market_data[sym] = [_create_bar(sym, t0 + timedelta(days=i), 100.0 + i * slope) for i in range(260)]

    market_data["SHV"] = [_create_bar("SHV", t0 + timedelta(days=i), 100.0 + i * 0.01) for i in range(260)]

    top_2 = select_top_sectors(market_data, n_top=2, require_absolute_hurdle=True)
    assert len(top_2) == 2
    assert top_2[0] == "XLRE"
    assert top_2[1] == "XLB"


def test_select_top_growth_leaders():
    """Verify selection of top Megacap leader."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market_data = {}

    leaders = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
    for sym in leaders:
        market_data[sym] = [_create_bar(sym, t0 + timedelta(days=i), 100.0 + i * 0.2) for i in range(260)]
    # Give NVDA highest slope
    market_data["NVDA"] = [_create_bar("NVDA", t0 + timedelta(days=i), 100.0 + i * 1.5) for i in range(260)]
    market_data["SHV"] = [_create_bar("SHV", t0 + timedelta(days=i), 100.0 + i * 0.01) for i in range(260)]

    top_leader = select_top_growth_leaders(market_data, n_top=1)
    assert top_leader == ["NVDA"]


def test_safe_haven_tlt_2022_disqualification():
    """Defend against 2022 duration trap: TLT below 200 SMA forces w(TLT) = strictly 0.0%."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market_data = {}

    # 2022 Rate Shock profile: TLT declining relentlessly
    market_data["TLT"] = [_create_bar("TLT", t0 + timedelta(days=i), 200.0 - i * 0.3) for i in range(260)]

    # GLD also failing or not in data
    market_data["GLD"] = [_create_bar("GLD", t0 + timedelta(days=i), 200.0 - i * 0.2) for i in range(260)]
    market_data["SHV"] = [_create_bar("SHV", t0 + timedelta(days=i), 100.0 + i * 0.01) for i in range(260)]

    sh_weights = evaluate_safe_haven_dual_momentum(market_data, defensive_capital=1.0)
    # TLT MUST be completely eliminated (strictly 0% weight)
    assert "TLT" not in sh_weights or sh_weights["TLT"] == 0.0
    assert "GLD" not in sh_weights or sh_weights["GLD"] == 0.0
    # 100% of defensive capital routed to cash (SHV)
    assert math.isclose(sh_weights.get("SHV", 0.0), 1.0, abs_tol=1e-5)


def test_safe_haven_allocation_gold_stagflation_split():
    """When TLT fails but GLD passes (inflation/commodity hedge): routes 40% GLD, 60% SHV."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market_data = {}

    # TLT down (failing)
    market_data["TLT"] = [_create_bar("TLT", t0 + timedelta(days=i), 200.0 - i * 0.3) for i in range(260)]
    # GLD up (qualifying: Price > SMA200 and Mom > 0)
    market_data["GLD"] = [_create_bar("GLD", t0 + timedelta(days=i), 100.0 + i * 0.5) for i in range(260)]
    market_data["SHV"] = [_create_bar("SHV", t0 + timedelta(days=i), 100.0 + i * 0.01) for i in range(260)]

    sh_weights = evaluate_safe_haven_dual_momentum(market_data, defensive_capital=0.80)
    assert "TLT" not in sh_weights or sh_weights["TLT"] == 0.0
    assert math.isclose(sh_weights["GLD"], 0.40 * 0.80, abs_tol=1e-5)
    assert math.isclose(sh_weights["SHV"], 0.60 * 0.80, abs_tol=1e-5)
    assert math.isclose(sum(sh_weights.values()), 0.80, abs_tol=1e-5)


def test_safe_haven_allocation_both_pass():
    """When both TLT and GLD qualify: 40% TLT, 40% GLD, 20% SHV."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market_data = {}

    # Both TLT and GLD rising firmly
    market_data["TLT"] = [_create_bar("TLT", t0 + timedelta(days=i), 100.0 + i * 0.5) for i in range(260)]
    market_data["GLD"] = [_create_bar("GLD", t0 + timedelta(days=i), 100.0 + i * 0.4) for i in range(260)]
    market_data["SHV"] = [_create_bar("SHV", t0 + timedelta(days=i), 100.0 + i * 0.01) for i in range(260)]

    sh_weights = evaluate_safe_haven_dual_momentum(market_data, defensive_capital=1.0)
    assert math.isclose(sh_weights["TLT"], 0.40, abs_tol=1e-5)
    assert math.isclose(sh_weights["GLD"], 0.40, abs_tol=1e-5)
    assert math.isclose(sh_weights["SHV"], 0.20, abs_tol=1e-5)
    assert math.isclose(sum(sh_weights.values()), 1.0, abs_tol=1e-5)
