"""
tests/unit/test_universe.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for universe specification, 4-tier taxonomy, and asset class mappings.
"""

import pytest

from strategy_engine.core.models import AssetClass
from strategy_engine.core.universe import (
    ALL_SYMBOLS,
    ALL_UNIVERSE,
    CORE_COMPOUNDERS,
    CYCLICAL_SECTORS,
    DEFENSIVE_SECTORS,
    MEGACAP_LEADERS,
    SAFE_HAVENS,
    SECTOR_ETFS,
    UNIVERSE_GROWTH,
    UNIVERSE_SAFE_HAVENS,
    UNIVERSE_SECTORS,
    get_asset,
    get_asset_class,
    get_cash_equivalents,
    get_cyclical_sectors,
    get_defensive_sectors,
    get_growth_symbols,
    get_megacap_symbols,
    get_safe_haven_symbols,
    get_sector_symbols,
    get_symbols_by_asset_class,
    get_symbols_by_tier,
    get_universe,
    is_safe_haven,
    is_sector_etf,
    is_valid_symbol,
)


def test_core_compounders_tier():
    assert "SPY" in CORE_COMPOUNDERS
    assert "QQQ" in CORE_COMPOUNDERS
    assert len(CORE_COMPOUNDERS) == 2


def test_megacap_leaders_tier():
    expected_leaders = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"}
    assert expected_leaders.issubset(set(MEGACAP_LEADERS))


def test_sector_etfs_eleven_sectors():
    assert len(SECTOR_ETFS) == 11
    assert set(DEFENSIVE_SECTORS) == {"XLV", "XLP", "XLU"}
    assert set(CYCLICAL_SECTORS) == {"XLK", "XLC", "XLY", "XLI", "XLF", "XLE", "XLB", "XLRE"}
    assert set(DEFENSIVE_SECTORS).union(set(CYCLICAL_SECTORS)) == set(SECTOR_ETFS)


def test_safe_havens_tier():
    expected_havens = {"TLT", "SHV", "BIL", "GLD"}
    assert expected_havens == set(SAFE_HAVENS)


def test_all_universe_breadth():
    assert len(ALL_UNIVERSE) >= 22
    assert set(CORE_COMPOUNDERS).issubset(set(ALL_UNIVERSE))
    assert set(MEGACAP_LEADERS).issubset(set(ALL_UNIVERSE))
    assert set(SECTOR_ETFS).issubset(set(ALL_UNIVERSE))
    assert set(SAFE_HAVENS).issubset(set(ALL_UNIVERSE))


def test_get_asset_class_mapping():
    assert get_asset_class("SPY") == AssetClass.EQUITY_INDEX
    assert get_asset_class("QQQ") == AssetClass.EQUITY_INDEX
    assert get_asset_class("AAPL") == AssetClass.EQUITY_LEADER
    assert get_asset_class("NVDA") == AssetClass.EQUITY_LEADER
    assert get_asset_class("XLK") == AssetClass.EQUITY_SECTOR
    assert get_asset_class("XLE") == AssetClass.EQUITY_SECTOR
    assert get_asset_class("TLT") == AssetClass.FIXED_INCOME_LONG
    assert get_asset_class("SHV") == AssetClass.CASH_EQUIVALENT
    assert get_asset_class("BIL") == AssetClass.CASH_EQUIVALENT
    assert get_asset_class("CASH") == AssetClass.CASH_EQUIVALENT
    assert get_asset_class("GLD") == AssetClass.COMMODITY


def test_get_asset_class_unknown_symbol():
    with pytest.raises(KeyError):
        get_asset_class("NONEXISTENT_TICKER_123")


def test_get_symbols_by_tier():
    tier1 = get_symbols_by_tier(1)
    assert set(tier1) == set(CORE_COMPOUNDERS)

    tier2 = get_symbols_by_tier(2)
    assert set(tier2) == set(MEGACAP_LEADERS)

    tier3 = get_symbols_by_tier(3)
    assert set(tier3) == set(SECTOR_ETFS)

    tier4 = get_symbols_by_tier(4)
    assert set(tier4) == set(SAFE_HAVENS)


def test_query_helpers():
    assert is_valid_symbol("SPY") is True
    assert is_valid_symbol("FAKE") is False
    assert is_safe_haven("TLT") is True
    assert is_safe_haven("SHV") is True
    assert is_safe_haven("SPY") is False
    assert is_sector_etf("XLK") is True
    assert is_sector_etf("QQQ") is False

    assert set(get_defensive_sectors()) == {"XLV", "XLP", "XLU"}
    assert "XLK" in get_cyclical_sectors()
    assert "SHV" in get_cash_equivalents()
    assert "BIL" in get_cash_equivalents()
    assert get_asset("SPY") is not None
    assert get_asset("SPY").tier == 1
