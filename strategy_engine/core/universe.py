"""
strategy_engine.core.universe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Universe specifications, 4-tier taxonomy, and asset classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from strategy_engine.core.models import AssetClass


@dataclass(frozen=True)
class UniverseAsset:
    """Metadata specification for a tradeable or benchmark universe asset."""
    symbol: str
    name: str
    asset_class: AssetClass
    tier: int
    is_tradable: bool = True
    is_defensive: bool = False
    benchmark_role: Optional[str] = None
    expense_ratio: Optional[float] = None


# Tier 1: Core Growth Compounders
CORE_COMPOUNDERS: List[str] = ["SPY", "QQQ"]

# Tier 2: Megacap Tech & Momentum Leaders
MEGACAP_LEADERS: List[str] = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

# Tier 3: 11 Select Sector SPDR ETFs
DEFENSIVE_SECTORS: List[str] = ["XLV", "XLP", "XLU"]
CYCLICAL_SECTORS: List[str] = ["XLK", "XLC", "XLY", "XLI", "XLF", "XLE", "XLB", "XLRE"]
SECTOR_ETFS: List[str] = ["XLK", "XLC", "XLY", "XLI", "XLF", "XLV", "XLP", "XLU", "XLE", "XLB", "XLRE"]

# Tier 4: Multi-Regime Safe-Havens & Cash Proxies
SAFE_HAVENS: List[str] = ["TLT", "SHV", "BIL", "GLD"]
CASH_EQUIVALENTS: List[str] = ["SHV", "BIL", "CASH"]

# Groupings and aliases
UNIVERSE_GROWTH: List[str] = sorted(list(set(CORE_COMPOUNDERS + MEGACAP_LEADERS)))
UNIVERSE_SECTORS: List[str] = SECTOR_ETFS
UNIVERSE_SAFE_HAVENS: List[str] = SAFE_HAVENS
ALL_UNIVERSE: List[str] = sorted(list(set(CORE_COMPOUNDERS + MEGACAP_LEADERS + SECTOR_ETFS + SAFE_HAVENS)))
ALL_SYMBOLS: List[str] = ALL_UNIVERSE

UNIVERSE_ASSETS: Dict[str, UniverseAsset] = {
    # Tier 1
    "SPY": UniverseAsset("SPY", "SPDR S&P 500 ETF Trust", AssetClass.EQUITY_INDEX, tier=1, benchmark_role="Primary Equity Benchmark"),
    "QQQ": UniverseAsset("QQQ", "Invesco QQQ Trust", AssetClass.EQUITY_INDEX, tier=1, benchmark_role="Growth Benchmark"),
    
    # Tier 2
    "AAPL": UniverseAsset("AAPL", "Apple Inc.", AssetClass.EQUITY_LEADER, tier=2),
    "MSFT": UniverseAsset("MSFT", "Microsoft Corporation", AssetClass.EQUITY_LEADER, tier=2),
    "NVDA": UniverseAsset("NVDA", "NVIDIA Corporation", AssetClass.EQUITY_LEADER, tier=2),
    "AMZN": UniverseAsset("AMZN", "Amazon.com Inc.", AssetClass.EQUITY_LEADER, tier=2),
    "GOOGL": UniverseAsset("GOOGL", "Alphabet Inc. Class A", AssetClass.EQUITY_LEADER, tier=2),
    "META": UniverseAsset("META", "Meta Platforms Inc.", AssetClass.EQUITY_LEADER, tier=2),
    "TSLA": UniverseAsset("TSLA", "Tesla Inc.", AssetClass.EQUITY_LEADER, tier=2),

    # Tier 3 - Cyclicals
    "XLK": UniverseAsset("XLK", "Technology Select Sector SPDR Fund", AssetClass.EQUITY_SECTOR, tier=3),
    "XLC": UniverseAsset("XLC", "Communication Services Select Sector SPDR Fund", AssetClass.EQUITY_SECTOR, tier=3),
    "XLY": UniverseAsset("XLY", "Consumer Discretionary Select Sector SPDR Fund", AssetClass.EQUITY_SECTOR, tier=3),
    "XLI": UniverseAsset("XLI", "Industrial Select Sector SPDR Fund", AssetClass.EQUITY_SECTOR, tier=3),
    "XLF": UniverseAsset("XLF", "Financial Select Sector SPDR Fund", AssetClass.EQUITY_SECTOR, tier=3),
    "XLE": UniverseAsset("XLE", "Energy Select Sector SPDR Fund", AssetClass.EQUITY_SECTOR, tier=3),
    "XLB": UniverseAsset("XLB", "Materials Select Sector SPDR Fund", AssetClass.EQUITY_SECTOR, tier=3),
    "XLRE": UniverseAsset("XLRE", "Real Estate Select Sector SPDR Fund", AssetClass.EQUITY_SECTOR, tier=3),

    # Tier 3 - Defensives
    "XLV": UniverseAsset("XLV", "Health Care Select Sector SPDR Fund", AssetClass.EQUITY_SECTOR, tier=3, is_defensive=True),
    "XLP": UniverseAsset("XLP", "Consumer Staples Select Sector SPDR Fund", AssetClass.EQUITY_SECTOR, tier=3, is_defensive=True),
    "XLU": UniverseAsset("XLU", "Utilities Select Sector SPDR Fund", AssetClass.EQUITY_SECTOR, tier=3, is_defensive=True),

    # Tier 4 - Safe Havens
    "TLT": UniverseAsset("TLT", "iShares 20+ Year Treasury Bond ETF", AssetClass.FIXED_INCOME_LONG, tier=4, is_defensive=True),
    "SHV": UniverseAsset("SHV", "iShares Short Treasury Bond ETF", AssetClass.CASH_EQUIVALENT, tier=4, is_defensive=True),
    "BIL": UniverseAsset("BIL", "SPDR Bloomberg 1-3 Month T-Bill ETF", AssetClass.CASH_EQUIVALENT, tier=4, is_defensive=True),
    "GLD": UniverseAsset("GLD", "SPDR Gold Shares", AssetClass.COMMODITY, tier=4, is_defensive=True),
}


def get_asset_class(symbol: str) -> AssetClass:
    """Map ticker symbol to its primary AssetClass."""
    sym = symbol.upper()
    if sym in UNIVERSE_ASSETS:
        return UNIVERSE_ASSETS[sym].asset_class
    if sym in ("CASH", "USD"):
        return AssetClass.CASH_EQUIVALENT
    raise KeyError(f"Unknown symbol '{symbol}' for asset class mapping")


def get_universe() -> Dict[str, UniverseAsset]:
    """Return all defined universe assets."""
    return UNIVERSE_ASSETS


def get_asset(symbol: str) -> Optional[UniverseAsset]:
    """Retrieve metadata for a specific universe asset."""
    return UNIVERSE_ASSETS.get(symbol.upper())


def get_symbols_by_tier(tier: int) -> List[str]:
    """Retrieve symbols belonging to a specific tier (1-4)."""
    return [sym for sym, a in UNIVERSE_ASSETS.items() if a.tier == tier]


def get_symbols_by_asset_class(asset_class: AssetClass) -> List[str]:
    """Retrieve symbols belonging to an AssetClass."""
    return [sym for sym, a in UNIVERSE_ASSETS.items() if a.asset_class == asset_class]


def get_growth_symbols() -> List[str]:
    """Retrieve growth / compounder symbols (Tiers 1 & 2)."""
    return UNIVERSE_GROWTH.copy()


def get_megacap_symbols() -> List[str]:
    """Retrieve megacap tech leader symbols."""
    return MEGACAP_LEADERS.copy()


def get_sector_symbols() -> List[str]:
    """Retrieve all 11 SPDR Sector ETF symbols."""
    return SECTOR_ETFS.copy()


def get_defensive_sectors() -> List[str]:
    """Retrieve defensive sector ETF symbols (XLV, XLP, XLU)."""
    return DEFENSIVE_SECTORS.copy()


def get_cyclical_sectors() -> List[str]:
    """Retrieve cyclical/offensive sector ETF symbols."""
    return CYCLICAL_SECTORS.copy()


def get_safe_haven_symbols() -> List[str]:
    """Retrieve safe-haven symbols (TLT, SHV, BIL, GLD)."""
    return SAFE_HAVENS.copy()


def get_cash_equivalents() -> List[str]:
    """Retrieve cash proxy symbols."""
    return CASH_EQUIVALENTS.copy()


def is_valid_symbol(symbol: str) -> bool:
    """Check if symbol is recognized within the universe."""
    return symbol.upper() in UNIVERSE_ASSETS or symbol.upper() in CASH_EQUIVALENTS


def is_safe_haven(symbol: str) -> bool:
    """Check if symbol is classified as a safe-haven or cash instrument."""
    return symbol.upper() in SAFE_HAVENS or symbol.upper() in CASH_EQUIVALENTS


def is_sector_etf(symbol: str) -> bool:
    """Check if symbol is one of the 11 SPDR sector ETFs."""
    return symbol.upper() in SECTOR_ETFS
