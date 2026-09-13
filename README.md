# AlpacaRelay Systematic Long-Term Holding & Rebalancing Strategy Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-532%20passed%20(100%25)-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-4--Regime%20Antonacci%20Dual%20Momentum-purple.svg)]()
[![Persistence](https://img.shields.io/badge/storage-SQLite%20WAL%20%2B%20JSONL-orange.svg)]()

An institutional-grade, programmatic long-term US equity holding and rebalancing decision engine. Built to systematically achieve **higher cumulative returns than SPY** while strictly maintaining a **lower maximum drawdown than SPY** ($\text{MaxDD} < 15\%$) across all historical macroeconomic regimes.

---

## Architecture Overview

The Strategy Engine operates on market signals ingested from AlpacaRelay data feeds (`/Users/mo/AlpacaRelay`), executing deterministic mathematical rules on daily, weekly, and monthly cadences:

```
                          ┌──────────────────────────┐
                          │   AlpacaRelay Data Feed  │
                          │   (REST Proxy & WSS SIP) │
                          └─────────────┬────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        Ingestion & Lifecycle Guard                         │
│  - REST Client with Token Bucket (180 req/min) & Auto-Pagination           │
│  - Async WebSocket Client (10s Auth Handshake, Dedicated Ring Buffer)      │
│  - Upstream Lifecycle State Machine (Auto-Failover to STALE_DATA_HOLD)    │
└───────────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         Quantitative Signal Engine                         │
│  - 20-Day Realized Volatility Targeting Engine (Target Vol = 12.0%)        │
│  - 50 / 200 SMA Dual Trend Filter & Market Breadth                         │
│  - Dynamic ATR Keltner Channel Breakout Circuit Breaker (Lower Band)       │
│  - Trailing Peak-to-Trough Drawdown Defense Gates (-5%, -10%, -15%)        │
│  - Stateful 3-Day Recovery Hysteresis Gate                                 │
└───────────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                     4-Regime Dynamic Asset Allocator                       │
│  - BULL_AGGRESSIVE: 50% QQQ + 30% Top-2 Momentum Sectors + 20% SPY        │
│  - BULL_NORMAL: 35% QQQ + 20% Top-1 Sector + 25% SPY + 20% Defensive       │
│  - CORRECTION_FRAGILE: 20% Defensive Sector (XLV) + 80% Safe Havens/Cash   │
│  - BEAR_CRISIS: 100% Defensive Capital (Antonacci Dual Momentum)           │
│  - STALE_DATA_HOLD: 0-Order Suppression Freeze & 100% Cash Preservation    │
└───────────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        Execution & Persistence Engine                      │
│  - Drift Band Filter (+/- 2.5%) & Micro-Order Suppression (< 0.5%)         │
│  - Sequenced Order Intents (SELL orders prioritized before BUYs)           │
│  - SQLite WAL Database (Point-in-Time State & Relational Storage)          │
│  - Daily Partitioned Append-Only JSONL Audit Logger                        │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Subsystem Modules

| Module | Purpose | Key Classes & Functions |
|---|---|---|
| `strategy_engine.core` | Immutable domain models, universe specifications, and deterministic mathematical utilities | `Bar`, `Quote`, `Trade`, `SignalSnapshot`, `TargetAllocation`, `OrderIntent`, `UNIVERSE_ASSETS`, `math_utils` |
| `strategy_engine.simulator` | Merton Jump-Diffusion SDE synthetic regime simulator calibrated to historical market stress scenarios | `MertonJumpDiffusionSimulator`, `generate_stress_scenario` (2008, 2020, 2022, 2017) |
| `strategy_engine.ingestion` | AlpacaRelay REST and WebSocket streaming clients with connection lifecycle handling | `AlpacaRelayRestClient`, `AlpacaRelayWSClient`, `IngestionStateMachine`, `UnifiedRelayClient` |
| `strategy_engine.signals` | Multi-timeframe quantitative signal engine, Gary Antonacci dual momentum ranker, and regime detector | `SignalEngine`, `compute_universe_momentum`, `evaluate_safe_haven_dual_momentum`, `MarketRegimeDetector` |
| `strategy_engine.allocator` | Point-in-time deterministic allocation rules, risk overlays, and drift band rebalancer | `compute_deterministic_allocation`, `PortfolioRebalancer`, `normalize_target_weights` |
| `strategy_engine.storage` | SQLite WAL database engine, point-in-time state repositories, and daily JSONL audit logger | `Database`, `StorageService`, `SignalSnapshotRepository`, `AllocationRepository`, `JSONLAuditLogger` |
| `strategy_engine.daemon` | Market calendar scheduler (NYSE holidays, DST shifts, early closes) and production daemon | `MarketCalendar`, `MarketScheduler`, `DecisionDaemon`, `DaemonConfig` |
| `strategy_engine.cli` | Production Typer CLI application with Rich diagnostic visual dashboard output | `dry-run`, `daemon`, `rebalance`, `backtest`, `status`, `export-metrics`, `ExplainabilityEngine` |

---

## CLI Usage & Commands

The CLI can be executed directly via `python -m strategy_engine.cli <command>` or via the `strategy-engine` script entrypoint.

### 1. Dry Run Evaluation
Evaluate market data, compute signals, classify regime, and print target allocation table:
```bash
python -m strategy_engine.cli dry-run --scenario 2008
python -m strategy_engine.cli dry-run --scenario 2020 --equity 250000 --json
```

### 2. Historical Stress Backtesting
Simulate strategy performance across calibrated historical stress scenarios and benchmark against SPY:
```bash
python -m strategy_engine.cli backtest --scenario all
python -m strategy_engine.cli backtest --scenario 2022 --export-csv backtest_2022.csv
```

### 3. Portfolio Rebalancing Evaluation
Evaluate current portfolio against target allocation, enforcing $\pm 2.5\%$ drift dead-bands:
```bash
python -m strategy_engine.cli rebalance -w '{"SPY": 0.50, "QQQ": 0.50}' --scenario 2017
python -m strategy_engine.cli rebalance -w '{"SPY": 1.00}' --force
```

### 4. Production Decision Daemon
Start the NYSE market calendar-aware daemon:
```bash
# Execute single evaluation tick and exit
python -m strategy_engine.cli daemon --once

# Continuous daemon mode (15:50 ET daily close evaluation & weekly rebalance)
python -m strategy_engine.cli daemon --interval 60s --relay-url https://alpacarelay-production.up.railway.app
```

### 5. System Status Dashboard
Inspect SQLite table counts, WAL status, latest snapshot, and AlpacaRelay `/health`:
```bash
python -m strategy_engine.cli status --db-path strategy_engine.db
```

### 6. Historical Metrics Export
Export signal snapshots, allocations, and order history to JSON or CSV:
```bash
python -m strategy_engine.cli export-metrics --format json --table all --output exported_metrics.json
python -m strategy_engine.cli export-metrics --format csv --table signals --output signals.csv
```

---

## Test Suite & Quality Verification

Run all 530+ tests across unit, integration, e2e, and adversarial stress tiers:

```bash
# Full test suite execution
pytest -v

# E2E Test Suite (Tiers 1 through 4)
pytest -v tests/e2e/

# Adversarial & Concurrency Stress Suite
pytest -v tests/adversarial/
```

### Verified Mathematical Invariants
- **Strict Sum-to-One Normalization**: $\sum_{i} w_i = 1.00000 \pm 10^{-5}$ across all regimes and allocations.
- **Zero Forward Lookahead**: Strict filtration $\mathcal{F}_t$ point-in-time bar selection with scrambled timestamp immunity.
- **Duration Shock Protection**: Gary Antonacci hurdle eliminates long Treasuries (`TLT`) whenever $P_{TLT} < SMA_{200}$, preventing 2022-style 60/40 duration trap drawdowns.
- **Concurrency Resilience**: 20 concurrent writer threads under SQLite WAL mode with 0 lock errors and 100% data integrity.
- **Stale Hold Order Suppression**: Immediate 0-order suppression when upstream market feeds disconnect or experience timeout.
