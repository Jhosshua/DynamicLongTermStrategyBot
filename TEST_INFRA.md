# TEST_INFRA — Dynamic Long-Term Strategy Bot 4-Tier Opaque-Box E2E Test Suite Infrastructure

## 1. Test Philosophy & Framework Architecture

### 1.1 Opaque-Box, Requirement-Driven Verification
The Dynamic Long-Term Strategy Bot is verified using **opaque-box, requirement-driven verification**. The test harness treats system components, background daemons, storage engines, alert sinks, and the web interface as black boxes evaluated against observable contracts:
1. **Zero Facade Testing**: No mock-matching facades that pass unconditionally without executing real logic. Every test exercises concrete calculations, SQLite WAL transactions, HTTP responses, embed color codes, HTML structure, or Git configurations.
2. **Authoritative Output Derivation**: Expected test outcomes are derived strictly from `ORIGINAL_REQUEST.md` and `PROJECT.md § Feature Inventory` (all 20 features), covering exact numerical balances ($50,000.00 cash), Discord institutional color codes (`0xE53935`, `0x43A047`, `0x1E88E5`), rate-limiting ceilings (<= 1 post/2s, max sleep 5s), responsive viewport bounds (375px–430px), and unauthenticated `/health` contracts.
3. **Progressive Testability & Independence**: Each test is atomic, self-contained, and isolated. Tests clean up temporary databases, do not depend on test execution order, and use dynamic contract resolvers (`resolve_paper_account_cls`, `resolve_feed_manager_cls`, `resolve_discord_notifier_cls`) to bind to real production implementations when present or verified contract adapters during phased development.

### 1.2 Systematic 4-Tier Testing Architecture (`tests/e2e_suite/`)
The test infrastructure is strictly organized into four complementary tiers:
- **Tier 1: Feature Coverage (Isolation / Happy Path)** (`tests/e2e_suite/test_tier1_feature_coverage.py`): Exercising Features 1 through 20 under nominal, valid operational conditions (5 tests per feature, exactly 100 tests).
- **Tier 2: Boundary & Corner Cases** (`tests/e2e_suite/test_tier2_boundary_corner.py`): Exercising edge cases, extreme market conditions, zero/negative inputs, rapid disconnects, truncated logs, and invalid commands (5 tests per feature, exactly 100 tests).
- **Tier 3: Cross-Feature Combinations & Pairwise Interactions** (`tests/e2e_suite/test_tier3_cross_feature.py`): Exercising multi-module workflows, state synchronizations, and feedback loops between subsystems (20 tests).
- **Tier 4: Real-World Application Scenarios** (`tests/e2e_suite/test_tier4_scenarios.py`): Exercising full-system behavior under multi-day simulations, real-world market crashes, disaster recovery, operator dashboard workflows, and public Railway deployment (5 comprehensive scenarios).

---

## 2. Feature Inventory Mapping (All 20 Features)

| # | Feature Name | Milestone | Target Module | Scope & Observable Contracts | Primary Test Tiers |
|---|--------------|-----------|---------------|------------------------------|--------------------|
| 1 | $50k Paper Account Initialization | M1 | `bot.paper_account` | Initialize isolated paper trading account with exactly $50,000.00 cash balance in SQLite WAL | Tier 1, Tier 2, Tier 3 |
| 2 | Persistent Paper Portfolio Ledger | M1 | `bot.paper_account` | SQLite WAL tables tracking positions, cash balance, realized/unrealized P&L, order intents, and execution history across restarts | Tier 1, Tier 2, Tier 3, Tier 4 |
| 3 | AlpacaRelay Ingestion Client | M1 | `bot.feed_manager` | REST and WebSocket client connecting to AlpacaRelay proxy with `RELAY_TOKEN` authentication | Tier 1, Tier 2, Tier 3 |
| 4 | Disconnect Detection & Fallback Simulation | M1 | `bot.feed_manager` | Detect `upstream_disconnected` or REST failures, seamlessly fall back to historical/synthetic market simulation, set dashboard banner alert flag | Tier 1, Tier 2, Tier 3, Tier 4 |
| 5 | 4-Regime Strategy Integration | M1 | `strategy_engine.allocator` | Execute 4-regime dynamic long-term rebalancing logic, drift bands, vol scaling, and Antonacci safe-haven rotation on paper portfolio | Tier 1, Tier 2, Tier 3, Tier 4 |
| 6 | Discord v2 Broken Alert Card | M2 | `bot.discord_alerts` | Institutional red embed card (`0xE53935` / 15022389) dispatched on disconnects or runtime errors with evidence and dashboard link | Tier 1, Tier 2, Tier 3, Tier 4 |
| 7 | Discord v2 Recovered Alert Card | M2 | `bot.discord_alerts` | Institutional green embed card (`0x43A047` / 4431943) dispatched on reconnection/recovery with duration and metrics | Tier 1, Tier 2, Tier 3, Tier 4 |
| 8 | Discord v2 Trade Execution Card | M2 | `bot.discord_alerts` | Institutional blue embed card (`0x1E88E5` / 2001125) dispatched on rebalance orders with fill details, weights, and NAV | Tier 1, Tier 2, Tier 3, Tier 4 |
| 9 | Discord Rate Limiting & Pytest Suppression | M2 | `bot.discord_alerts` | Honor Discord 429 rate limits with exponential backoff (<= 1 post/2s, max sleep 5s) and drop alerts under pytest | Tier 1, Tier 2, Tier 3 |
| 10 | Light & Airy Mobile-Centric Dashboard | M3 | `web.app`, `dashboard.html` | Fast responsive web dashboard with light slate/white palette, crisp typography, clean card hierarchy | Tier 1, Tier 2, Tier 3, Tier 4 |
| 11 | Mobile Viewport Responsiveness | M3 | `web.templates`, CSS | Flawless mobile rendering on 375px–430px viewports with touch-friendly (>=44px) targets and zero horizontal scroll | Tier 1, Tier 2, Tier 3, Tier 4 |
| 12 | Live Portfolio & Regime Metrics Display | M3 | `web.app` | Display live portfolio equity, $50k paper balance, active positions, strategy regime/signals, market news, connection health | Tier 1, Tier 2, Tier 3, Tier 4 |
| 13 | AlpacaRelay Disconnect Alert Banner | M3 | `web.templates`, `web.app` | Prominent visual warning banner on dashboard triggered whenever AlpacaRelay is disconnected or operating in fallback | Tier 1, Tier 2, Tier 3, Tier 4 |
| 14 | Operator Real-Time Controls | M3 | `web.app` | Web controls for Pause (`POST /api/operator/pause`), Resume (`POST /api/operator/resume`), and Trigger Manual Rebalance evaluation | Tier 1, Tier 2, Tier 3, Tier 4 |
| 15 | Multi-Agent Adversarial Review & Smoke Testing | M4 | `scripts.e2e_smoke_test` | Phased adversarial code reviews and end-to-end UI smoke test with injected synthetic market and trade data | Tier 1, Tier 2, Tier 3, Tier 4 |
| 16 | Multi-Agent UI Design & Usability Audit | M4 | Review Audits | Multiple UI reviewer agents auditing operator-first simplicity, visual hierarchy, and absence of confusion | Tier 1, Tier 2, Tier 3 |
| 17 | Pristine State Reset for Monday's Open | M4 | `scripts.reset_pristine_for_monday` | Complete purge of synthetic test data, resetting paper portfolio to clean $50,000.00 cash, 0 positions, 0 open orders | Tier 1, Tier 2, Tier 3, Tier 4 |
| 18 | Dedicated Git Repository Setup | M5 | Git Repo Root | Initialize Git repo at `/Users/mo/DynamicLongTermStrategyBot`, commit all code with clean structured commits | Tier 1, Tier 2, Tier 3, Tier 4 |
| 19 | GitHub Remote Push (`Jhosshua`) | M5 | GitHub Remote | Create and push to GitHub repository under account `Jhosshua/DynamicLongTermStrategyBot` | Tier 1, Tier 2, Tier 3 |
| 20 | Public Token-Free Railway Deployment | M5 | Deployment, `/health` | Deploy service to Railway with publicly accessible token-free HTTPS URL, verifying HTTP 200 on `/health` and dashboard | Tier 1, Tier 2, Tier 3, Tier 4 |

---

## 3. Coverage Thresholds & Tier Taxonomy

### 3.1 Tier 1: Feature Coverage (Happy Path & Nominal Execution)
- **Target**: Exactly 5 tests per feature across all 20 features (total 100 tests).
- **File**: `tests/e2e_suite/test_tier1_feature_coverage.py`
- **Focus**: Nominal valid execution, API contracts, schema adherence, parameter verification, and expected output types.

### 3.2 Tier 2: Boundary & Corner Cases
- **Target**: Exactly 5 tests per feature across all 20 features (total 100 tests).
- **File**: `tests/e2e_suite/test_tier2_boundary_corner.py`
- **Focus**: Extreme inputs, zero/null values, insufficient cash, overselling, large order batching, clock skews, 429 rate limits, and network anomalies.

### 3.3 Tier 3: Pairwise & Cross-Feature Interactions
- **Target**: 20 comprehensive pairwise interaction tests.
- **File**: `tests/e2e_suite/test_tier3_cross_feature.py`
- **Focus**: Multi-module coordination:
  1. Paper account initialization + transaction execution lifecycle (F1 + F2)
  2. Trading accumulation followed by atomic pristine reset (F1 + F17)
  3. Ingestion client upstream disconnect triggering synthetic fallback (F3 + F4)
  4. Feed disconnect activating dashboard alert banner (F4 + F13)
  5. Feed disconnect dispatching Discord v2 broken alert card (F4 + F6)
  6. Feed reconnection dispatching Discord v2 recovered alert card (F4 + F7)
  7. Strategy rebalancer calculating deltas and executing into SQLite WAL ledger (F5 + F2)
  8. Strategy rebalance dispatching Discord v2 trade execution card (F5 + F8)
  9. Consecutive error alerts passing through 429 rate limiter (F6 + F9)
  10. Trade execution card suppressed under pytest environment (F8 + F9)
  11. Mobile dashboard rendering responsive on 375px–430px viewports (F10 + F11)
  12. Dashboard root page displaying live portfolio metrics (F10 + F12)
  13. API synchronizing simultaneous position updates and disconnect banner (F12 + F13)
  14. Operator Pause control freezing strategy rebalances, Resume unfreezing (F14 + F5)
  15. Operator manual rebalance trigger updating dashboard NAV (F14 + F12)
  16. Adversarial smoke test injection followed by complete pristine reset (F15 + F17)
  17. Usability audit validating light/airy theme contrast and clean hierarchy (F16 + F10)
  18. Git repository initialization and remote push alignment with `Jhosshua` (F18 + F19)
  19. Railway unauthenticated health check querying live paper account NAV (F20 + F12)
  20. Complete closed-loop trade cycle: Regime -> Alloc -> Execution -> Card -> Web (F2 + F5 + F8 + F12)

### 3.4 Tier 4: Realistic Application Workload Scenarios
- **Target**: 5 full-system workload scenarios.
- **File**: `tests/e2e_suite/test_tier4_scenarios.py`
- **Focus**:
  1. **Scenario 1: Bull-to-Bear Market Crash & Safe-Haven Rotation**: Multi-day crash (-15%) triggers regime transition to BEAR_CRISIS; capital rotates from equities into SHV cash; Discord card dispatched; capital preserved.
  2. **Scenario 2: AlpacaRelay Outage, Synthetic Fallback, and Seamless Recovery**: Upstream disconnect activates synthetic fallback and dashboard alert banner; broken card posted; simulation maintains trading; recovery restores live feed and green card.
  3. **Scenario 3: Operator Real-Time Interventions & Manual Rebalance**: Mobile dashboard monitoring; operator pauses during volatility; rebalance frozen; operator resumes and triggers manual rebalance.
  4. **Scenario 4: Multi-Agent Adversarial Smoke Test & Pristine Reset for Monday**: 50 synthetic bars and 10 fake orders injected; reactive UI verified; `reset_to_pristine()` executed; database restored to exact $50,000.00 cash, 0 positions, ready for Monday's open.
  5. **Scenario 5: Production Deployment Pipeline & Railway Health Verification**: Unauthenticated `/health` returns HTTP 200 with service status, live portfolio NAV ($50k), and relay state; root `/` serves responsive light/airy dashboard.

---

## 4. Test Harness Infrastructure Components

### 4.1 Canonical Contracts & Adapters (`tests/e2e_suite/contracts.py`)
- Standardized dataclasses: `PositionDetail`, `PortfolioSummary`, `ConnectionStatus`, `RebalanceOrder`, `DiscordEmbedCard`.
- Institutional Discord color codes: `DISCORD_COLOR_BROKEN` (0xE53935), `DISCORD_COLOR_RECOVERED` (0x43A047), `DISCORD_COLOR_TRADE` (0x1E88E5).
- Reference implementations: `PaperAccountManagerContract`, `DataFeedManagerContract`, `DiscordNotifierContract`, `OperatorAppContract`.
- Dynamic progressive resolvers: `resolve_paper_account_cls()`, `resolve_feed_manager_cls()`, `resolve_discord_notifier_cls()`.

### 4.2 Pytest Fixtures (`tests/e2e_suite/conftest.py`)
- `temp_paper_db`: Isolated temporary SQLite database with automatic WAL cleanup.
- `paper_account`: PaperAccountManager instance initialized to $50,000.00 cash.
- `feed_manager`: DataFeedManager instance managing connection and fallback simulation.
- `discord_notifier`: DiscordNotifier instance with pytest suppression enabled.
- `operator_app`: OperatorAppContract instance for HTTP and API dispatch.
- `sample_rebalance_orders`: Pre-configured list of RebalanceOrder objects.

---

## 5. Verification & Test Execution Protocol

```bash
# Run complete 4-Tier E2E test suite (225 tests)
uv run pytest tests/e2e_suite/

# Run individual tiers
uv run pytest tests/e2e_suite/test_tier1_feature_coverage.py   # Tier 1 (100 tests)
uv run pytest tests/e2e_suite/test_tier2_boundary_corner.py    # Tier 2 (100 tests)
uv run pytest tests/e2e_suite/test_tier3_cross_feature.py      # Tier 3 (20 tests)
uv run pytest tests/e2e_suite/test_tier4_scenarios.py          # Tier 4 (5 scenarios)

# Concise execution report
uv run pytest -q --tb=short tests/e2e_suite/
```
