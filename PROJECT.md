# Project: Dynamic Long-Term Strategy Bot

## Architecture
The Dynamic Long-Term Strategy Bot is an institutional-grade automated trading system executing a 4-regime dynamic allocation strategy on US equity instruments (SPY, QQQ, top sectors, TLT, SHV/BIL, GLD). The bot is architected as an integrated single-process async daemon and web server providing:
1. **Virtual Paper Trading Engine**: Isolated $50,000.00 starting cash balance, persistent SQLite WAL storage, position tracking, P&L calculations, and execution history.
2. **Resilient Data Ingestion**: AlpacaRelay REST & WebSocket client with a synthetic price display when the relay is down, plus a dashboard alert banner. The bot never evaluates signals or trades unless the live relay feed is up (fail closed).
3. **Institutional Discord v2 Notifications**: Rate-limited (<= 1 post/2s, 429 exponential backoff) rich embed cards (`broken`, `recovered`, `trade_execution`) with deep links to the public dashboard and pytest suppression.
4. **Light & Airy Operator Dashboard**: Fast, mobile-first responsive web interface (375px–430px) featuring modern soft-slate/pure-white palette, operator controls (Pause/Resume, Manual Rebalance), live regime & signal metrics, and AlpacaRelay status badge.
5. **E2E Verification & Pristine Reset**: Injected synthetic stress smoke tests verifying reactive UI and order execution, followed by a guaranteed purge and reset to pristine $50,000.00 cash balance for Monday's market open.
6. **Public Cloud Deployment**: Dedicated Git repository under `Jhosshua` and Railway public HTTPS deployment. Viewing (`/`, `/health`, GET APIs) is public; operator POSTs need `X-Operator-Token`.

```
+-----------------------------------------------------------------------------------+
|                           DynamicLongTermStrategyBot                              |
|                                                                                   |
|  +--------------------------+             +------------------------------------+  |
|  |     AlpacaRelay Client   |             |       Mobile Operator Dashboard    |  |
|  |  - WebSocket & REST proxy|  events     |  - Light & airy modern UX (Tailwind|  |
|  |  - Auto Fallback to Sim  +---------->  |  - Live NAV, P&L, Regime, Positions|  |
|  |  - Disconnect Alert Flag |             |  - Controls: Pause, Resume, Rebal  |  |
|  +------------+-------------+             +-----------------+------------------+  |
|               | bars                                        ^                     |
|               v                                             | SSE / REST          |
|  +--------------------------+             +-----------------+------------------+  |
|  |  4-Regime Decision Engine|  signals    |    Paper Trading Storage (WAL)     |  |
|  |  - Antonacci Momentum    +---------->  |  - $50,000.00 Cash Initial Ledger  |  |
|  |  - Vol Scaler / Drawdown |  orders     |  - Positions, Executions, P&L      |  |
|  +------------+-------------+             |  - Pristine Reset for Monday       |  |
|               |                           +-----------------+------------------+  |
|               | order intents                               |                     |
|               v                                             v                     |
|  +----------------------------------------------------------+------------------+  |
|  |                    Institutional Discord v2 Alert Cards                     |  |
|  |  - Broken (Red), Recovered (Green), Trade Execution (Blue)                  |  |
|  |  - Rate limiter (<= 1 post / 2s), 429 backoff, Pytest suppression           |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | $50k Paper Account Initialization | Initialize isolated paper trading account with exactly $50,000.00 cash balance in SQLite WAL | M1 | ORIGINAL_REQUEST R1 |
| 2 | Persistent Paper Portfolio Ledger | SQLite WAL tables tracking positions, cash balance, realized/unrealized P&L, order intents, and execution history across restarts | M1 | ORIGINAL_REQUEST R1 |
| 3 | AlpacaRelay Ingestion Client | REST and WebSocket client connecting to AlpacaRelay proxy with `RELAY_TOKEN` authentication | M1 | ORIGINAL_REQUEST R1 |
| 4 | Disconnect Detection & Fallback Simulation | Detect `upstream_disconnected` or REST failures, switch to synthetic display prices, set dashboard banner alert flag, and refuse all evaluation and trading until live | M1 | ORIGINAL_REQUEST R1 |
| 5 | 4-Regime Strategy Integration | Execute 4-regime dynamic long-term rebalancing logic, drift bands, vol scaling, and Antonacci safe-haven rotation on paper portfolio | M1 | ORIGINAL_REQUEST R1 |
| 6 | Discord v2 Broken Alert Card | Institutional red embed card (0xE53935) dispatched on disconnects or runtime errors with evidence and dashboard link | M2 | ORIGINAL_REQUEST R3 |
| 7 | Discord v2 Recovered Alert Card | Institutional green embed card (0x43A047) dispatched on reconnection/recovery with duration and metrics | M2 | ORIGINAL_REQUEST R3 |
| 8 | Discord v2 Trade Execution Card | Institutional blue embed card (0x1E88E5) dispatched on rebalance orders with fill details, weights, and NAV | M2 | ORIGINAL_REQUEST R3 |
| 9 | Discord Rate Limiting & Pytest Suppression | Honor Discord 429 rate limits with exponential backoff (<= 1 post/2s, max sleep 5s) and drop alerts under pytest | M2 | ORIGINAL_REQUEST R3 |
| 10 | Light & Airy Mobile-Centric Dashboard | Fast responsive web dashboard with light slate/white palette, crisp typography, clean card hierarchy | M3 | ORIGINAL_REQUEST R2 |
| 11 | Mobile Viewport Responsiveness | Flawless mobile rendering on 375px–430px viewports with touch-friendly (>=44px) targets and zero horizontal scroll | M3 | ORIGINAL_REQUEST R2 |
| 12 | Live Portfolio & Regime Metrics Display | Display live portfolio equity, $50k paper balance, active positions, strategy regime/signals, market news, connection health | M3 | ORIGINAL_REQUEST R2 |
| 13 | AlpacaRelay Disconnect Alert Banner | Prominent visual warning banner on dashboard triggered whenever AlpacaRelay is disconnected or operating in fallback | M3 | ORIGINAL_REQUEST R2 |
| 14 | Operator Real-Time Controls | Web controls for Pause, Resume, and Trigger Manual Rebalance evaluation | M3 | ORIGINAL_REQUEST R2 |
| 15 | Multi-Agent Adversarial Review & Smoke Testing | Phased adversarial code reviews and end-to-end UI smoke test with injected synthetic market and trade data | M4 | ORIGINAL_REQUEST R4 |
| 16 | Multi-Agent UI Design & Usability Audit | Multiple UI reviewer agents auditing operator-first simplicity, visual hierarchy, and absence of confusion | M4 | ORIGINAL_REQUEST R4 |
| 17 | Pristine State Reset for Monday's Open | Complete purge of synthetic test data, resetting paper portfolio to clean $50,000.00 cash, 0 positions, 0 open orders | M4 | ORIGINAL_REQUEST R4 |
| 18 | Dedicated Git Repository Setup | Initialize Git repo at `/Users/mo/DynamicLongTermStrategyBot`, commit all code with clean structured commits | M5 | ORIGINAL_REQUEST R5 |
| 19 | GitHub Remote Push (`Jhosshua`) | Create and push to GitHub repository under account `Jhosshua/DynamicLongTermStrategyBot` | M5 | ORIGINAL_REQUEST R5 |
| 20 | Public Token-Free Railway Deployment | Deploy service to Railway with a public read-only HTTPS URL; operator controls need `OPERATOR_TOKEN` | M5 | ORIGINAL_REQUEST R5 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Virtual Paper Trading Engine & Fallback Data Manager | Features 1, 2, 3, 4, 5: $50k WAL ledger, AlpacaRelay client, fallback simulation feed, 4-regime rebalancing execution | none | DONE |
| M2 | Discord v2 Institutional Alert Engine | Features 6, 7, 8, 9: Broken, recovered, trade execution embed cards, rate limiter, exponential backoff, pytest suppression | M1 | DONE |
| M3 | Mobile-Centric Light & Airy Operator Dashboard | Features 10, 11, 12, 13, 14: FastAPI + SSE server, light/airy UI, 375px–430px layout, alert banner, operator controls | M1 | DONE |
| M4 | Adversarial E2E Smoke Testing & Pristine Reset | Features 15, 16, 17: Multi-agent adversarial testing, injected fake data smoke test, UI review, wipe and reset to $50,000.00 for Monday | M1, M2, M3 | DONE |
| M5 | Dedicated Git Repository & Public Railway Deployment | Features 18, 19, 20: Git init, commit history, GitHub push to `Jhosshua`, Railway deployment, public URL and health check verification | M4 | DONE |

## Interface Contracts

### M1 (Paper Engine) ↔ M3 (Dashboard)
- `PaperAccountManager.get_portfolio_state() -> PortfolioSummary`:
  - `cash: float` (starts at 50000.00)
  - `equity: float` (total market value of open positions)
  - `total_nav: float` (cash + equity)
  - `realized_pnl: float`
  - `unrealized_pnl: float`
  - `positions: list[PositionDetail]` (`symbol`, `qty`, `avg_entry_price`, `current_price`, `market_value`, `unrealized_pnl`, `weight`)
- `DataFeedManager.get_connection_status() -> ConnectionStatus`:
  - `is_connected: bool`
  - `feed_source: str` ("alpaca_relay" or "synthetic_fallback")
  - `alert_banner_active: bool`
  - `last_heartbeat_timestamp: str`
- `PaperAccountManager.reset_to_pristine() -> None`:
  - Purges all execution history, open orders, and positions
  - Restores `cash = 50000.00, equity = 0.00, total_nav = 50000.00, positions = {}`

### M1 (Paper Engine) ↔ M2 (Discord Alerts)
- `DiscordNotifier.post_trade_execution(orders: list[RebalanceOrder], nav: float, regime: str, dashboard_url: str) -> bool`
- `DiscordNotifier.post_broken_alert(component: str, error_message: str, evidence: str, dashboard_url: str) -> bool`
- `DiscordNotifier.post_recovered_alert(component: str, downtime_duration_s: float, status_info: str, dashboard_url: str) -> bool`

### M3 (Dashboard Controls) ↔ Bot Daemon
All operator POSTs require header `X-Operator-Token` matching env `OPERATOR_TOKEN` (403 if unset in production, 401 if wrong). Dashboard: open `/#token=SECRET` once per browser.
- `POST /api/operator/pause`: Sets daemon state to `PAUSED`. Freezes rebalance evaluation.
- `POST /api/operator/resume`: Resets daemon state to `RUNNING`.
- `POST /api/operator/rebalance`: Out-of-cadence evaluation. `force` overrides PAUSE only, never a non-live feed (`REJECTED_UNSAFE_FEED`).
- `POST /api/operator/reset`: Wipes the paper account back to $50,000.
- `GET /health`: `status` is `"ok"` only when the feed is live and the service is RUNNING/PAUSED, else `"degraded"` (HTTP 200 either way).

## Code Layout
```
/Users/mo/DynamicLongTermStrategyBot/
├── strategy_engine/                   # Core quantitative engine (existing 857-test verified)
│   ├── allocator/                     # Allocation rules, rebalancer
│   ├── cli/                           # CLI commands (dry-run, daemon, status)
│   ├── core/                          # Enums, types, config
│   ├── daemon/                        # Scheduling daemon, execution loop
│   ├── ingestion/                     # AlpacaRelay REST & WS client
│   ├── signals/                       # Indicators, regime detector
│   ├── simulator/                     # Stress scenarios, synthetic generator
│   └── storage/                       # SQLite WAL storage & schema
├── bot/                               # Bot application layer
│   ├── __init__.py
│   ├── paper_account.py               # $50k virtual paper trading account & WAL ledger
│   ├── feed_manager.py                # AlpacaRelay client wrapper with auto fallback simulation
│   ├── discord_alerts.py              # Discord v2 institutional cards & rate limiter
│   └── service.py                     # Unified async service uniting daemon, feed, & web
├── web/                               # Mobile-Centric Light & Airy Operator Dashboard
│   ├── __init__.py
│   ├── app.py                         # FastAPI application, routes, SSE endpoints
│   ├── templates/
│   │   └── dashboard.html             # Light & airy responsive operator dashboard (375px–430px)
│   └── static/
│       ├── css/
│       │   └── custom.css             # Supplementary styling, clean card shadows, mobile tokens
│       └── js/
│           └── dashboard.js           # Live SSE client, operator action dispatchers
├── tests/
│   ├── e2e_suite/                     # Independent 4-tier requirement-driven E2E test suite
│   ├── unit/                          # Unit tests
│   ├── integration/                   # Integration tests
│   ├── e2e/                           # Existing end-to-end tests
│   └── adversarial/                   # Adversarial stress tests
├── scripts/
│   ├── e2e_smoke_test.py              # Synthetic data injection & full UI smoke test script
│   └── reset_pristine_for_monday.py   # Atomic reset restoring $50,000.00 cash & zeroing state
├── Dockerfile                         # Production container image for Railway
├── railway.json                       # Railway service configuration
└── pyproject.toml                     # Dependencies and project metadata
```
