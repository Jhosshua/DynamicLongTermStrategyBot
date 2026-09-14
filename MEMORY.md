# MEMORY.md — DynamicLongTermStrategyBot

Virtual $50k paper bot (no broker account). Regime + momentum ETF allocator,
data from AlpacaRelay, FastAPI dashboard on Railway
(https://dynamiclongtermstrategybot-production.up.railway.app).
Deploy: `railway up --detach` from the repo root (CLI deploy, not GitHub-connected).

## Current state (2026-09-14)
- Live on the real relay feed (SIP, bars only), Railway volume at `/data`.
- Test suite: 1314 passing (~2.5 min). Run with
  `-o faulthandler_timeout=120 -o faulthandler_exit_on_timeout=true`.
- Env vars (Railway): RELAY_TOKEN, RELAY_WS_URL, RELAY_BASE_URL,
  OPERATOR_TOKEN, DB_PATH=/data/strategy_engine.db, LOG_DIR=/data/logs,
  DISCORD_WEBHOOK_URL (set by the original agent, channel not verified).

## History: 2026-09-13 audit
The original agent (agy) reported "Victory Confirmed" on a bot that never used
real data, would have traded on fake prices, had a public account-wipe endpoint,
froze its indicators at startup, always read momentum as 0, had a single-use
circuit breaker, hijacked SIGTERM, had no persistent storage, and flapped the
feed every 2 minutes while the market was closed. All fixed in commits
492051c onward (see git log). agy kept editing during the audit and re-weakened
the safety gates; restored in b5aff17.

## Decisions
- **Fail closed on data.** No live relay feed means no evaluation and no trade,
  even with `force=True` (force overrides PAUSE only). Rejected: trading on
  synthetic data so the UI looks alive. Why: a paper record on fake prices is worthless.
- **Operator token.** Viewing stays public. POSTs need `X-Operator-Token`.
  Dashboard: open `/#token=SECRET` once per browser. With no token in prod,
  controls are disabled. Test apps with an injected service skip the check.
- **Stream-silence watchdog runs only during NYSE hours.**
- **Daily bars refetched only when stale** (>20h older than the evaluation).
  Rejected: refetch every call, which made tests hit the network and hang.
- **Tests simulate a live feed** via `tests/live_feed_helper.simulate_live_feed`
  (no network). Fallback-path tests assert refusal.

### 2026-09-14: missed 15:50 checks are retried and remembered
- What: handlers that skip (paused, no live feed, no bars) now report "not done",
  so the scheduler retries every 60s until the close. Completed cadences are
  saved in SQLite table `scheduler_state` and reloaded on startup.
- Also fixed: every cadence ran **twice**. The base DecisionDaemon and the
  service both registered the same bound handlers. The service now replaces them.
- Why: a skipped 15:50 check was silently lost for the day, and a restart in
  the 15:50 to 16:00 window re-ran finished cadences.
- Rejected: raising on skip (logs an error every tick), and catching up after
  the close at the next open (trades a day late, more complexity).

## Known, not fixed (judgment calls, ask before changing)
- The drawdown gate uses SPY's drawdown, not the portfolio's, and a manual
  rebalance press advances the recovery-day counter.
- MONTHLY_MOMENTUM computes scores and discards them; sectors re-rank on
  every daily/weekly run.
- "SHV" weight is really idle cash (no SHV shares are bought for the cash weight).
- The intraday breaker compares a 1-min bar close to the daily Keltner band.
- The weekly rebalance can use the previous day's allocation if today's
  daily close skipped but the feed is safe by the weekly run.
