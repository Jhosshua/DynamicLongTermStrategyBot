# MEMORY.md — DynamicLongTermStrategyBot

Virtual $50k paper bot (no broker account). Regime + momentum ETF allocator,
data from AlpacaRelay, FastAPI dashboard on Railway
(https://dynamiclongtermstrategybot-production.up.railway.app).
Deploy: the Railway service is GitHub-connected (source repo Jhosshua/DynamicLongTermStrategyBot),
so `git push origin main` redeploys. `railway up --detach` also works as a fallback.

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

### 2026-09-15: watch day, Discord alert grace period
- Watched the bot live all day (Tuesday). Monday's 15:50 DAILY_CLOSE ran and
  produced target weights (QQQ 50 / SPY 20 / XLK 15 / XLE 15) with zero trades.
  That is by design: DAILY_CLOSE only evaluates; trades happen on the Friday
  WEEKLY_REBALANCE (or the intraday breaker). First real fills expected Fri 09-18 15:50 ET.
- Found: Railway's edge cuts the relay websocket every few minutes for EVERY
  relay client (code 1006, no close frame, relay logs ~225 client drops in 2h
  across ~9 clients, no 1013 "too slow" evictions). The bot reconnects in ~2.6s.
  Each blip posted a BROKEN and a RECOVERED Discord card: 345 posts in 24h.
- Fix: `ServiceConfig.alert_grace_seconds` (90s). BROKEN posts only if the feed
  is still down after the grace; RECOVERED only follows a posted BROKEN.
  Tests in tests/unit/test_alert_grace_period.py. Full suite 1317 passing.
- Rejected: fixing the drops at the relay (root cause is the edge proxy, would
  need private networking on IPv6 and touches every bot) and lengthening the
  websocket ping timeout (the client sees no close frame, pings are not the trigger).
- Second fix (08:55 ET): after the redeploy the dashboard showed regime UNKNOWN
  and the breaker had no Keltner band, because the last signal/allocation lived
  only in memory. `start()` now calls `_restore_latest_decision()` which reloads
  the latest saved signal + allocation (ignored if older than 5 days).
  Tests in tests/unit/test_restore_latest_decision.py. Suite 1320 passing.
- Not fixed: `reconnect_attempts` in /health keeps counting every blip (cosmetic).

## Known, not fixed (judgment calls, ask before changing)
- The drawdown gate uses SPY's drawdown, not the portfolio's, and a manual
  rebalance press advances the recovery-day counter.
- MONTHLY_MOMENTUM computes scores and discards them; sectors re-rank on
  every daily/weekly run.
- "SHV" weight is really idle cash (no SHV shares are bought for the cash weight).
- The intraday breaker compares a 1-min bar close to the daily Keltner band.
- The weekly rebalance can use the previous day's allocation if today's
  daily close skipped but the feed is safe by the weekly run.
