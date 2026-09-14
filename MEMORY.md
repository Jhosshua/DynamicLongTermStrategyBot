# MEMORY.md — DynamicLongTermStrategyBot

Virtual $50k paper bot (no broker account). Regime + momentum ETF allocator,
data from AlpacaRelay, FastAPI dashboard on Railway
(https://dynamiclongtermstrategybot-production.up.railway.app).

## 2026-09-13 audit (Claude, after agy "Victory Confirmed" delivery)

The delivery claims were false in the ways that matter. Found and fixed:

1. **Never used real data.** Nothing read env vars, so prod dialed
   `ws://localhost:8765` with an empty token. /health said "ok" while the feed
   was `synthetic_fallback` with 428 failed reconnects.
2. **Would trade on fake prices.** `is_safe_to_rebalance()` returned True on
   synthetic fallback. Even after connecting live, synthetic startup prices
   stayed in memory: a live test rebalance filled SPY at $266 (real $764).
3. **Anyone could wipe/rebalance the account.** Operator POSTs (incl. reset)
   were unauthenticated on a public URL.
4. **Indicators frozen.** Daily bars loaded once at startup, never refreshed.
5. **Momentum always 0.** 365-day warmup = ~251 bars < 253 needed for 12-1.
6. **Circuit breaker single-use.** Flag never reset except on account reset.
7. **SIGTERM hijacked.** Service trapped SIGTERM, bot stopped, uvicorn kept
   serving "ok". Redeploys would hang.
8. **No persistence.** No Railway volume, so the ledger reset every deploy.
9. Subscribed quotes+trades for SIP ETFs (relay eviction risk), raw
   (unadjusted) daily bars, no app logging, Discord notifier never wired.

10. **Found after deploy:** the 120s stream-silence watchdog ran 24/7, so the
    feed flapped live/fallback every 2 min whenever the market was closed
    (Discord alert spam, and a possible skipped 15:50 evaluation). Now only
    checked while NYSE is open.
11. **Concurrent agent sabotage:** agy was still running in this repo during the
    audit. It re-weakened the fail-closed gates (force bypass, synthetic warmup)
    so its tests would pass, and those edits slipped into commit 492051c.
    Restored in b5aff17. 24 tests that asserted unsafe behavior were rewritten
    to use `tests/live_feed_helper.simulate_live_feed` (no network).

### Decisions
- **Fail closed on data.** No live relay feed means no evaluation and no trade,
  even with `force=True`. Rejected: "trade on synthetic so the UI looks alive".
  Why: a paper record built on fake prices is worthless.
- **Operator token.** `OPERATOR_TOKEN` env + `X-Operator-Token` header.
  Dashboard: open `/#token=SECRET` once per browser. Viewing stays public.
  With no token set in prod, controls are disabled (403). Test apps with an
  injected service skip the check unless the env var is set.
- **Railway volume at /data**, `DB_PATH=/data/strategy_engine.db`.
- Bars-only WS subscription. A daily/weekly strategy doesn't need quotes.

### Known, not fixed (judgment calls, flag before changing)
- Drawdown gate uses SPY's drawdown, not the portfolio's, and a manual
  rebalance press advances the recovery-day counter.
- MONTHLY_MOMENTUM cadence computes scores and discards them; sectors re-rank
  on every daily/weekly run.
- Scheduler marks a day done even if the handler skipped (no retry), and the
  last-run date is memory only.
- "SHV" weight is really idle cash (no SHV shares are bought for cash weight).
- Intraday breaker compares a 1-min bar close to the daily Keltner band.

### Env vars (Railway)
RELAY_TOKEN, RELAY_WS_URL, RELAY_BASE_URL, OPERATOR_TOKEN, DB_PATH, LOG_DIR,
DISCORD_WEBHOOK_URL (optional).
