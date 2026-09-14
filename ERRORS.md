# ERRORS.md — DynamicLongTermStrategyBot

## Local smoke test hit a stale server (2026-09-13)
- What did not work: `pkill -f "port 8931"` then restart. The old uvicorn
  ignored SIGTERM (the service had hijacked it), so the new one failed to bind
  and I re-tested old code twice.
- What worked: kill by PID, then fix the signal hijack itself.
- Note: after restarting a local server, confirm the PID changed before trusting results.

## Full test suite hung for an hour (2026-09-13)
- What did not work: refetching bars on every DAILY_CLOSE. Tests that inject
  their own bars made real relay calls, got 401, and stalled. A plain -q run
  shows no hint, and the faulthandler dump only shows the asyncio select loop.
- What worked: `-v -o log_cli=true` on the single test showed the 401 call.
  Refetch only when the cached SPY bar is >20h older than the evaluation time.
- Note: run with `-o faulthandler_timeout=120 -o faulthandler_exit_on_timeout=true`.

## Fills looked right in API but were fake (2026-09-13)
- What did not work: trusting `EXECUTED` from /api/operator/rebalance.
- What worked: compare `paper_trades.price` against the real last daily close in `market_bars`.
- Note: always check fill prices against a real source.
