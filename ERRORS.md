# ERRORS.md — DynamicLongTermStrategyBot

## Local smoke test hit a stale server (2026-09-13)
- What did not work: `pkill -f "port 8931"` then restart. The old uvicorn
  ignored SIGTERM (the service had hijacked it), so the new one failed to bind
  and I re-tested old code twice.
- What worked: kill by PID, then fix the signal hijack itself.
- Note: after restarting a local server, confirm the PID changed before trusting results.

## Fills looked right in API but were fake (2026-09-13)
- What did not work: trusting `EXECUTED` from /api/operator/rebalance.
- What worked: compare `paper_trades.price` against the real last daily close in `market_bars`.
- Note: always check fill prices against a real source.
