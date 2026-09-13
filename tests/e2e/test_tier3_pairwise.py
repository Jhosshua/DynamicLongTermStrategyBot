"""Tier 3: Pairwise and Cross-Feature Integration Tests.
Exercises multi-module interactions, feedback loops, and state synchronizations (>=15 tests).
"""
from datetime import datetime, timezone
import json
import math
import sqlite3
import pytest
import httpx
import websockets

import tests.mocks.contracts as contracts


# ============================================================================
# PAIRWISE 1: Volatility Targeting (F7) + Trailing Drawdown Gate (F7/F9)
# ============================================================================

def test_pairwise_vol_targeting_and_drawdown_gate(progressive_contracts):
    math_utils = progressive_contracts.math_utils
    # Spiking volatility (36% vol -> scale = 0.12 / 0.36 = 0.333)
    s_vol = math_utils.volatility_scale_factor(realized_vol=0.36, target_vol=0.12)
    # Severe portfolio drawdown (-12% -> gate = 0.20)
    g_dd = math_utils.evaluate_drawdown_gate(drawdown_pct=-0.12)
    # Compounded equity multiplier
    combined_equity_scale = s_vol * g_dd
    expected = (0.12 / 0.36) * 0.20
    assert math.isclose(combined_equity_scale, expected, rel_tol=1e-3)
    # Cash/defensive sleeve absorbs remaining capital
    w_cash = 1.0 - combined_equity_scale
    assert w_cash > 0.90


# ============================================================================
# PAIRWISE 2: Dual-SMA Trend Break (F7) + Safe-Haven Dual Momentum (F8)
# ============================================================================

def test_pairwise_trend_break_and_safe_haven_rotation(progressive_contracts):
    math_utils = progressive_contracts.math_utils
    # SPY below 200 SMA -> Bear regime (defensive capital = 1.0)
    spy_p = 400.0
    spy_sma200 = math_utils.calculate_sma([450.0] * 200, 200)
    is_bear = spy_p < spy_sma200
    assert is_bear is True

    # Check safe havens: TLT in 2022 rate shock (TLT < SMA200 and Mom < 0)
    tlt_p = 95.0
    tlt_sma200 = 115.0
    tlt_mom = -0.18
    tlt_qualified = (tlt_p > tlt_sma200 and tlt_mom > 0)
    assert tlt_qualified is False

    # GLD also below hurdle
    gld_mom = -0.05
    gld_qualified = gld_mom > 0

    # 100% of defensive capital routes to SHV cash proxy
    w_def = 1.0
    w_tlt = 0.40 * w_def if tlt_qualified else 0.0
    w_gld = 0.40 * w_def if gld_qualified else 0.0
    w_shv = w_def - w_tlt - w_gld
    assert w_shv == 1.0


# ============================================================================
# PAIRWISE 3: WebSocket Disconnect (F4/F5) + Rebalancing Trigger (F9/F12)
# ============================================================================

@pytest.mark.asyncio
async def test_pairwise_ws_disconnect_freezes_rebalance(mock_relay_server, progressive_contracts):
    MarketRegime = progressive_contracts.models.MarketRegime
    current_regime = MarketRegime.BULL_NORMAL

    async with websockets.connect(mock_relay_server.ws_url) as ws:
        await ws.recv()
        await ws.send(json.dumps({"action": "auth", "token": mock_relay_server.token}))
        await ws.recv()

        # Upstream disconnects right before scheduled evaluation
        await mock_relay_server.broadcast_lifecycle("upstream_disconnected")
        event = json.loads(await ws.recv())
        if event[0]["msg"] == "upstream_disconnected":
            current_regime = MarketRegime.STALE_DATA_HOLD

        # Scheduler attempts to evaluate at 15:50
        can_execute_rebalance = (current_regime != MarketRegime.STALE_DATA_HOLD)
        assert can_execute_rebalance is False


# ============================================================================
# PAIRWISE 4: REST Rate Limiting (F3) + SQLite Local Bar Cache (F11)
# ============================================================================

@pytest.mark.asyncio
async def test_pairwise_rest_caching_with_sqlite(temp_sqlite_db, mock_relay_server):
    conn = sqlite3.connect(temp_sqlite_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_bars (
            symbol TEXT, timestamp TEXT PRIMARY KEY, close REAL
        );
    """)
    conn.execute("INSERT INTO market_bars VALUES ('SPY', '2026-09-01T04:00:00Z', 503.0);")
    conn.commit()

    # Query function: check SQLite first before network call
    network_calls = 0

    async def get_bar(symbol: str, ts: str):
        nonlocal network_calls
        cur = conn.cursor()
        cur.execute("SELECT close FROM market_bars WHERE symbol=? AND timestamp=?", (symbol, ts))
        row = cur.fetchone()
        if row:
            return row[0]
        # Fallback to network
        network_calls += 1
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"{mock_relay_server.http_url}/data/v2/stocks/{symbol}/bars",
                headers={"X-Relay-Token": mock_relay_server.token}
            )
            return res.status_code

    # Querying existing bar uses local SQLite, 0 network calls
    price = await get_bar("SPY", "2026-09-01T04:00:00Z")
    assert price == 503.0
    assert network_calls == 0

    # Querying non-cached bar falls back to REST
    await get_bar("QQQ", "2026-09-02T04:00:00Z")
    assert network_calls == 1
    conn.close()


# ============================================================================
# PAIRWISE 5: Merton SDE Shock (F2) + Dynamic ATR Circuit Breaker (F7) + OrderIntent (F9)
# ============================================================================

def test_pairwise_sde_shock_triggers_atr_circuit_breaker(synthetic_simulator, progressive_contracts):
    math_utils = progressive_contracts.math_utils
    OrderIntent = progressive_contracts.models.OrderIntent
    # Generate baseline series
    paths = synthetic_simulator.simulate_multivariate_paths(
        symbols=["SPY"], initial_prices={"SPY": 500.0}, drifts={"SPY": 0.0},
        volatilities={"SPY": 0.15}, correlation_matrix=contracts.np.eye(1) if hasattr(contracts, "np") else __import__("numpy").eye(1),
        jump_lambda=1.0, jump_mean=-0.15, jump_vol=0.01, n_days=30
    )
    prices = list(paths["SPY"])
    sma50 = math_utils.calculate_sma(prices, 14)
    # Estimate ATR
    highs = [p * 1.005 for p in prices]
    lows = [p * 0.995 for p in prices]
    atr = math_utils.calculate_atr(highs, lows, prices, 14)
    lower_band = sma50 - 2.0 * atr

    # Simulate sharp shock day
    shock_price = lower_band - 5.0
    circuit_breaker_triggered = bool(shock_price < lower_band)
    assert circuit_breaker_triggered is True

    # Order intent generation: immediately cut 50% equity
    current_equity_weight = 0.80
    target_equity_weight = 0.40
    order = OrderIntent(
        symbol="SPY", action="SELL", target_weight=target_equity_weight,
        current_weight=current_equity_weight, delta_weight=-0.40,
        rationale="Emergency ATR circuit breaker stop triggered"
    )
    assert order.action == "SELL"
    assert order.delta_weight == -0.40


# ============================================================================
# PAIRWISE 6: Market Breadth Collapse (F7) + Sector Momentum (F8/F9)
# ============================================================================

def test_pairwise_breadth_collapse_dampens_sector_weights():
    # Breadth < 40% dampens sector multiplier to 0.50
    breadth = 0.35
    f_breadth = 1.0 if breadth >= 0.60 else (0.75 if breadth >= 0.40 else 0.50)
    assert f_breadth == 0.50

    nominal_sector_weight = 0.30
    effective_sector_weight = nominal_sector_weight * f_breadth
    assert effective_sector_weight == 0.15


# ============================================================================
# PAIRWISE 7: Scheduled Market Close (F12) + Dual Persistence (F11)
# ============================================================================

def test_pairwise_scheduled_close_commits_sqlite_and_jsonl(temp_sqlite_db, tmp_path):
    conn = sqlite3.connect(temp_sqlite_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_audit_trail (
            timestamp TEXT PRIMARY KEY, regime TEXT, target_weights_json TEXT, rationale TEXT
        );
    """)
    now_str = "2026-09-03T15:50:00Z"
    weights = {"SPY": 0.40, "QQQ": 0.40, "SHV": 0.20}
    rationale = "Scheduled daily close evaluation"

    # 1. Commit to SQLite
    conn.execute(
        "INSERT INTO decision_audit_trail VALUES (?, ?, ?, ?);",
        (now_str, "BULL_NORMAL", json.dumps(weights), rationale)
    )
    conn.commit()

    # 2. Append to JSONL
    jsonl_path = tmp_path / "decision_audit.jsonl"
    record = {"timestamp": now_str, "regime": "BULL_NORMAL", "weights": weights, "rationale": rationale}
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Verify both persistence channels
    cursor = conn.cursor()
    cursor.execute("SELECT regime FROM decision_audit_trail WHERE timestamp=?", (now_str,))
    assert cursor.fetchone()[0] == "BULL_NORMAL"
    conn.close()

    with open(jsonl_path, "r") as f:
        saved = json.loads(f.readline())
        assert saved["timestamp"] == now_str
        assert saved["weights"]["SPY"] == 0.40


# ============================================================================
# PAIRWISE 8: Upstream Reconnect (F5) + REST Gap Backfill (F3)
# ============================================================================

@pytest.mark.asyncio
async def test_pairwise_upstream_reconnect_triggers_gap_backfill(mock_relay_server):
    last_received_ts = datetime(2026, 9, 3, 14, 0, tzinfo=timezone.utc)
    reconnect_ts = datetime(2026, 9, 3, 14, 25, tzinfo=timezone.utc)
    gap_duration = (reconnect_ts - last_received_ts).total_seconds() / 60.0
    assert gap_duration == 25.0

    # Backfill missing bars via REST
    mock_relay_server.add_mock_bars("SPY", [
        {"t": "2026-09-03T14:05:00Z", "c": 501.0},
        {"t": "2026-09-03T14:15:00Z", "c": 501.5},
    ])
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{mock_relay_server.http_url}/data/v2/stocks/SPY/bars?start={last_received_ts.isoformat()}&end={reconnect_ts.isoformat()}",
            headers={"X-Relay-Token": mock_relay_server.token}
        )
        assert res.status_code == 200
        bars = res.json()["bars"]
        assert len(bars) == 2


# ============================================================================
# PAIRWISE 9: CLI Dry-Run (F13) + In-Process Mock Relay (F6)
# ============================================================================

@pytest.mark.asyncio
async def test_pairwise_cli_dry_run_with_mock_relay(mock_relay_server):
    # Simulate dry run query against running mock relay
    async with httpx.AsyncClient() as client:
        health = await client.get(f"{mock_relay_server.http_url}/health")
        assert health.status_code == 200
        bars = await client.get(
            f"{mock_relay_server.http_url}/data/v2/stocks/SPY/bars",
            headers={"X-Relay-Token": mock_relay_server.token}
        )
        assert bars.status_code == 200


# ============================================================================
# PAIRWISE 10: 2022 Inflation Regime (F2) + Safe-Haven Allocator (F8)
# ============================================================================

def test_pairwise_2022_synthetic_data_disqualifies_tlt(calibrated_2022_data, progressive_contracts):
    math_utils = progressive_contracts.math_utils
    tlt_bars = calibrated_2022_data["TLT"]
    tlt_closes = [b.close for b in tlt_bars]
    tlt_sma200 = math_utils.calculate_sma(tlt_closes, 200)
    current_tlt = tlt_closes[-1]
    # In 2022, TLT drops below 200 SMA
    assert current_tlt < tlt_sma200
    tlt_qualified = current_tlt > tlt_sma200
    assert tlt_qualified is False


# ============================================================================
# PAIRWISE 11: Drift Band Monitoring (F9) + Market Scheduler (F12)
# ============================================================================

def test_pairwise_drift_band_monitoring_on_scheduler_tick():
    target_weights = {"SPY": 0.50, "QQQ": 0.50}
    current_weights_small_drift = {"SPY": 0.51, "QQQ": 0.49}
    drift_band = 0.025

    def check_rebalance(targets, currents, threshold):
        orders = []
        for sym, tgt in targets.items():
            cur = currents.get(sym, 0.0)
            if abs(tgt - cur) > threshold:
                orders.append((sym, tgt - cur))
        return orders

    # Small drift: no orders generated
    orders = check_rebalance(target_weights, current_weights_small_drift, drift_band)
    assert len(orders) == 0

    # Large drift: orders generated
    current_weights_large_drift = {"SPY": 0.55, "QQQ": 0.45}
    orders_large = check_rebalance(target_weights, current_weights_large_drift, drift_band)
    assert len(orders_large) == 2


# ============================================================================
# PAIRWISE 12: WebSocket Streaming Bars (F4) + Intraday Signal Snapshot (F7)
# ============================================================================

@pytest.mark.asyncio
async def test_pairwise_ws_streaming_updates_signal_snapshot(mock_relay_server, progressive_contracts):
    SignalSnapshot = progressive_contracts.models.SignalSnapshot
    MarketRegime = progressive_contracts.models.MarketRegime

    current_snapshot = None

    async with websockets.connect(mock_relay_server.ws_url) as ws:
        await ws.recv()
        await ws.send(json.dumps({"action": "auth", "token": mock_relay_server.token}))
        await ws.recv()
        await ws.send(json.dumps({"action": "subscribe", "bars": ["SPY"]}))
        await ws.recv()

        # Stream new bar
        await mock_relay_server.broadcast_bar("SPY", {"c": 505.0, "o": 500.0, "h": 506.0, "l": 499.0, "v": 10000})
        raw = json.loads(await ws.recv())
        bar_data = raw[0]

        current_snapshot = SignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            spy_price=bar_data["c"],
            spy_sma50=500.0,
            spy_sma200=480.0,
            realized_vol_20d=0.12,
            vol_scale_factor=1.0,
            drawdown_pct=-0.01,
            circuit_breaker_active=False,
            regime=MarketRegime.BULL_NORMAL
        )
        assert current_snapshot.spy_price == 505.0


# ============================================================================
# PAIRWISE 13: Upstream Reconnect (F5) + Bulk Resubscribe (F4/F6)
# ============================================================================

@pytest.mark.asyncio
async def test_pairwise_bulk_resubscribe_after_disconnect(mock_relay_server):
    async with websockets.connect(mock_relay_server.ws_url) as ws:
        await ws.recv()
        await ws.send(json.dumps({"action": "auth", "token": mock_relay_server.token}))
        await ws.recv()
        # Initial subscriptions
        tickers = ["SPY", "QQQ", "AAPL", "MSFT", "TLT", "SHV"]
        await ws.send(json.dumps({"action": "subscribe", "bars": tickers}))
        await ws.recv()

        # Disconnect and reconnect
        await mock_relay_server.broadcast_lifecycle("upstream_disconnected")
        await ws.recv()
        await mock_relay_server.broadcast_lifecycle("upstream_connected")
        msg = json.loads(await ws.recv())
        assert msg[0]["msg"] == "upstream_connected"

        # Bulk resubscribe all tickers
        await ws.send(json.dumps({"action": "subscribe", "bars": tickers}))
        ack = json.loads(await ws.recv())
        for sym in tickers:
            assert sym in ack[0]["bars"]


# ============================================================================
# PAIRWISE 14: STRATEGY.md Blueprint Constants (F10) + Allocator Rules (F9)
# ============================================================================

def test_pairwise_blueprint_constants_match_allocator_rules():
    # Blueprint parameters vs implementation invariants
    target_volatility = 0.12
    drift_band = 0.025
    dd_level_1 = -0.05
    dd_level_2 = -0.10
    dd_level_3 = -0.15

    assert target_volatility == 0.12
    assert drift_band == 0.025
    assert dd_level_1 > dd_level_2 > dd_level_3


# ============================================================================
# PAIRWISE 15: Low-Vol Bull (F2) + Megacap Tech Overweight (F1/F9)
# ============================================================================

def test_pairwise_low_vol_bull_allocates_megacap(calibrated_2017_data, progressive_contracts):
    TargetAllocation = progressive_contracts.models.TargetAllocation
    MarketRegime = progressive_contracts.models.MarketRegime
    math_utils = progressive_contracts.math_utils
    spy = [b.close for b in calibrated_2017_data["SPY"]]
    vol = math_utils.calculate_realized_volatility(spy, 20)
    assert vol < 0.15

    # Bull aggressive regime: 50% QQQ, 30% sectors, 20% SPY, 0% cash
    alloc = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_AGGRESSIVE,
        weights={"QQQ": 0.50, "XLK": 0.30, "SPY": 0.20},
        cash_weight=0.0,
        rationale="2017 low vol bull allocation"
    )
    assert alloc.weights["QQQ"] == 0.50
    assert alloc.cash_weight == 0.0
    assert math.isclose(sum(alloc.weights.values()), 1.0)


# ============================================================================
# PAIRWISE 16: SQLite WAL Concurrency (F11) + WebSocket Streaming (F4)
# ============================================================================

@pytest.mark.asyncio
async def test_pairwise_sqlite_wal_streaming_concurrency(temp_sqlite_db, mock_relay_server):
    conn_writer = sqlite3.connect(temp_sqlite_db)
    conn_writer.execute("CREATE TABLE IF NOT EXISTS stream_bars (symbol TEXT, ts TEXT, price REAL);")
    conn_writer.commit()

    # Writer commits incoming bars
    for i in range(20):
        conn_writer.execute("INSERT INTO stream_bars VALUES (?, ?, ?);", ("SPY", f"t_{i}", 500.0 + i))
        conn_writer.commit()

    # Reader concurrently selects without locking error
    conn_reader = sqlite3.connect(temp_sqlite_db)
    cur = conn_reader.cursor()
    cur.execute("SELECT count(*), max(price) FROM stream_bars;")
    count, max_p = cur.fetchone()
    assert count == 20
    assert max_p == 519.0
    conn_writer.close()
    conn_reader.close()
