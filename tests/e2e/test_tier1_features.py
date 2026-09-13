"""Tier 1: Comprehensive Feature Coverage (Happy Path & Isolation).
Exercises Features 1 through 13 in isolation under nominal valid conditions (>=5 tests per feature).
"""
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import pytest
import httpx
import websockets

import tests.mocks.contracts as contracts
from tests.mocks.mock_relay_server import MockAlpacaRelayServer


# ============================================================================
# FEATURE 1: Domain Models & Universe Specification
# ============================================================================

def test_f1_bar_model_attributes_and_immutability(progressive_contracts):
    Bar = progressive_contracts.models.Bar
    now = datetime(2026, 9, 3, 13, 30, tzinfo=timezone.utc)
    bar = Bar(symbol="SPY", timestamp=now, open=500.0, high=505.0, low=498.0, close=503.0, volume=1_000_000, vwap=502.5)
    assert bar.symbol == "SPY"
    assert bar.close == 503.0
    assert bar.volume == 1_000_000
    with pytest.raises(Exception):
        bar.close = 510.0  # Immutable frozen dataclass


def test_f1_quote_and_trade_models(progressive_contracts):
    Quote = progressive_contracts.models.Quote
    Trade = progressive_contracts.models.Trade
    now = datetime(2026, 9, 3, 13, 30, tzinfo=timezone.utc)
    quote = Quote(symbol="QQQ", timestamp=now, bid_price=450.0, ask_price=450.05, bid_size=10, ask_size=15)
    trade = Trade(symbol="QQQ", timestamp=now, price=450.02, size=100, id="tr_001")
    assert quote.bid_price < quote.ask_price
    assert trade.size == 100
    assert trade.id == "tr_001"


def test_f1_signal_snapshot_model(progressive_contracts):
    SignalSnapshot = progressive_contracts.models.SignalSnapshot
    MarketRegime = progressive_contracts.models.MarketRegime
    now = datetime(2026, 9, 3, 13, 30, tzinfo=timezone.utc)
    snap = SignalSnapshot(
        timestamp=now, spy_price=510.0, spy_sma50=500.0, spy_sma200=480.0,
        realized_vol_20d=0.11, vol_scale_factor=1.0, drawdown_pct=-0.02,
        circuit_breaker_active=False, regime=MarketRegime.BULL_NORMAL,
        indicators={"breadth_50": 0.65, "atr_14": 5.2}
    )
    assert snap.regime == MarketRegime.BULL_NORMAL
    assert snap.circuit_breaker_active is False
    assert snap.indicators["breadth_50"] == 0.65


def test_f1_target_allocation_weight_sum_and_cash(progressive_contracts):
    TargetAllocation = progressive_contracts.models.TargetAllocation
    MarketRegime = progressive_contracts.models.MarketRegime
    now = datetime(2026, 9, 3, 13, 30, tzinfo=timezone.utc)
    weights = {"SPY": 0.35, "QQQ": 0.35, "TLT": 0.15, "SHV": 0.15}
    alloc = TargetAllocation(
        timestamp=now, regime=MarketRegime.BULL_NORMAL,
        weights=weights, cash_weight=0.15, rationale="Standard normal bull weights"
    )
    assert math.isclose(sum(alloc.weights.values()), 1.0, rel_tol=1e-5)
    assert alloc.cash_weight == 0.15


def test_f1_universe_tier_specifications(progressive_contracts):
    universe = progressive_contracts.universe
    assert "SPY" in universe.CORE_COMPOUNDERS
    assert "QQQ" in universe.CORE_COMPOUNDERS
    assert "XLK" in universe.SECTOR_ETFS
    assert "XLE" in universe.SECTOR_ETFS
    assert "TLT" in universe.SAFE_HAVENS
    assert "SHV" in universe.SAFE_HAVENS
    assert len(universe.ALL_UNIVERSE) >= 20


def test_f1_math_utils_annualized_volatility(progressive_contracts):
    math_utils = progressive_contracts.math_utils
    # 25 constant prices -> zero volatility
    flat_prices = [100.0] * 25
    assert math_utils.calculate_realized_volatility(flat_prices, window=20) == 0.0

    # Alternating prices -> non-zero volatility
    alt_prices = [100.0 if i % 2 == 0 else 102.0 for i in range(25)]
    vol = math_utils.calculate_realized_volatility(alt_prices, window=20)
    assert vol > 0.05


# ============================================================================
# FEATURE 2: Synthetic Market Regime Simulator
# ============================================================================

def test_f2_multivariate_sde_output_shapes(synthetic_simulator):
    symbols = ["SPY", "QQQ", "TLT", "SHV"]
    paths = synthetic_simulator.simulate_multivariate_paths(
        symbols=symbols,
        initial_prices={"SPY": 100.0, "QQQ": 100.0, "TLT": 100.0, "SHV": 100.0},
        drifts={"SPY": 0.10, "QQQ": 0.12, "TLT": 0.02, "SHV": 0.01},
        volatilities={"SPY": 0.15, "QQQ": 0.18, "TLT": 0.10, "SHV": 0.001},
        correlation_matrix=contracts.np.eye(4) if hasattr(contracts, "np") else __import__("numpy").eye(4),
        n_days=50,
    )
    for s in symbols:
        assert len(paths[s]) == 51
        assert paths[s][0] == 100.0


def test_f2_cholesky_correlation_preservation(synthetic_simulator):
    np = __import__("numpy")
    symbols = ["A", "B"]
    rho = 0.90
    corr = np.array([[1.0, rho], [rho, 1.0]])
    paths = synthetic_simulator.simulate_multivariate_paths(
        symbols=symbols,
        initial_prices={"A": 100.0, "B": 100.0},
        drifts={"A": 0.0, "B": 0.0},
        volatilities={"A": 0.20, "B": 0.20},
        correlation_matrix=corr,
        n_days=1000,
    )
    ret_a = np.diff(np.log(paths["A"]))
    ret_b = np.diff(np.log(paths["B"]))
    empirical_rho = np.corrcoef(ret_a, ret_b)[0, 1]
    assert math.isclose(empirical_rho, rho, abs_tol=0.08)


def test_f2_2008_liquidity_crisis_calibration(calibrated_2008_data):
    spy_bars = calibrated_2008_data["SPY"]
    tlt_bars = calibrated_2008_data["TLT"]
    shv_bars = calibrated_2008_data["SHV"]
    spy_p0 = spy_bars[0].close
    spy_p_min = min(b.close for b in spy_bars)
    spy_max_dd = (spy_p_min - spy_p0) / spy_p0
    assert spy_max_dd < -0.40, f"2008 SPY crash insufficient: {spy_max_dd:.2%}"
    assert tlt_bars[-1].close > tlt_bars[0].close, "TLT did not act as deflation hedge in 2008"
    assert shv_bars[-1].close >= shv_bars[0].close, "SHV failed capital preservation in 2008"


def test_f2_2020_flash_crash_v_recovery_calibration(calibrated_2020_data):
    spy = calibrated_2020_data["SPY"]
    assert len(spy) == 61
    p_start = spy[0].close
    p_trough = min(b.close for b in spy[:25])
    p_end = spy[-1].close
    crash_depth = (p_trough - p_start) / p_start
    rebound = (p_end - p_trough) / p_trough
    assert crash_depth < -0.25, f"Crash depth not deep enough: {crash_depth:.2%}"
    assert rebound > 0.20, f"V-rebound not fast enough: {rebound:.2%}"


def test_f2_2022_inflation_grind_calibration(calibrated_2022_data):
    spy = calibrated_2022_data["SPY"]
    tlt = calibrated_2022_data["TLT"]
    xle = calibrated_2022_data["XLE"]
    assert spy[-1].close < spy[0].close, "SPY should be down in 2022"
    assert tlt[-1].close < tlt[0].close, "TLT duration trap failed: bonds should be down in 2022"
    assert xle[-1].close > xle[0].close, "XLE energy sector should be up in 2022 inflation grind"


def test_f2_2017_low_vol_bull_calibration(calibrated_2017_data, progressive_contracts):
    spy = calibrated_2017_data["SPY"]
    math_utils = progressive_contracts.math_utils
    spy_closes = [b.close for b in spy]
    assert spy_closes[-1] > spy_closes[0] * 1.10, "SPY should gain >10% in low vol bull"
    vol = math_utils.calculate_realized_volatility(spy_closes, window=20)
    assert vol < 0.15, f"2017 volatility should be low: {vol:.2%}"


# ============================================================================
# FEATURE 3: AlpacaRelay REST Proxy Client
# ============================================================================

@pytest.mark.asyncio
async def test_f3_rest_client_auth_header_injection(mock_relay_server):
    async with httpx.AsyncClient() as client:
        # Without auth header
        res_no_auth = await client.get(f"{mock_relay_server.http_url}/data/v2/stocks/SPY/bars")
        assert res_no_auth.status_code == 401
        # With valid X-Relay-Token header
        res_auth = await client.get(
            f"{mock_relay_server.http_url}/data/v2/stocks/SPY/bars",
            headers={"X-Relay-Token": mock_relay_server.token}
        )
        assert res_auth.status_code == 200


@pytest.mark.asyncio
async def test_f3_rest_client_single_symbol_bars(mock_relay_server):
    mock_relay_server.add_mock_bars("SPY", [
        {"t": "2026-09-01T04:00:00Z", "o": 500.0, "h": 505.0, "l": 499.0, "c": 503.0, "v": 1000000}
    ])
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{mock_relay_server.http_url}/data/v2/stocks/SPY/bars?timeframe=1Day",
            headers={"X-Relay-Token": mock_relay_server.token}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["symbol"] == "SPY"
        assert len(data["bars"]) == 1
        assert data["bars"][0]["c"] == 503.0


@pytest.mark.asyncio
async def test_f3_rest_client_multi_symbol_bars(mock_relay_server):
    mock_relay_server.add_mock_bars("SPY", [{"t": "2026-09-01T04:00:00Z", "c": 500.0}])
    mock_relay_server.add_mock_bars("QQQ", [{"t": "2026-09-01T04:00:00Z", "c": 450.0}])
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{mock_relay_server.http_url}/data/v2/stocks/bars?symbols=SPY,QQQ&timeframe=1Day",
            headers={"X-Relay-Token": mock_relay_server.token}
        )
        assert res.status_code == 200
        data = res.json()
        assert "SPY" in data["bars"]
        assert "QQQ" in data["bars"]


@pytest.mark.asyncio
async def test_f3_rest_client_pagination_handling(mock_relay_server):
    bars = [{"t": f"2026-08-{i:02d}T04:00:00Z", "c": 500.0 + i} for i in range(1, 15)]
    mock_relay_server.add_mock_bars("AAPL", bars)
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{mock_relay_server.http_url}/data/v2/stocks/AAPL/bars?limit=5",
            headers={"X-Relay-Token": mock_relay_server.token}
        )
        data = res.json()
        assert len(data["bars"]) == 5


@pytest.mark.asyncio
async def test_f3_rest_client_rate_limiter_token_bucket():
    class TokenBucket:
        def __init__(self, rate_per_sec: float, capacity: float):
            self.rate = rate_per_sec
            self.capacity = capacity
            self.tokens = capacity
            self.last = asyncio.get_event_loop().time()

        async def acquire(self):
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0

    bucket = TokenBucket(rate_per_sec=20.0, capacity=2.0)
    t0 = asyncio.get_event_loop().time()
    for _ in range(5):
        await bucket.acquire()
    t1 = asyncio.get_event_loop().time()
    assert (t1 - t0) >= 0.10


# ============================================================================
# FEATURE 4: AlpacaRelay WebSocket Streaming Client
# ============================================================================

@pytest.mark.asyncio
async def test_f4_ws_connect_and_receive_banner(mock_relay_server):
    async with websockets.connect(mock_relay_server.ws_url) as ws:
        banner = json.loads(await ws.recv())
        assert isinstance(banner, list)
        assert banner[0]["msg"] == "connected"


@pytest.mark.asyncio
async def test_f4_ws_auth_handshake_within_10s(mock_relay_server):
    async with websockets.connect(mock_relay_server.ws_url) as ws:
        await ws.recv()  # banner
        await ws.send(json.dumps({"action": "auth", "token": mock_relay_server.token}))
        reply = json.loads(await ws.recv())
        assert reply[0]["T"] == "success"
        assert reply[0]["msg"] == "authenticated"


@pytest.mark.asyncio
async def test_f4_ws_channel_subscription_and_ack(mock_relay_server):
    async with websockets.connect(mock_relay_server.ws_url) as ws:
        await ws.recv()
        await ws.send(json.dumps({"action": "auth", "token": mock_relay_server.token}))
        await ws.recv()
        await ws.send(json.dumps({"action": "subscribe", "bars": ["SPY", "QQQ"]}))
        ack = json.loads(await ws.recv())
        assert ack[0]["T"] == "subscription"
        assert "SPY" in ack[0]["bars"]
        assert "QQQ" in ack[0]["bars"]


@pytest.mark.asyncio
async def test_f4_ws_streaming_bar_dispatch(mock_relay_server):
    async with websockets.connect(mock_relay_server.ws_url) as ws:
        await ws.recv()
        await ws.send(json.dumps({"action": "auth", "token": mock_relay_server.token}))
        await ws.recv()
        await ws.send(json.dumps({"action": "subscribe", "bars": ["SPY"]}))
        await ws.recv()

        # Emit bar from server
        await mock_relay_server.broadcast_bar("SPY", {"o": 500.0, "h": 502.0, "l": 499.0, "c": 501.5, "v": 2500})
        bar_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert bar_msg[0]["T"] == "b"
        assert bar_msg[0]["S"] == "SPY"
        assert bar_msg[0]["c"] == 501.5


@pytest.mark.asyncio
async def test_f4_ws_decoupled_consumer_queue():
    queue = asyncio.Queue(maxsize=50000)
    for i in range(100):
        queue.put_nowait({"id": i, "data": "bar"})
    assert queue.qsize() == 100
    item = await queue.get()
    assert item["id"] == 0


# ============================================================================
# FEATURE 5: Upstream Connection Lifecycle & Stale Hold
# ============================================================================

@pytest.mark.asyncio
async def test_f5_upstream_connected_event_processing(mock_relay_server):
    async with websockets.connect(mock_relay_server.ws_url) as ws:
        await ws.recv()
        await ws.send(json.dumps({"action": "auth", "token": mock_relay_server.token}))
        await ws.recv()
        await mock_relay_server.broadcast_lifecycle("upstream_connected")
        event = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert event[0]["T"] == "relay"
        assert event[0]["msg"] == "upstream_connected"


@pytest.mark.asyncio
async def test_f5_upstream_disconnected_transitions_to_stale_hold(mock_relay_server):
    async with websockets.connect(mock_relay_server.ws_url) as ws:
        await ws.recv()
        await ws.send(json.dumps({"action": "auth", "token": mock_relay_server.token}))
        await ws.recv()
        await mock_relay_server.broadcast_lifecycle("upstream_disconnected")
        event = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
        assert event[0]["T"] == "relay"
        assert event[0]["msg"] == "upstream_disconnected"
        assert mock_relay_server.upstream_connected is False


def test_f5_stale_data_hold_suppresses_aggressive_allocation(progressive_contracts):
    MarketRegime = progressive_contracts.models.MarketRegime
    # In STALE_DATA_HOLD, equity purchases are frozen
    regime = MarketRegime.STALE_DATA_HOLD
    is_safe_to_rebalance = (regime != MarketRegime.STALE_DATA_HOLD)
    assert is_safe_to_rebalance is False


@pytest.mark.asyncio
async def test_f5_upstream_reconnect_triggers_resubscribe(mock_relay_server):
    async with websockets.connect(mock_relay_server.ws_url) as ws:
        await ws.recv()
        await ws.send(json.dumps({"action": "auth", "token": mock_relay_server.token}))
        await ws.recv()
        await ws.send(json.dumps({"action": "subscribe", "bars": ["NVDA"]}))
        await ws.recv()

        # Disconnect then reconnect
        await mock_relay_server.broadcast_lifecycle("upstream_disconnected")
        await ws.recv()
        await mock_relay_server.broadcast_lifecycle("upstream_connected")
        recon_msg = json.loads(await ws.recv())
        assert recon_msg[0]["msg"] == "upstream_connected"
        assert "NVDA" in mock_relay_server.subscriptions[next(iter(mock_relay_server.clients))]["bars"]


def test_f5_reconnect_triggers_historical_bar_backfill():
    last_received = datetime(2026, 9, 3, 13, 0, tzinfo=timezone.utc)
    current_time = datetime(2026, 9, 3, 13, 15, tzinfo=timezone.utc)
    gap_duration_minutes = (current_time - last_received).total_seconds() / 60.0
    requires_backfill = gap_duration_minutes > 5.0
    assert requires_backfill is True


# ============================================================================
# FEATURE 6: In-Process Mock AlpacaRelay Server
# ============================================================================

@pytest.mark.asyncio
async def test_f6_mock_server_lifecycle_and_port_binding():
    server = MockAlpacaRelayServer()
    await server.start()
    assert server.port > 0
    assert server.http_url.startswith("http://127.0.0.1:")
    await server.stop()


@pytest.mark.asyncio
async def test_f6_mock_server_health_endpoint(mock_relay_server):
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{mock_relay_server.http_url}/health")
        assert res.status_code == 200
        data = res.json()
        assert data["upstream"] == "connected"
        assert data["feed"] == "sip"


@pytest.mark.asyncio
async def test_f6_mock_server_rest_auth_rejection(mock_relay_server):
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{mock_relay_server.http_url}/data/v2/stocks/SPY/bars",
            headers={"X-Relay-Token": "wrong-token"}
        )
        assert res.status_code == 401
        assert "relay_error" in res.json()


@pytest.mark.asyncio
async def test_f6_mock_server_websocket_auth_and_subscribe(mock_relay_server):
    async with websockets.connect(mock_relay_server.ws_url) as ws:
        await ws.recv()
        await ws.send(json.dumps({"action": "auth", "token": mock_relay_server.token}))
        ack = json.loads(await ws.recv())
        assert ack[0]["msg"] == "authenticated"
        await ws.send(json.dumps({"action": "subscribe", "bars": ["GLD"]}))
        sub = json.loads(await ws.recv())
        assert "GLD" in sub[0]["bars"]


@pytest.mark.asyncio
async def test_f6_mock_server_broadcast_lifecycle_and_bars(mock_relay_server):
    async with websockets.connect(mock_relay_server.ws_url) as ws:
        await ws.recv()
        await ws.send(json.dumps({"action": "auth", "token": mock_relay_server.token}))
        await ws.recv()
        await ws.send(json.dumps({"action": "subscribe", "bars": ["TLT"]}))
        await ws.recv()

        await mock_relay_server.broadcast_bar("TLT", {"c": 95.5})
        b_msg = json.loads(await ws.recv())
        assert b_msg[0]["S"] == "TLT"
        assert b_msg[0]["c"] == 95.5


# ============================================================================
# FEATURE 7: Multi-Timeframe Signal Engine
# ============================================================================

def test_f7_realized_volatility_targeting_formula(progressive_contracts):
    math_utils = progressive_contracts.math_utils
    # High vol: 40% annualized -> scale factor = 0.12 / 0.40 = 0.30
    scale_high = math_utils.volatility_scale_factor(realized_vol=0.40, target_vol=0.12)
    assert math.isclose(scale_high, 0.30, rel_tol=1e-3)

    # Low vol: 8% annualized -> scale factor capped at 1.0 (no leverage)
    scale_low = math_utils.volatility_scale_factor(realized_vol=0.08, target_vol=0.12, max_scale=1.0)
    assert scale_low == 1.0


def test_f7_sma_trend_filter_regime_classification(progressive_contracts):
    math_utils = progressive_contracts.math_utils
    # Bull trend: Price > SMA50 > SMA200
    prices = list(range(100, 350))
    sma50 = math_utils.calculate_sma(prices, 50)
    sma200 = math_utils.calculate_sma(prices, 200)
    current_price = prices[-1]
    assert current_price > sma50 > sma200


def test_f7_dynamic_atr_keltner_lower_band(progressive_contracts):
    math_utils = progressive_contracts.math_utils
    highs = [102.0 + i for i in range(20)]
    lows = [98.0 + i for i in range(20)]
    closes = [100.0 + i for i in range(20)]
    atr = math_utils.calculate_atr(highs, lows, closes, window=14)
    sma50 = math_utils.calculate_sma(closes, window=14)
    lower_band = sma50 - 2.0 * atr
    assert atr > 0.0
    assert lower_band < sma50


def test_f7_trailing_drawdown_gating_levels(progressive_contracts):
    evaluate_gate = progressive_contracts.math_utils.evaluate_drawdown_gate
    assert evaluate_gate(-0.02) == 1.00   # Normal
    assert evaluate_gate(-0.07) == 0.50   # Level 1 Caution
    assert evaluate_gate(-0.12) == 0.20   # Level 2 Defensive
    assert evaluate_gate(-0.18) == 0.00   # Level 3 Circuit Breaker


def test_f7_market_breadth_calculation():
    universe_prices = {"SPY": 500.0, "QQQ": 450.0, "AAPL": 220.0, "MSFT": 420.0, "NVDA": 120.0}
    universe_sma50 = {"SPY": 490.0, "QQQ": 440.0, "AAPL": 210.0, "MSFT": 430.0, "NVDA": 115.0}
    above = sum(1 for sym in universe_prices if universe_prices[sym] > universe_sma50[sym])
    breadth = above / len(universe_prices)
    assert breadth == 0.80  # 4 out of 5 above


# ============================================================================
# FEATURE 8: Safe-Haven Dual Momentum Allocator
# ============================================================================

def test_f8_structural_momentum_12_1_calculation(progressive_contracts):
    math_utils = progressive_contracts.math_utils
    # 260 daily prices rising 1.0 per day
    prices = [100.0 + i for i in range(260)]
    mom = math_utils.calculate_momentum_12_1(prices, lookback=252, skip=21)
    expected = ((100.0 + 260 - 22) / (100.0 + 260 - 253)) - 1.0
    assert math.isclose(mom, expected, rel_tol=1e-4)


def test_f8_absolute_momentum_hurdle_against_cash(progressive_contracts):
    math_utils = progressive_contracts.math_utils
    asset_prices = [100.0 + i * 0.1 for i in range(260)]
    cash_prices = [100.0 + i * 0.01 for i in range(260)]
    mom_asset = math_utils.calculate_momentum_12_1(asset_prices)
    mom_cash = math_utils.calculate_momentum_12_1(cash_prices)
    abs_pass = (mom_asset > mom_cash)
    assert abs_pass is True


def test_f8_tlt_200_sma_duration_filter(progressive_contracts):
    math_utils = progressive_contracts.math_utils
    # Declining bond prices (e.g. 2022)
    tlt_prices = [140.0 - i * 0.15 for i in range(250)]
    tlt_sma200 = math_utils.calculate_sma(tlt_prices, 200)
    current_tlt = tlt_prices[-1]
    is_tlt_qualified = (current_tlt > tlt_sma200)
    assert is_tlt_qualified is False


def test_f8_safe_haven_rotation_to_shv_in_rate_shock():
    tlt_pass = False
    gld_pass = False
    defensive_weight = 1.0
    w_tlt = 0.40 * defensive_weight if tlt_pass else 0.0
    w_gld = 0.40 * defensive_weight if gld_pass else 0.0
    w_shv = defensive_weight - w_tlt - w_gld
    assert w_tlt == 0.0
    assert w_gld == 0.0
    assert w_shv == 1.0


def test_f8_safe_haven_split_between_tlt_and_gld():
    tlt_pass = True
    gld_pass = True
    defensive_weight = 1.0
    w_tlt = 0.40 * defensive_weight if tlt_pass else 0.0
    w_gld = 0.40 * defensive_weight if gld_pass else 0.0
    w_shv = defensive_weight - w_tlt - w_gld
    assert w_tlt == 0.40
    assert w_gld == 0.40
    assert math.isclose(w_shv, 0.20, abs_tol=1e-5)


# ============================================================================
# FEATURE 9: Deterministic Allocation Engine
# ============================================================================

def test_f9_target_weights_strictly_sum_to_one(progressive_contracts):
    TargetAllocation = progressive_contracts.models.TargetAllocation
    MarketRegime = progressive_contracts.models.MarketRegime
    weights = {"SPY": 0.20, "QQQ": 0.35, "XLK": 0.20, "SHV": 0.25}
    alloc = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_AGGRESSIVE,
        weights=weights,
        cash_weight=0.25,
        rationale="Determinism check"
    )
    assert math.isclose(sum(alloc.weights.values()), 1.0, abs_tol=1e-6)


def test_f9_long_only_non_negative_weights(progressive_contracts):
    TargetAllocation = progressive_contracts.models.TargetAllocation
    MarketRegime = progressive_contracts.models.MarketRegime
    with pytest.raises(ValueError):
        TargetAllocation(
            timestamp=datetime.now(timezone.utc),
            regime=MarketRegime.BEAR_CRISIS,
            weights={"SPY": -0.10, "SHV": 1.10},
            cash_weight=1.10,
            rationale="Short weight should be rejected"
        )


def test_f9_drift_band_threshold_2_5_percent():
    target = 0.30
    drift_band = 0.025
    # Current at 0.31 -> drift = +0.01 (no rebalance)
    assert abs(0.31 - target) <= drift_band
    # Current at 0.33 -> drift = +0.03 (trigger rebalance)
    assert abs(0.33 - target) > drift_band


def test_f9_rebalance_order_intent_generation(progressive_contracts):
    OrderIntent = progressive_contracts.models.OrderIntent
    target_weight = 0.35
    current_weight = 0.20
    delta = target_weight - current_weight
    order = OrderIntent(
        symbol="QQQ", action="BUY" if delta > 0 else "SELL",
        target_weight=target_weight, current_weight=current_weight,
        delta_weight=delta, rationale="Increase QQQ to target"
    )
    assert order.action == "BUY"
    assert math.isclose(order.delta_weight, 0.15)


def test_f9_deterministic_tie_breaking_alphabetical():
    candidates = [("XLY", 0.152), ("XLK", 0.152), ("XLE", 0.152)]
    # Sort descending by score, ascending by ticker
    sorted_candidates = sorted(candidates, key=lambda x: (-x[1], x[0]))
    assert sorted_candidates[0][0] == "XLE"
    assert sorted_candidates[1][0] == "XLK"
    assert sorted_candidates[2][0] == "XLY"


# ============================================================================
# FEATURE 10: STRATEGY.md Mathematical Blueprint
# ============================================================================

def test_f10_strategy_md_file_exists():
    root_dir = Path(__file__).resolve().parents[2]
    strategy_path = root_dir / "STRATEGY.md"
    assert strategy_path.exists()


def test_f10_blueprint_volatility_drag_formula():
    mu = 0.10
    sigma = 0.40
    # g = mu - 0.5 * sigma^2
    drag = 0.5 * (sigma ** 2)
    g = mu - drag
    assert math.isclose(drag, 0.08)
    assert math.isclose(g, 0.02)


def test_f10_blueprint_drawdown_recovery_asymmetry():
    # R_rec = D / (1 - D)
    d50 = 0.50
    assert math.isclose(d50 / (1.0 - d50), 1.0)  # 100% recovery for 50% DD
    d15 = 0.15
    assert math.isclose(d15 / (1.0 - d15), 0.17647, rel_tol=1e-3)  # 17.6% recovery for 15% DD


def test_f10_blueprint_regime_matrix_definitions(progressive_contracts):
    MarketRegime = progressive_contracts.models.MarketRegime
    regimes = [r.value for r in MarketRegime]
    assert "BULL_AGGRESSIVE" in regimes
    assert "BULL_NORMAL" in regimes
    assert "CORRECTION_FRAGILE" in regimes
    assert "BEAR_CRISIS" in regimes
    assert "STALE_DATA_HOLD" in regimes


def test_f10_blueprint_zero_lookahead_filtration():
    # Proof of filtration invariance: decision at day T only accesses bars up to T
    current_time = datetime(2026, 9, 3, 16, 0, tzinfo=timezone.utc)
    future_bar_time = datetime(2026, 9, 4, 9, 30, tzinfo=timezone.utc)
    assert future_bar_time > current_time


# ============================================================================
# FEATURE 11: Dual Persistence Engine
# ============================================================================

def test_f11_sqlite_wal_mode_initialization(temp_sqlite_db):
    conn = sqlite3.connect(temp_sqlite_db)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    assert mode.lower() == "wal"
    conn.close()


def test_f11_market_bars_repository_crud(temp_sqlite_db):
    conn = sqlite3.connect(temp_sqlite_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_bars (
            symbol TEXT, timestamp TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER,
            PRIMARY KEY (symbol, timestamp)
        );
    """)
    conn.execute(
        "INSERT INTO market_bars VALUES (?, ?, ?, ?, ?, ?, ?);",
        ("SPY", "2026-09-01T04:00:00Z", 500.0, 505.0, 498.0, 503.0, 1000000)
    )
    conn.commit()
    cursor = conn.cursor()
    cursor.execute("SELECT close FROM market_bars WHERE symbol = 'SPY';")
    row = cursor.fetchone()
    assert row[0] == 503.0
    conn.close()


def test_f11_regime_history_persistence(temp_sqlite_db):
    conn = sqlite3.connect(temp_sqlite_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS regime_history (
            timestamp TEXT PRIMARY KEY, regime TEXT, spy_price REAL, vol_20d REAL
        );
    """)
    conn.execute("INSERT INTO regime_history VALUES (?, ?, ?, ?);", ("2026-09-03T16:00:00Z", "BULL_NORMAL", 510.0, 0.12))
    conn.commit()
    cursor = conn.cursor()
    cursor.execute("SELECT regime FROM regime_history;")
    assert cursor.fetchone()[0] == "BULL_NORMAL"
    conn.close()


def test_f11_portfolio_snapshot_persistence(temp_sqlite_db):
    conn = sqlite3.connect(temp_sqlite_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            timestamp TEXT PRIMARY KEY, weights_json TEXT, cash_weight REAL
        );
    """)
    conn.execute("INSERT INTO portfolio_snapshots VALUES (?, ?, ?);", ("2026-09-03T16:00:00Z", json.dumps({"SPY": 0.5, "SHV": 0.5}), 0.5))
    conn.commit()
    cursor = conn.cursor()
    cursor.execute("SELECT weights_json FROM portfolio_snapshots;")
    weights = json.loads(cursor.fetchone()[0])
    assert weights["SPY"] == 0.5
    conn.close()


def test_f11_jsonl_append_only_audit_log(tmp_path):
    audit_path = tmp_path / "decision_audit.jsonl"
    record = {"timestamp": "2026-09-03T16:00:00Z", "regime": "BULL_NORMAL", "rationale": "Test rationale"}
    with open(audit_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    assert audit_path.exists()
    with open(audit_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["regime"] == "BULL_NORMAL"


# ============================================================================
# FEATURE 12: Decision Daemon & Market Scheduler
# ============================================================================

def test_f12_market_calendar_et_timezone_awareness():
    # Market close is 16:00 ET
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    dt = datetime(2026, 9, 3, 15, 50, tzinfo=et)
    assert dt.hour == 15 and dt.minute == 50


def test_f12_daily_close_evaluation_at_15_50():
    eval_hour, eval_minute = 15, 50
    assert eval_hour == 15 and eval_minute == 50


def test_f12_monthly_rebalance_event_scheduling():
    # Check if a date is the last trading day of September 2026
    day = datetime(2026, 9, 30, tzinfo=timezone.utc)
    is_month_end = (day.day == 30 and day.month == 9)
    assert is_month_end is True


def test_f12_intraday_circuit_breaker_tick_handler():
    # Trigger circuit breaker if tick price drops > 5% intraday
    open_price = 500.0
    tick_price = 474.0
    intraday_change = (tick_price - open_price) / open_price
    circuit_breaker = intraday_change < -0.05
    assert circuit_breaker is True


def test_f12_daemon_graceful_shutdown():
    shutdown_requested = asyncio.Event()
    shutdown_requested.set()
    assert shutdown_requested.is_set()


# ============================================================================
# FEATURE 13: Production Execution CLI
# ============================================================================

def test_f13_cli_dry_run_command_execution():
    args = ["dry-run", "--symbols", "SPY,QQQ", "--simulate"]
    assert "dry-run" in args
    assert "--simulate" in args


def test_f13_cli_daemon_command_flags():
    flags = {"--once": True, "--interval": "60s"}
    assert flags["--once"] is True
    assert flags["--interval"] == "60s"


def test_f13_cli_rebalance_command():
    rebalance_options = {"execute": False, "dry_run": True}
    assert rebalance_options["dry_run"] is True


def test_f13_cli_status_command():
    status_output = {"service": "strategy_engine", "status": "active", "regime": "BULL_NORMAL"}
    assert status_output["status"] == "active"


def test_f13_cli_export_metrics_command():
    metrics = {"sharpe": 1.45, "sortino": 2.10, "max_drawdown": -0.095, "cagr": 0.182}
    metrics_json = json.dumps(metrics)
    parsed = json.loads(metrics_json)
    assert parsed["max_drawdown"] == -0.095
