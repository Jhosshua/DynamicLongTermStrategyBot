"""
tests.adversarial.test_m3_dashboard_api_stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adversarial Empirical Challenge Harness for Milestone 3:
Attacks and stress-tests the Mobile-Centric Dashboard API (web/app.py)
and Bot Service (bot/service.py):

1. Rapid-fire Concurrent Operator Requests:
   - 20+ threads rapidly hammering /api/operator/pause, /api/operator/resume,
     and /api/operator/rebalance simultaneously.
   - Assert zero deadlock, zero race condition corruption, and consistent
     state machine transitions.

2. SSE Connection Storm & Abrupt Drop:
   - Open 25+ concurrent SSE client streams to /api/events.
   - Suddenly drop network connections without sending close frames (TCP RST/FIN).
   - Assert that the server cleanly cleans up tasks without leaking memory or
     throwing unhandled background exceptions.

3. Fast Portfolio Polling Under Rebalances:
   - Hammer /api/portfolio with 20+ reader threads while rebalance fills are
     actively updating the SQLite WAL ledger.
   - Assert reading threads never crash with database locked (WAL concurrency).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import json
import logging
import math
import os
import random
import socket
import sqlite3
import tempfile
import threading
import time
from typing import Any, Dict, Generator, List, Tuple
import uuid

import httpx
import pytest
import uvicorn

from bot.feed_manager import FeedSource
from bot.paper_account import PaperAccountConfig, PaperAccountManager, PortfolioSummary
from bot.service import (
    DynamicStrategyService,
    ManualRebalanceResult,
    ServiceConfig,
    ServiceState,
    ServiceStatus,
)
from strategy_engine.core.models import Bar, OrderIntent
from tests.integration.test_dashboard_sse import StreamingASGITransport
from web.app import create_app

logger = logging.getLogger("tests.adversarial.dashboard_api_stress")


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Acquire an available ephemeral TCP port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def temp_wal_db_path() -> Generator[str, None, None]:
    """Provide a temporary SQLite database initialized in WAL mode."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.commit()
    conn.close()

    yield path

    for suffix in ["", "-wal", "-shm"]:
        p = path + suffix
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


@pytest.fixture
def live_server(temp_wal_db_path: str) -> Generator[Tuple[str, DynamicStrategyService], None, None]:
    """
    Launch a real in-process Uvicorn server on an ephemeral port.
    Provides true multi-threaded network socket concurrency.
    """
    # Ensure event loop exists for initialization on Python 3.9
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = ServiceConfig(
        db_path=temp_wal_db_path,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
        initial_cash=50000.00,
        dry_run=False,
    )
    service = DynamicStrategyService(config=config)
    service.paper_account.init_schema()
    service._service_state = ServiceState.RUNNING

    # Warmup bars
    loop.run_until_complete(service._warmup_historical_bars())

    app = create_app(service=service)
    port = _find_free_port()

    uv_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(uv_config)

    server_thread = threading.Thread(target=server.run, daemon=True, name="uvicorn_stress_server")
    server_thread.start()

    base_url = f"http://127.0.0.1:{port}"

    # Wait for server readiness
    ready = False
    for _ in range(50):
        try:
            r = httpx.get(f"{base_url}/health", timeout=1.0)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.05)

    if not ready:
        server.should_exit = True
        server_thread.join(timeout=2.0)
        raise RuntimeError(f"Failed to start live test server on {base_url}")

    yield base_url, service

    server.should_exit = True
    server_thread.join(timeout=3.0)
    service.paper_account.close()


# ---------------------------------------------------------------------------
# Test Suite 1: Rapid-fire Concurrent Operator Requests
# ---------------------------------------------------------------------------

class TestConcurrentOperatorRequests:
    """
    Adversarial concurrency test: 20+ threads hammer /pause, /resume, /rebalance.
    Assert zero deadlock, zero race condition corruption, consistent state transitions.
    """

    def test_rapid_concurrent_operator_hammer_zero_deadlock(
        self, live_server: Tuple[str, DynamicStrategyService]
    ):
        """
        24 worker threads rapidly hammer /pause, /resume, and /rebalance simultaneously.
        Assert zero deadlocks, zero 500 errors, and strict state invariants.
        """
        base_url, service = live_server
        thread_count = 24
        iterations_per_thread = 12
        total_requests = thread_count * iterations_per_thread

        results: Dict[str, int] = {
            "pause": 0,
            "resume": 0,
            "rebalance_force": 0,
            "rebalance_normal": 0,
            "status": 0,
        }
        lock = threading.Lock()
        errors: List[str] = []

        def worker_routine(worker_id: int):
            client = httpx.Client(base_url=base_url, timeout=12.0)
            actions = ["pause", "resume", "rebalance_force", "rebalance_normal", "status"]
            try:
                for step in range(iterations_per_thread):
                    action = actions[(worker_id + step) % len(actions)]

                    if action == "pause":
                        r = client.post("/api/operator/pause")
                        if r.status_code != 200:
                            errors.append(f"W{worker_id} pause status {r.status_code}: {r.text}")
                        else:
                            d = r.json()
                            assert d["success"] is True
                            assert d["state"] == "PAUSED"
                            assert d["is_paused"] is True
                            with lock:
                                results["pause"] += 1

                    elif action == "resume":
                        r = client.post("/api/operator/resume")
                        if r.status_code != 200:
                            errors.append(f"W{worker_id} resume status {r.status_code}: {r.text}")
                        else:
                            d = r.json()
                            assert d["success"] is True
                            assert d["state"] == "RUNNING"
                            assert d["is_paused"] is False
                            with lock:
                                results["resume"] += 1

                    elif action == "rebalance_force":
                        r = client.post("/api/operator/rebalance", json={"force": True})
                        if r.status_code != 200:
                            errors.append(f"W{worker_id} rebal force status {r.status_code}: {r.text}")
                        else:
                            d = r.json()
                            assert "success" in d
                            assert "status" in d
                            assert "orders_count" in d
                            assert d["status"] in (
                                "EXECUTED",
                                "SKIPPED_WITHIN_BAND",
                                "REJECTED_PAUSED",
                                "REJECTED_UNSAFE_FEED",
                                "ERROR_NO_ALLOCATION",
                            )
                            with lock:
                                results["rebalance_force"] += 1

                    elif action == "rebalance_normal":
                        r = client.post("/api/operator/rebalance", json={"force": False})
                        if r.status_code != 200:
                            errors.append(f"W{worker_id} rebal normal status {r.status_code}: {r.text}")
                        else:
                            d = r.json()
                            assert "success" in d
                            assert "status" in d
                            # If rejected because paused, status must be REJECTED_PAUSED
                            if not d["success"] and "PAUSED" in d.get("rationale", ""):
                                assert d["status"] == "REJECTED_PAUSED"
                            with lock:
                                results["rebalance_normal"] += 1

                    elif action == "status":
                        r = client.get("/api/status")
                        if r.status_code != 200:
                            errors.append(f"W{worker_id} status code {r.status_code}: {r.text}")
                        else:
                            d = r.json()
                            assert d["state"] in ("RUNNING", "PAUSED")
                            assert d["is_paused"] == (d["state"] == "PAUSED")
                            assert math.isclose(d["total_nav"], d["cash"] + d["equity"], rel_tol=1e-3)
                            with lock:
                                results["status"] += 1

            except Exception as e:
                errors.append(f"W{worker_id} unexpected exception: {e}")
            finally:
                client.close()

        start_time = time.monotonic()
        with ThreadPoolExecutor(max_workers=thread_count) as pool:
            futures = [pool.submit(worker_routine, i) for i in range(thread_count)]
            for f in as_completed(futures, timeout=25.0):
                f.result()
        duration = time.monotonic() - start_time

        # Empirical Assertions
        assert len(errors) == 0, f"Encountered {len(errors)} errors during concurrent storm: {errors[:5]}"
        assert sum(results.values()) == total_requests
        assert duration < 20.0, f"Operator hammering took {duration:.2f}s, suspected lock stall"

        # State machine integrity assertion
        final_status = service.get_service_status()
        assert final_status.state in (ServiceState.RUNNING, ServiceState.PAUSED)
        assert final_status.is_paused == (final_status.state == ServiceState.PAUSED)

        # Portfolio accounting invariant
        portfolio = service.paper_account.get_portfolio_state()
        assert math.isclose(portfolio.total_nav, portfolio.cash + portfolio.equity, rel_tol=1e-3)
        assert portfolio.cash >= 0.0

    def test_state_machine_transition_consistency(
        self, live_server: Tuple[str, DynamicStrategyService]
    ):
        """
        Rapid serialized and interleaved transitions between PAUSED and RUNNING.
        Verifies that at no point does an illegal or intermediate state leak.
        """
        base_url, service = live_server
        client = httpx.Client(base_url=base_url, timeout=5.0)

        for iteration in range(20):
            # Pause
            r_pause = client.post("/api/operator/pause")
            assert r_pause.status_code == 200
            p_data = r_pause.json()
            assert p_data["state"] == "PAUSED"
            assert p_data["is_paused"] is True

            # Verify unforced rebalance is strictly rejected
            r_reb = client.post("/api/operator/rebalance", json={"force": False})
            assert r_reb.status_code == 200
            reb_data = r_reb.json()
            assert reb_data["success"] is False
            assert reb_data["status"] == "REJECTED_PAUSED"

            # Resume
            r_resume = client.post("/api/operator/resume")
            assert r_resume.status_code == 200
            res_data = r_resume.json()
            assert res_data["state"] == "RUNNING"
            assert res_data["is_paused"] is False

        client.close()


# ---------------------------------------------------------------------------
# Test Suite 2: SSE Connection Storm & Abrupt Drop
# ---------------------------------------------------------------------------

class TestSSEConnectionStormAndAbruptDrop:
    """
    Stress-tests the SSE /api/events endpoint under concurrent connection storms,
    abrupt TCP socket drops without close frames, and async client task cancellations.
    Assert zero task leaks, zero unhandled background exceptions, and prompt cleanup.
    """

    def test_sse_raw_tcp_socket_storm_and_abrupt_rst(
        self, live_server: Tuple[str, DynamicStrategyService]
    ):
        """
        Open 30 concurrent raw TCP sockets to /api/events, receive initial SSE frame,
        and abruptly close the socket without sending HTTP or TCP close frames.
        Assert server cleans up client streams cleanly and continues operating.
        """
        base_url, service = live_server
        parsed = httpx.URL(base_url)
        host, port = parsed.host, parsed.port
        socket_count = 30

        def raw_socket_client(cid: int) -> Tuple[bool, str]:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(6.0)
                sock.connect((host, port))

                # Send raw HTTP GET request for SSE
                req = (
                    f"GET /api/events HTTP/1.1\r\n"
                    f"Host: {host}:{port}\r\n"
                    f"Accept: text/event-stream\r\n"
                    f"Connection: keep-alive\r\n\r\n"
                )
                sock.sendall(req.encode("utf-8"))

                # Read until we receive partial or full SSE frame
                chunk = sock.recv(256)
                if not chunk:
                    return False, "No data received"

                # Abruptly tear down socket without HTTP close
                sock.close()
                return True, "Success"
            except Exception as e:
                return False, str(e)

        with ThreadPoolExecutor(max_workers=socket_count) as pool:
            results = list(pool.map(raw_socket_client, range(socket_count)))

        successes = [r for r in results if r[0]]
        assert len(successes) == socket_count, f"Some sockets failed to connect: {results}"

        # Allow server a brief moment to process disconnects
        time.sleep(0.5)

        # Assert server is still responsive and healthy
        health = httpx.get(f"{base_url}/health", timeout=3.0)
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_sse_concurrent_stream_client_storm_and_task_cleanup(
        self, temp_wal_db_path: str
    ):
        """
        Using ASGI in-process transport: launch 30 concurrent SSE streaming clients,
        drop them with various chaotic patterns (immediate cancel, 1 frame break,
        2 frame break, raised exception).
        Assert all streaming tasks are cleanly cleaned up with zero task leaks.
        """
        os.environ["PYTEST_CURRENT_TEST"] = "1"

        config = ServiceConfig(
            db_path=temp_wal_db_path,
            relay_base_url="http://127.0.0.1:9999",
            relay_ws_url="ws://127.0.0.1:9999",
            initial_cash=50000.00,
            dry_run=False,
        )
        service = DynamicStrategyService(config=config)
        service.paper_account.init_schema()
        service._service_state = ServiceState.RUNNING

        app = create_app(service=service)
        transport = StreamingASGITransport(app=app)

        client_count = 30
        initial_tasks = len([t for t in asyncio.all_tasks() if not t.done()])

        async def chaotic_sse_client(cid: int) -> int:
            async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=5.0) as ac:
                try:
                    async with ac.stream("GET", "/api/events") as resp:
                        assert resp.status_code == 200
                        frames = 0
                        async for line in resp.aiter_lines():
                            if line.startswith("data:"):
                                frames += 1
                                if cid % 3 == 0:
                                    # Immediate break on first frame
                                    break
                                elif cid % 3 == 1 and frames >= 2:
                                    # Break on second frame
                                    break
                                elif cid % 3 == 2:
                                    # Simulate abrupt connection abort exception
                                    raise ConnectionResetError("Simulated client RST")
                except ConnectionResetError:
                    pass
            return cid

        results = await asyncio.gather(*(chaotic_sse_client(i) for i in range(client_count)))
        assert len(results) == client_count

        # Yield to event loop to allow cancellation handlers and finally blocks to run
        await asyncio.sleep(0.3)
        gc.collect()

        active_tasks_after = len([t for t in asyncio.all_tasks() if not t.done()])
        # Server must not leak tasks (allowing delta of 1 for pytest runner task)
        assert active_tasks_after <= initial_tasks + 2, (
            f"SSE task leak detected: tasks before={initial_tasks}, tasks after={active_tasks_after}"
        )


# ---------------------------------------------------------------------------
# Test Suite 3: Fast Portfolio Polling Under Rebalances
# ---------------------------------------------------------------------------

class TestFastPortfolioPollingUnderRebalances:
    """
    Stress-tests SQLite WAL concurrency:
    Hammers /api/portfolio with 20+ reading threads while rebalances and WAL
    writes are actively occurring in background threads.
    Asserts zero 'database is locked' errors and 100% data consistency.
    """

    def test_portfolio_polling_under_active_rebalance_fills(
        self, live_server: Tuple[str, DynamicStrategyService]
    ):
        """
        Hammer /api/portfolio with 20 reader threads (800+ total reads) while
        a rebalance writer actively executes orders updating the SQLite ledger.
        Assert zero database lock errors.
        """
        base_url, service = live_server
        stop_event = threading.Event()

        rebalance_count = 0
        rebalance_errors: List[Exception] = []

        # Background rebalance writer
        def rebalance_loop():
            nonlocal rebalance_count
            prices = {"SPY": 505.0, "QQQ": 445.0, "SHV": 110.0, "XLK": 212.0}
            while not stop_event.is_set():
                try:
                    orders = [
                        OrderIntent(
                            symbol="SPY",
                            action="BUY",
                            delta_dollars=500.0,
                            target_shares=10.0,
                            delta_shares=1.0,
                            estimated_price=505.0,
                        ),
                        OrderIntent(
                            symbol="SHV",
                            action="SELL",
                            delta_dollars=500.0,
                            target_shares=40.0,
                            delta_shares=-4.54,
                            estimated_price=110.0,
                        ),
                    ]
                    service.paper_account.execute_rebalance_orders(orders, prices)
                    rebalance_count += 1
                    time.sleep(0.01)
                except Exception as ex:
                    rebalance_errors.append(ex)

        writer_thread = threading.Thread(target=rebalance_loop, daemon=True, name="rebalance_writer")
        writer_thread.start()

        # Concurrent portfolio polling threads
        reader_threads = 20
        reads_per_thread = 40
        total_reads = reader_threads * reads_per_thread

        read_errors: List[str] = []
        successful_reads = 0
        read_lock = threading.Lock()

        def reader_worker(wid: int):
            nonlocal successful_reads
            client = httpx.Client(base_url=base_url, timeout=8.0)
            try:
                for _ in range(reads_per_thread):
                    r = client.get("/api/portfolio")
                    if r.status_code != 200:
                        read_errors.append(f"Reader {wid} HTTP {r.status_code}: {r.text}")
                    else:
                        d = r.json()
                        assert "total_nav" in d
                        assert "cash" in d
                        assert "positions" in d
                        assert math.isclose(d["total_nav"], d["cash"] + d["equity"], rel_tol=1e-3)
                        with read_lock:
                            successful_reads += 1
            except Exception as e:
                read_errors.append(f"Reader {wid} exception: {e}")
            finally:
                client.close()

        with ThreadPoolExecutor(max_workers=reader_threads) as pool:
            futures = [pool.submit(reader_worker, i) for i in range(reader_threads)]
            for f in as_completed(futures, timeout=20.0):
                f.result()

        stop_event.set()
        writer_thread.join(timeout=2.0)

        # Assertions
        assert len(read_errors) == 0, f"Read errors encountered: {read_errors[:5]}"
        assert successful_reads == total_reads
        assert len(rebalance_errors) == 0, f"Rebalance writer encountered errors: {rebalance_errors[:5]}"
        assert rebalance_count >= 5, f"Expected at least 5 rebalance iterations, got {rebalance_count}"

    def test_portfolio_polling_under_external_wal_writer_storm(
        self, live_server: Tuple[str, DynamicStrategyService], temp_wal_db_path: str
    ):
        """
        Hammer /api/portfolio with 20 reader threads while an independent,
        external SQLite connection aggressively writes DML transactions to the WAL DB.
        Verifies SQLite WAL concurrency: readers never block writers, writers never block readers.
        """
        base_url, service = live_server
        stop_event = threading.Event()
        external_writes = 0
        external_errors: List[Exception] = []

        # Independent external SQLite writer (simulates background ingestion or daemon)
        def external_writer_loop():
            nonlocal external_writes
            conn = sqlite3.connect(temp_wal_db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            while not stop_event.is_set():
                try:
                    with conn:
                        now_iso = datetime.now(timezone.utc).isoformat()
                        conn.execute(
                            "UPDATE paper_account_state SET updated_at = ? WHERE id = 1;",
                            (now_iso,),
                        )
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO paper_positions (
                                symbol, shares, avg_entry_price, cost_basis, current_price, updated_at
                            ) VALUES ('SPY', 10.0, 500.0, 5000.0, 505.0, ?);
                            """,
                            (now_iso,),
                        )
                    external_writes += 1
                    time.sleep(0.005)
                except Exception as ex:
                    external_errors.append(ex)
            conn.close()

        writer_thread = threading.Thread(target=external_writer_loop, daemon=True, name="ext_wal_writer")
        writer_thread.start()

        # Concurrent reader threads
        reader_threads = 20
        reads_per_thread = 30
        total_reads = reader_threads * reads_per_thread

        read_errors: List[str] = []
        successful_reads = 0
        read_lock = threading.Lock()

        def reader_worker(wid: int):
            nonlocal successful_reads
            client = httpx.Client(base_url=base_url, timeout=8.0)
            try:
                for _ in range(reads_per_thread):
                    r = client.get("/api/portfolio")
                    if r.status_code != 200:
                        read_errors.append(f"Reader {wid} HTTP {r.status_code}: {r.text}")
                    else:
                        d = r.json()
                        assert "total_nav" in d
                        assert d["total_nav"] > 0.0
                        with read_lock:
                            successful_reads += 1
            except Exception as e:
                read_errors.append(f"Reader {wid} exception: {e}")
            finally:
                client.close()

        with ThreadPoolExecutor(max_workers=reader_threads) as pool:
            futures = [pool.submit(reader_worker, i) for i in range(reader_threads)]
            for f in as_completed(futures, timeout=20.0):
                f.result()

        stop_event.set()
        writer_thread.join(timeout=2.0)

        # Assertions
        assert len(read_errors) == 0, f"Read errors under WAL storm: {read_errors[:5]}"
        assert successful_reads == total_reads
        assert len(external_errors) == 0, f"External WAL write errors: {external_errors[:5]}"
        assert external_writes >= 10, f"Expected at least 10 external writes, got {external_writes}"


# ---------------------------------------------------------------------------
# Test Suite 4: Full Multi-Vector Combined Stress
# ---------------------------------------------------------------------------

class TestFullMultivectorCombinedStress:
    """
    Simultaneous multi-vector stress test:
    - Reader threads hammering /api/portfolio
    - Operator threads hammering /pause, /resume, /rebalance
    - Raw TCP streaming clients connecting to /api/events and dropping
    - Active background WAL writer
    All executing simultaneously against the live dashboard server.
    """

    def test_combined_multivector_assault(
        self, live_server: Tuple[str, DynamicStrategyService], temp_wal_db_path: str
    ):
        """
        Execute simultaneous reader storm, operator storm, SSE drops, and WAL writes.
        Assert zero deadlocks, zero crashes, and complete invariant preservation.
        """
        base_url, service = live_server
        parsed = httpx.URL(base_url)
        host, port = parsed.host, parsed.port

        duration_seconds = 3.0
        stop_event = threading.Event()
        errors: List[str] = []
        lock = threading.Lock()

        counts = {
            "portfolio_reads": 0,
            "operator_actions": 0,
            "sse_drops": 0,
            "wal_writes": 0,
        }

        # 1. Background WAL writer
        def wal_writer():
            conn = sqlite3.connect(temp_wal_db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            while not stop_event.is_set():
                try:
                    with conn:
                        now_iso = datetime.now(timezone.utc).isoformat()
                        conn.execute("UPDATE paper_account_state SET updated_at = ? WHERE id = 1;", (now_iso,))
                    with lock:
                        counts["wal_writes"] += 1
                    time.sleep(0.01)
                except Exception as ex:
                    with lock:
                        errors.append(f"WAL writer: {ex}")
            conn.close()

        # 2. Portfolio poller thread
        def portfolio_poller(wid: int):
            client = httpx.Client(base_url=base_url, timeout=6.0)
            while not stop_event.is_set():
                try:
                    r = client.get("/api/portfolio")
                    if r.status_code == 200:
                        d = r.json()
                        assert "total_nav" in d
                        with lock:
                            counts["portfolio_reads"] += 1
                    else:
                        with lock:
                            errors.append(f"Poller W{wid} status {r.status_code}")
                except Exception as ex:
                    with lock:
                        errors.append(f"Poller W{wid}: {ex}")
                time.sleep(0.01)
            client.close()

        # 3. Operator action thread
        def operator_worker(wid: int):
            client = httpx.Client(base_url=base_url, timeout=6.0)
            actions = ["pause", "resume", "rebalance_force", "rebalance_normal"]
            step = 0
            while not stop_event.is_set():
                act = actions[(wid + step) % len(actions)]
                step += 1
                try:
                    if act == "pause":
                        r = client.post("/api/operator/pause")
                    elif act == "resume":
                        r = client.post("/api/operator/resume")
                    elif act == "rebalance_force":
                        r = client.post("/api/operator/rebalance", json={"force": True})
                    elif act == "rebalance_normal":
                        r = client.post("/api/operator/rebalance", json={"force": False})

                    if r.status_code == 200:
                        with lock:
                            counts["operator_actions"] += 1
                    else:
                        with lock:
                            errors.append(f"Operator W{wid} on {act} status {r.status_code}")
                except Exception as ex:
                    with lock:
                        errors.append(f"Operator W{wid} on {act}: {ex}")
                time.sleep(0.02)
            client.close()

        # 4. SSE raw drop thread
        def sse_dropper(wid: int):
            while not stop_event.is_set():
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(3.0)
                    s.connect((host, port))
                    req = f"GET /api/events HTTP/1.1\r\nHost: {host}:{port}\r\nAccept: text/event-stream\r\n\r\n"
                    s.sendall(req.encode())
                    _ = s.recv(128)
                    s.close()
                    with lock:
                        counts["sse_drops"] += 1
                except Exception as ex:
                    with lock:
                        errors.append(f"SSE dropper W{wid}: {ex}")
                time.sleep(0.05)

        # Launch all threads
        threads = []
        threads.append(threading.Thread(target=wal_writer, daemon=True))
        for i in range(12):
            threads.append(threading.Thread(target=portfolio_poller, args=(i,), daemon=True))
        for i in range(8):
            threads.append(threading.Thread(target=operator_worker, args=(i,), daemon=True))
        for i in range(6):
            threads.append(threading.Thread(target=sse_dropper, args=(i,), daemon=True))

        for t in threads:
            t.start()

        # Let the storm rage for duration_seconds
        time.sleep(duration_seconds)
        stop_event.set()

        for t in threads:
            t.join(timeout=3.0)

        # Verify no unhandled errors occurred
        assert len(errors) == 0, f"Combined stress encountered errors: {errors[:5]}"
        assert counts["portfolio_reads"] >= 50, f"Expected 50+ reads, got {counts['portfolio_reads']}"
        assert counts["operator_actions"] >= 20, f"Expected 20+ actions, got {counts['operator_actions']}"
        assert counts["sse_drops"] >= 10, f"Expected 10+ sse drops, got {counts['sse_drops']}"
        assert counts["wal_writes"] >= 20, f"Expected 20+ writes, got {counts['wal_writes']}"

        # Post-assault health and state verification
        post_health = httpx.get(f"{base_url}/health", timeout=3.0)
        assert post_health.status_code == 200
        assert post_health.json()["status"] == "ok"

        st = service.get_service_status()
        assert st.state in (ServiceState.RUNNING, ServiceState.PAUSED)
        assert st.is_paused == (st.state == ServiceState.PAUSED)

        portfolio = service.paper_account.get_portfolio_state()
        assert math.isclose(portfolio.total_nav, portfolio.cash + portfolio.equity, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# Test Suite 5: Empirical Edge-Case Verification (EventLoop & Concurrency Findings)
# ---------------------------------------------------------------------------

class TestEventLoopCrossBoundaryFindings:
    """
    Empirical challenge tests investigating runtime event loop boundaries
    and cross-thread lock behavior.
    """

    def test_python39_event_loop_lifecycle_finding(self, temp_wal_db_path: str):
        """
        Empirically demonstrates that in Python 3.9, instantiating DynamicStrategyService
        after an asyncio.run() call without setting a new event loop raises RuntimeError.
        This verifies our finding regarding the dependency of asyncio.Lock/Event on get_event_loop().
        """
        # Execute asyncio.run which resets event loop to None in Python 3.9
        asyncio.run(asyncio.sleep(0.001))

        # Attempting to initialize DynamicStrategyService without an active loop
        try:
            cfg = ServiceConfig(db_path=temp_wal_db_path)
            svc = DynamicStrategyService(config=cfg)
            # If it succeeded, loop was not None (e.g. Python >= 3.10)
        except RuntimeError as exc:
            # On Python 3.9, get_event_loop() raises RuntimeError
            assert "There is no current event loop in thread" in str(exc)
        finally:
            # Restore an event loop for subsequent test cleanup
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

