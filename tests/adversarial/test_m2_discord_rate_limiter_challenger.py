"""Empirical Challenger & Stress Harness for Milestone 2 Discord Alerts Rate Limiter & 429 Engine.

Fulfills Authoritative Request:
- ORIGINAL_REQUEST.md (R3)
- PROJECT.md (Feature 9)

Attacks:
1. Concurrency burst storm: 20+ threads rapidly firing alerts simultaneously.
   Asserts that inter-post intervals are paced >= 2.0s without data loss or race condition corruption.
2. HTTP 429 flood injection: Inject simulated 429 responses with various retry_after values (0.2s, 1.5s, 600s).
   Verifies exponential backoff, jitter, retry cap (max 3 retries), and that sleep >= 5.0s is aborted immediately.
3. Multi-instance rate limiting collision test: Checks whether independent DiscordNotifier instances
   synchronize process-wide or collide due to instance-level locks.
4. 429 retry pacing under concurrency: Tests whether retrying threads re-synchronize with the rate limiter
   or fire out-of-cadence (< 2.0s delta) relative to concurrent posts.
5. Large rate limit (e.g. 1200s) millisecond truncation check.
6. Async event loop starvation probe: Verifies whether async_post_* blocks the event loop.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import logging
import math
import os
import random
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

import bot.discord_alerts as da
from bot.discord_alerts import (
    DEFAULT_MAX_BACKOFF_SLEEP_S,
    DEFAULT_RATE_LIMIT_INTERVAL_S,
    DISCORD_COLOR_BROKEN,
    DISCORD_COLOR_RECOVERED,
    DISCORD_COLOR_TRADE,
    DiscordEmbedCard,
    DiscordNotifier,
    _extract_retry_after,
    _post,
)


@pytest.fixture(autouse=True)
def reset_rate_limiter_state():
    """Reset global rate limiter timestamp before each test."""
    with da._process_lock:
        da._last_post_monotonic = 0.0
    yield
    with da._process_lock:
        da._last_post_monotonic = 0.0


# ============================================================================
# 1. Concurrency Burst Storm (20+ Threads)
# ============================================================================

def test_concurrency_burst_storm_25_threads_no_data_loss_thread_safety():
    """Attack 1A: Concurrency write storm with 25 simultaneous threads.
    
    Verifies thread safety, 0 dropped alerts, 0 data corruptions, and strictly
    ordered audit entries under scaled rate limiting.
    """
    post_records: List[Dict[str, Any]] = []
    record_lock = threading.Lock()

    def mock_poster(url: str, json: dict):
        now = time.monotonic()
        with record_lock:
            post_records.append({
                "time": now,
                "title": json["embeds"][0]["title"],
                "thread": threading.get_ident(),
            })
        resp = MagicMock()
        resp.status_code = 204
        return resp

    # Scaled rate limit interval of 0.04s so 25 threads complete in ~1.0s
    scaled_interval = 0.04
    notifier = DiscordNotifier(
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=mock_poster,
        rate_limit_interval_s=scaled_interval,
        max_backoff_sleep_s=1.0,
        suppress_in_test=False,
    )

    thread_count = 25
    barrier = threading.Barrier(thread_count)
    results = [None] * thread_count

    def worker(idx: int):
        barrier.wait()  # Synchronize threads to fire simultaneously
        ok = notifier.post_broken_alert(
            component=f"WorkerThread_{idx:02d}",
            error_message=f"Simulated fault {idx}",
            evidence=f"Stack trace for thread {idx}",
            dashboard_url="https://bot.railway.app",
        )
        results[idx] = ok

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    # Assertions
    assert all(r is True for r in results), "All 25 threads must report successful post"
    assert len(notifier.dispatched_cards) == thread_count, (
        f"Expected {thread_count} audit cards, found {len(notifier.dispatched_cards)}"
    )
    assert len(post_records) == thread_count, (
        f"Expected {thread_count} HTTP posts, recorded {len(post_records)}"
    )

    # Check pacing between consecutive posts
    post_records.sort(key=lambda r: r["time"])
    deltas = [
        post_records[i]["time"] - post_records[i - 1]["time"]
        for i in range(1, len(post_records))
    ]
    
    # Margin allows 5ms jitter for OS context switching
    min_required = scaled_interval - 0.005
    violations = [d for d in deltas if d < min_required]
    assert len(violations) == 0, (
        f"Found {len(violations)} pacing violations (< {min_required:.4f}s): {violations}"
    )


def test_concurrency_burst_storm_real_2s_pacing_sample():
    """Attack 1B: Real-time 2.0s pacing verification across concurrent threads.
    
    Fires 4 concurrent threads with DEFAULT_RATE_LIMIT_INTERVAL_S = 2.0s.
    Verifies that real inter-post intervals are strictly >= 2.0s.
    Duration ~6.0s.
    """
    call_timestamps: List[float] = []
    ts_lock = threading.Lock()

    def mock_poster(url: str, json: dict):
        ts = time.monotonic()
        with ts_lock:
            call_timestamps.append(ts)
        resp = MagicMock()
        resp.status_code = 204
        return resp

    notifier = DiscordNotifier(
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=mock_poster,
        rate_limit_interval_s=2.0,
        max_backoff_sleep_s=5.0,
        suppress_in_test=False,
    )

    thread_count = 4
    barrier = threading.Barrier(thread_count)

    def worker(idx: int):
        barrier.wait()
        notifier.post_broken_alert(f"Subsystem_{idx}", f"Crash {idx}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert len(call_timestamps) == 4, f"Expected 4 calls, got {len(call_timestamps)}"
    call_timestamps.sort()

    deltas = [call_timestamps[i] - call_timestamps[i - 1] for i in range(1, len(call_timestamps))]
    for i, d in enumerate(deltas):
        # Allow 0.001s floating point / OS resolution tolerance
        assert d >= 1.999, f"Interval between post {i} and {i+1} was {d:.4f}s (< 2.0s)"


def test_concurrency_simulated_clock_20_threads_burst():
    """Attack 1C: 20-thread simulated clock burst storm verifying >= 2.0s pacing.
    
    Simulates 20 concurrent threads attempting to fire at the exact same moment.
    Hooks time.sleep to advance a monotonic virtual clock, asserting that every
    single inter-post interval is >= 2.0s across all 20 threads.
    """
    simulated_now = 1000.0
    clock_lock = threading.Lock()
    post_timestamps: List[float] = []

    def mock_sleep(seconds: float):
        nonlocal simulated_now
        with clock_lock:
            simulated_now += seconds

    def mock_monotonic() -> float:
        with clock_lock:
            return simulated_now

    def mock_poster(url: str, json: dict):
        with clock_lock:
            post_timestamps.append(simulated_now)
        resp = MagicMock()
        resp.status_code = 204
        return resp

    with patch("time.sleep", side_effect=mock_sleep), patch("time.monotonic", side_effect=mock_monotonic):
        notifier = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/mock",
            http_post=mock_poster,
            rate_limit_interval_s=2.0,
            max_backoff_sleep_s=5.0,
            suppress_in_test=False,
        )

        # Fire 20 sequential calls through the rate limiter
        for i in range(20):
            notifier.post_broken_alert(f"Comp_{i}", f"Error {i}")

    assert len(post_timestamps) == 20
    deltas = [post_timestamps[i] - post_timestamps[i - 1] for i in range(1, len(post_timestamps))]
    for i, d in enumerate(deltas):
        assert d >= 2.0, f"Virtual interval between post {i} and {i+1} was {d}s (< 2.0s)"


# ============================================================================
# 2. HTTP 429 Flood Injection & Backoff Math
# ============================================================================

def test_http_429_retry_after_0_2s_backoff():
    """Attack 2A: Inject 429 with small retry_after=0.2s.
    
    Verifies exponential backoff math:
    backoff_delay = max(0.2, 2.0) * (1.5 ** 0) + jitter
    Because delay (0.2s) < rate_limit_interval_s (2.0s), backoff must clamp to at least
    rate_limit_interval_s (2.0s) + jitter [0.1, 0.5].
    """
    sleep_calls: List[float] = []
    attempts = 0

    def mock_sleep(seconds: float):
        sleep_calls.append(seconds)

    def mock_poster(url: str, json: dict):
        nonlocal attempts
        attempts += 1
        resp = MagicMock()
        if attempts == 1:
            resp.status_code = 429
            resp.json.return_value = {"retry_after": 0.2}
            resp.headers = {}
        else:
            resp.status_code = 204
        return resp

    with patch("time.sleep", side_effect=mock_sleep):
        ok = _post(
            payload={"embeds": [{"title": "Test 0.2s"}]},
            webhook_url="https://mock",
            http_post=mock_poster,
            rate_limit_interval_s=2.0,
            max_backoff_sleep_s=5.0,
            suppress_in_test=False,
        )

    assert ok is True
    assert attempts == 2
    # Sleep calls: first for rate limiter (if any), second for 429 backoff
    # 429 backoff must be max(0.2, 2.0) * 1.0 + [0.1, 0.5] => in [2.1, 2.5]
    backoff_sleep = sleep_calls[-1]
    assert 2.1 <= backoff_sleep <= 2.5, (
        f"Expected backoff sleep in [2.1, 2.5], got {backoff_sleep:.4f}"
    )


def test_http_429_retry_after_1_5s_backoff():
    """Attack 2B: Inject 429 with retry_after=1.5s.
    
    Verifies backoff delay calculation with rate_limit_interval_s=2.0s:
    max(1.5, 2.0) * 1.0 + jitter [0.1, 0.5] => [2.1, 2.5]
    """
    sleep_calls: List[float] = []
    attempts = 0

    def mock_sleep(seconds: float):
        sleep_calls.append(seconds)

    def mock_poster(url: str, json: dict):
        nonlocal attempts
        attempts += 1
        resp = MagicMock()
        if attempts == 1:
            resp.status_code = 429
            resp.json.return_value = {"retry_after": 1.5}
            resp.headers = {}
        else:
            resp.status_code = 204
        return resp

    with patch("time.sleep", side_effect=mock_sleep):
        ok = _post(
            payload={"embeds": [{"title": "Test 1.5s"}]},
            webhook_url="https://mock",
            http_post=mock_poster,
            rate_limit_interval_s=2.0,
            max_backoff_sleep_s=5.0,
            suppress_in_test=False,
        )

    assert ok is True
    assert attempts == 2
    backoff_sleep = sleep_calls[-1]
    assert 2.1 <= backoff_sleep <= 2.5, (
        f"Expected backoff sleep in [2.1, 2.5], got {backoff_sleep:.4f}"
    )


def test_http_429_retry_after_600s_aborts_immediately():
    """Attack 2C: Inject simulated 429 with retry_after=600s (Discord IP rate limit).
    
    Verifies that sleep >= 5.0s aborts immediately on attempt 1 without sleeping
    or blocking daemon execution.
    """
    sleep_calls: List[float] = []
    attempts = 0

    def mock_sleep(seconds: float):
        sleep_calls.append(seconds)

    def mock_poster(url: str, json: dict):
        nonlocal attempts
        attempts += 1
        resp = MagicMock()
        resp.status_code = 429
        resp.json.return_value = {"retry_after": 600.0}
        resp.headers = {}
        return resp

    with patch("time.sleep", side_effect=mock_sleep):
        start = time.monotonic()
        ok = _post(
            payload={"embeds": [{"title": "Test 600s"}]},
            webhook_url="https://mock",
            http_post=mock_poster,
            rate_limit_interval_s=2.0,
            max_backoff_sleep_s=5.0,
            suppress_in_test=False,
        )
        elapsed = time.monotonic() - start

    assert ok is False
    assert attempts == 1, "Must abort immediately after attempt 1"
    assert len(sleep_calls) == 0, f"Must not sleep when retry_after=600s, but called sleep({sleep_calls})"
    assert elapsed < 0.2, f"Execution was blocked for {elapsed:.4f}s"


def test_http_429_retry_cap_exact_3_retries():
    """Attack 2D: Persistent 429 flood with retry_after=0.1s.
    
    Verifies:
    1. Exactly 3 retries (total 4 attempts: 1 initial + 3 retries).
    2. Exponential backoff progression: attempt 0 (*1.0), attempt 1 (*1.5), attempt 2 (*2.25).
    3. Final return value is False upon retry exhaustion.
    """
    sleep_calls: List[float] = []
    attempts = 0

    def mock_sleep(seconds: float):
        sleep_calls.append(seconds)

    def mock_poster(url: str, json: dict):
        nonlocal attempts
        attempts += 1
        resp = MagicMock()
        resp.status_code = 429
        resp.json.return_value = {"retry_after": 0.5}
        resp.headers = {}
        return resp

    with patch("time.sleep", side_effect=mock_sleep):
        ok = _post(
            payload={"embeds": [{"title": "Test Exhaustion"}]},
            webhook_url="https://mock",
            http_post=mock_poster,
            rate_limit_interval_s=1.0,
            max_backoff_sleep_s=5.0,
            max_retries=3,
            suppress_in_test=False,
        )

    assert ok is False
    assert attempts == 4, f"Expected 4 total attempts (1 initial + 3 retries), got {attempts}"
    assert len(sleep_calls) == 3, f"Expected 3 backoff sleeps, got {len(sleep_calls)}"

    # Check that backoff sleeps increase monotonically
    # sleep 0: max(0.5, 1.0) * 1.0 + jitter = 1.0 + jitter [0.1, 0.5] => [1.1, 1.5]
    # sleep 1: max(0.5, 1.0) * 1.5 + jitter = 1.5 + jitter [0.1, 0.5] => [1.6, 2.0]
    # sleep 2: max(0.5, 1.0) * 2.25 + jitter = 2.25 + jitter [0.1, 0.5] => [2.35, 2.75]
    assert 1.1 <= sleep_calls[0] <= 1.5, f"Sleep 0 invalid: {sleep_calls[0]}"
    assert 1.6 <= sleep_calls[1] <= 2.0, f"Sleep 1 invalid: {sleep_calls[1]}"
    assert 2.35 <= sleep_calls[2] <= 2.75, f"Sleep 2 invalid: {sleep_calls[2]}"


# ============================================================================
# 3. Empirical Bug & Vulnerability Reproduction
# ============================================================================

def test_vulnerability_429_exact_5s_boundary_flaw():
    """Attack 3A: Off-by-one check on retry_after == 5.0s causes 15 seconds blocking sleep.
    
    Requirement: "sleep >= 5.0s is aborted immediately without blocking execution."
    Observed implementation: `if delay > max_backoff_sleep_s:`
    Empirically demonstrates that when delay == 5.0s, the engine does NOT abort immediately.
    Instead, it retries 3 times, sleeping 5.0s each time (total 15s blocking sleep).
    """
    sleep_calls: List[float] = []
    attempts = 0

    def mock_sleep(seconds: float):
        sleep_calls.append(seconds)

    def mock_poster(url: str, json: dict):
        nonlocal attempts
        attempts += 1
        resp = MagicMock()
        resp.status_code = 429
        resp.json.return_value = {"retry_after": 5.0}
        resp.headers = {}
        return resp

    with patch("time.sleep", side_effect=mock_sleep):
        ok = _post(
            payload={"embeds": [{"title": "Boundary 5.0s"}]},
            webhook_url="https://mock",
            http_post=mock_poster,
            rate_limit_interval_s=2.0,
            max_backoff_sleep_s=5.0,
            max_retries=3,
            suppress_in_test=False,
        )

    # Remediation assertion: The requirement states >= 5.0s must abort immediately without sleep.
    assert ok is False, "Must return False when delay >= max_backoff_sleep_s"
    assert attempts == 1, f"Remediation verified: attempts={attempts} (aborted immediately at attempt 1)"
    assert len(sleep_calls) == 0, f"Remediation verified: 0 sleeps called ({sleep_calls})"


def test_vulnerability_multi_instance_notifier_lock_isolation_collision():
    """Attack 3B: Multi-instance DiscordNotifier lock isolation race condition.
    
    Each DiscordNotifier uses the unified `_process_lock`.
    When two instances run concurrently, they share the global `_process_lock`
    and `_last_post_monotonic`, ensuring mutual exclusion and >= 1.0s pacing.
    """
    post_times: List[float] = []
    lock = threading.Lock()

    def mock_poster(url, json):
        with lock:
            post_times.append(time.monotonic())
        return MagicMock(status_code=204)

    notifier_a = DiscordNotifier(http_post=mock_poster, rate_limit_interval_s=1.0, suppress_in_test=False)
    notifier_b = DiscordNotifier(http_post=mock_poster, rate_limit_interval_s=1.0, suppress_in_test=False)

    # Pre-seed _last_post_monotonic to 0.5s ago so both threads determine they must wait 0.5s
    da._last_post_monotonic = time.monotonic() - 0.5

    barrier = threading.Barrier(2)

    def worker_a():
        barrier.wait()
        notifier_a.post_broken_alert("A", "err")

    def worker_b():
        barrier.wait()
        notifier_b.post_broken_alert("B", "err")

    ta = threading.Thread(target=worker_a)
    tb = threading.Thread(target=worker_b)
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    assert len(post_times) == 2
    delta = abs(post_times[1] - post_times[0])
    # Remediation assertion: Mutex ensures pacing >= 0.45s (allowing thread switch tolerance)
    assert delta >= 0.45, f"Remediation verified: instances synchronized with delta={delta:.6f}s"


def test_vulnerability_429_retry_inter_post_pacing_violation_under_concurrency():
    """Attack 3C: 429 Retrying thread bypasses rate-limiter lock, violating inter-post pacing.
    
    In _post(), the rate-limiter lock is acquired on every attempt.
    When a thread retries after 429, it re-acquires the lock and checks `_last_post_monotonic`,
    ensuring retries honor the >= 1.0s spacing relative to concurrent threads.
    """
    post_times: List[Tuple[float, str]] = []
    lock = threading.Lock()

    def mock_poster(url, json):
        now = time.monotonic()
        title = json["embeds"][0]["title"]
        with lock:
            post_times.append((now, title))
        resp = MagicMock(status_code=204, headers={})
        # Thread 0 gets 429 on its first attempt
        if "Thread_0" in title and len([p for p in post_times if "Thread_0" in p[1]]) == 1:
            resp.status_code = 429
            resp.json = lambda: {"retry_after": 0.1}
        return resp

    notifier = DiscordNotifier(http_post=mock_poster, rate_limit_interval_s=1.0, suppress_in_test=False)

    barrier = threading.Barrier(3)

    def worker(idx: int):
        barrier.wait()
        notifier.post_broken_alert(f"Thread_{idx}", "err")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    post_times.sort(key=lambda x: x[0])
    deltas = [post_times[i][0] - post_times[i - 1][0] for i in range(1, len(post_times))]
    min_delta = min(deltas)
    # Remediation assertion: The minimum delta between posts is >= 0.85s (satisfying 1.0s limit with tolerance)
    assert min_delta >= 0.85, (
        f"Remediation verified: 429 retry respected rate limit pacing (min_delta={min_delta:.4f}s >= 0.85s)"
    )


def test_vulnerability_extract_retry_after_legacy_millisecond_overflow_hazard():
    """Attack 3D: Heuristic treats Discord retry_after >= 1000s as milliseconds.
    
    In Discord API v10, retry_after is in seconds.
    If Discord returns 1200.0 (a 20-minute temporary ban), _extract_retry_after preserves 1200.0s.
    """
    resp = MagicMock()
    resp.json.return_value = {"retry_after": 1200.0}
    resp.headers = {}
    parsed = _extract_retry_after(resp)
    # Remediation assertion: 1200.0 seconds is preserved without truncation to 1.2s
    assert parsed == 1200.0, f"Remediation verified: 1200s preserved as {parsed}s"


@pytest.mark.asyncio
async def test_vulnerability_async_post_blocks_event_loop():
    """Attack 3E: async_post_* methods block asyncio event loop with synchronous sleep.
    
    Because async_post_broken_alert delegates to asyncio.to_thread,
    it does not block the active event loop thread, allowing background tasks to proceed.
    """
    notifier = DiscordNotifier(
        http_post=lambda url, json: MagicMock(status_code=204),
        rate_limit_interval_s=0.5,
        suppress_in_test=False,
    )

    ticks = [0]

    async def heartbeat():
        while True:
            await asyncio.sleep(0.02)
            ticks[0] += 1

    hb_task = asyncio.create_task(heartbeat())
    t0 = time.monotonic()
    await notifier.async_post_broken_alert("A", "err")
    await notifier.async_post_broken_alert("B", "err")
    t1 = time.monotonic()
    hb_task.cancel()
    try:
        await hb_task
    except asyncio.CancelledError:
        pass

    elapsed = t1 - t0
    # Because async_post is non-blocking, the 20ms heartbeat ticks during rate limit delays
    assert ticks[0] > 0, (
        f"Remediation verified: event loop remained responsive during {elapsed:.2f}s, ticks={ticks[0]}"
    )
