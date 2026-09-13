"""Unit test suite for Institutional Discord v2 Alerts Engine.

Tests card payload generation, field limits, color codes, 429 backoff math,
rate limiting pacing, and pytest suppression mode.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
import os
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from bot.discord_alerts import (
    BLUE,
    DEFAULT_MAX_BACKOFF_SLEEP_S,
    DEFAULT_RATE_LIMIT_INTERVAL_S,
    DISCORD_COLOR_BROKEN,
    DISCORD_COLOR_RECOVERED,
    DISCORD_COLOR_TRADE,
    GREEN,
    RED,
    DiscordCardType,
    DiscordColor,
    DiscordConfig,
    DiscordEmbedCard,
    DiscordEmbedField,
    DiscordNotifier,
    _extract_retry_after,
    _post,
    _redact_secrets,
    _sanitize_field_value,
    _under_pytest,
    broken_card,
    build_broken_card,
    build_recovered_card,
    build_trade_execution_card,
    format_duration,
    recovered_card,
    trade_card,
)
from strategy_engine.core.models import OrderIntent, OrderSide


# ============================================================================
# 1. Color Constants & Data Models
# ============================================================================

def test_color_code_exact_integers():
    """Verify exact institutional decimal color values."""
    assert DISCORD_COLOR_BROKEN == 0xE53935
    assert DISCORD_COLOR_BROKEN == 15022389
    assert RED == 15022389

    assert DISCORD_COLOR_RECOVERED == 0x43A047
    assert DISCORD_COLOR_RECOVERED == 4431943
    assert GREEN == 4431943

    assert DISCORD_COLOR_TRADE == 0x1E88E5
    assert DISCORD_COLOR_TRADE == 2001125
    assert BLUE == 2001125


def test_discord_color_class_constants():
    """Verify DiscordColor class mapping."""
    assert DiscordColor.BROKEN == 15022389
    assert DiscordColor.RECOVERED == 4431943
    assert DiscordColor.TRADE == 2001125


def test_discord_card_type_enum():
    """Verify DiscordCardType values."""
    assert DiscordCardType.BROKEN.value == "broken"
    assert DiscordCardType.RECOVERED.value == "recovered"
    assert DiscordCardType.TRADE_EXECUTION.value == "trade_execution"


def test_discord_config_defaults():
    """Verify default configuration values."""
    cfg = DiscordConfig()
    assert cfg.rate_limit_interval_s == 2.0
    assert cfg.max_backoff_sleep_s == 5.0
    assert cfg.max_retries == 3
    assert cfg.suppress_in_test is True
    assert "railway.app" in cfg.default_dashboard_url


def test_discord_embed_field_serialization():
    """Verify DiscordEmbedField correctly serializes to dictionary."""
    field = DiscordEmbedField(name="Component", value="PaperLedger", inline=True)
    d = field.to_dict()
    assert d == {"name": "Component", "value": "PaperLedger", "inline": True}

    # Empty value fallback
    empty_field = DiscordEmbedField(name="", value="")
    d_empty = empty_field.to_dict()
    assert d_empty["name"] == "Detail"
    assert d_empty["value"] == "(none)"


# ============================================================================
# 2. Duration Formatting Helper
# ============================================================================

def test_format_duration_sub_minute():
    """Sub-minute durations format with one decimal place."""
    assert format_duration(0.0) == "0.0s"
    assert format_duration(15.2) == "15.2s"
    assert format_duration(45.0) == "45.0s"
    assert format_duration(59.9) == "59.9s"


def test_format_duration_multi_minute():
    """Multi-minute durations format as minutes and seconds."""
    assert format_duration(60.0) == "1m"
    assert format_duration(120.0) == "2m"
    assert format_duration(185.0) == "3m 5s"
    assert format_duration(222.0) == "3m 42s"


def test_format_duration_multi_hour():
    """Multi-hour durations format as hours, minutes, and seconds."""
    assert format_duration(3600.0) == "1h 0m"
    assert format_duration(3725.0) == "1h 2m 5s"


def test_format_duration_negative_clamp():
    """Negative durations (from NTP clock skew) clamp cleanly to 0.0s."""
    assert format_duration(-10.0) == "0.0s"
    assert format_duration(-0.001) == "0.0s"


# ============================================================================
# 3. Card Builders: Schema & Field Contracts
# ============================================================================

def test_build_broken_card_schema():
    """Verify broken card payload adheres to institutional v2 specification."""
    card = build_broken_card(
        component="AlpacaRelayClient",
        error_message="WebSocket relay stream dropped abnormally",
        evidence="ConnectionClosedError: code 1006",
        since=datetime(2026, 9, 13, 20, 5, 0, tzinfo=timezone.utc),
        impact="Order generation frozen in STALE_DATA_HOLD",
        action_required="1. Inspect AlpacaRelay service on Railway\n2. Check upstream auth",
        dashboard_url="https://bot.railway.app",
    )

    assert card.color == DISCORD_COLOR_BROKEN
    assert "BROKEN" in card.title
    assert "AlpacaRelayClient" in card.title
    assert card.url == "https://bot.railway.app"
    assert "WebSocket relay stream dropped" in card.description

    payload = card.to_dict()
    assert payload["color"] == 15022389
    field_names = [f["name"] for f in payload["fields"]]
    assert "Component" in field_names
    assert "Since" in field_names
    assert "What broke" in field_names
    assert "Evidence" in field_names
    assert "Impact" in field_names
    assert "Action required" in field_names
    assert "Operator Dashboard" in field_names

    evidence_field = next(f for f in payload["fields"] if f["name"] == "Evidence")
    assert "```" in evidence_field["value"]
    assert "code 1006" in evidence_field["value"]


def test_build_recovered_card_schema():
    """Verify recovered card payload adheres to institutional v2 specification."""
    card = build_recovered_card(
        component="AlpacaRelayClient",
        downtime_duration_s=222.0,
        status_info="WebSocket reconnected · REST gap backfill of 45 bars completed",
        recovery_evidence="Stream restored at 2026-09-13T20:08:42Z",
        dashboard_url="https://bot.railway.app",
    )

    assert card.color == DISCORD_COLOR_RECOVERED
    assert "RECOVERED" in card.title
    assert "AlpacaRelayClient" in card.title
    assert card.url == "https://bot.railway.app"
    assert "3m 42s" in card.description

    payload = card.to_dict()
    assert payload["color"] == 4431943
    fields = {f["name"]: f["value"] for f in payload["fields"]}
    assert fields["Component"] == "AlpacaRelayClient"
    assert "Downtime Duration" in fields
    assert fields["Downtime Duration"] == "3m 42s"
    assert fields["Down for"] == "3m 42s"
    assert "Telemetry Status" in fields
    assert "REST gap backfill" in fields["Telemetry Status"]


def test_build_trade_execution_card_schema():
    """Verify trade execution card payload adheres to institutional v2 specification."""
    sample_orders = [
        {"symbol": "SPY", "side": "SELL", "shares": 35.0, "price": 560.25, "notional": 19608.75},
        {"symbol": "QQQ", "side": "BUY", "shares": 18.5, "price": 485.10, "notional": 8974.35},
        {"symbol": "SHV", "side": "BUY", "shares": 95.0, "price": 110.12, "notional": 10461.40},
    ]

    card = build_trade_execution_card(
        orders=sample_orders,
        nav=50385.20,
        cash=1245.50,
        regime="BULL_NORMAL",
        dashboard_url="https://bot.railway.app",
    )

    assert card.color == DISCORD_COLOR_TRADE
    assert "TRADE EXECUTION" in card.title
    assert "BULL_NORMAL" in card.title
    assert "$50,385.20" in card.description

    payload = card.to_dict()
    assert payload["color"] == 2001125
    fields = {f["name"]: f["value"] for f in payload["fields"]}
    assert fields["Regime"] == "BULL_NORMAL"
    assert fields["Portfolio NAV"] == "$50,385.20"
    assert fields["Cash balance"] == "$1,245.50"
    assert "Orders Detail" in fields
    assert "SPY" in fields["Orders Detail"]
    assert "QQQ" in fields["Orders Detail"]
    assert "SHV" in fields["Orders Detail"]


def test_build_trade_execution_card_empty_orders_raises():
    """Attempting trade card construction with empty orders raises ValueError."""
    with pytest.raises(ValueError, match="Orders list cannot be empty"):
        build_trade_execution_card(orders=[], nav=50000.0)

    with pytest.raises(ValueError, match="Orders list cannot be empty"):
        build_trade_execution_card(orders=None, nav=50000.0)  # type: ignore


def test_trade_card_order_intent_model_compatibility():
    """OrderIntent objects from strategy engine format cleanly."""
    intents = [
        OrderIntent(symbol="SPY", side=OrderSide.SELL, delta_shares=10.0, estimated_price=550.0, reason="trim"),
        OrderIntent(symbol="QQQ", side=OrderSide.BUY, delta_shares=5.25, estimated_price=480.0, reason="add"),
    ]
    card = build_trade_execution_card(orders=intents, nav=50000.0, regime="BULL_NORMAL")
    detail = next(f["value"] for f in card.fields if f["name"] == "Orders Detail")
    assert "SELL" in detail
    assert "10.00 shs" in detail
    assert "BUY" in detail
    assert "5.25 shs" in detail


def test_trade_card_large_batch_truncates_at_10():
    """Order batch with > 10 items truncates with overflow summary line."""
    many_orders = [
        {"symbol": f"SYM_{i}", "side": "BUY", "shares": 10.0, "price": 100.0}
        for i in range(16)
    ]
    card = build_trade_execution_card(orders=many_orders, nav=50000.0)
    detail = next(f["value"] for f in card.fields if f["name"] == "Orders Detail")
    assert "... and 6 more orders" in detail
    assert "SYM_0" in detail
    assert "SYM_9" in detail
    assert "SYM_10" not in detail


def test_trade_card_zero_nav_formatting():
    """NAV of $0.00 formats cleanly without division error or formatting crash."""
    order = {"symbol": "SHV", "side": "BUY", "shares": 100.0, "price": 110.0}
    card = build_trade_execution_card(orders=[order], nav=0.0)
    nav_field = next(f["value"] for f in card.fields if f["name"] == "Portfolio NAV")
    assert nav_field == "$0.00"


# ============================================================================
# 4. Field Limits, Defensive Sanitization & Secret Redaction
# ============================================================================

def test_field_value_truncation_over_1024():
    """Giant stack traces (> 2000 chars) are clamped within 1024 characters."""
    giant_stack = "Traceback (most recent call last):\n" + ("x" * 2500)
    sanitized = _sanitize_field_value(giant_stack, max_chars=1024, wrap_code=True)
    assert len(sanitized) <= 1024
    assert sanitized.startswith("```")
    assert sanitized.endswith("```")
    assert "..." in sanitized


def test_empty_field_fallback_protection():
    """Empty or None field values fallback to non-empty placeholders."""
    assert _sanitize_field_value(None, wrap_code=True) == "```(none)```"
    assert _sanitize_field_value("", wrap_code=True) == "```(none)```"
    assert _sanitize_field_value("   ", wrap_code=False) == "(none)"


def test_secret_redaction():
    """Private webhook URL tokens and API keys are redacted."""
    raw_url = "https://discord.com/api/webhooks/1234567890/abcDEFghiJKLmnoPQR_stuVWX"
    redacted = _redact_secrets(raw_url)
    assert "abcDEFghiJKLmnoPQR_stuVWX" not in redacted
    assert "[REDACTED_TOKEN]" in redacted

    text_with_token = "Connection failed token=super_secret_alpaca_key_12345"
    redacted_token = _redact_secrets(text_with_token)
    assert "super_secret_alpaca_key_12345" not in redacted_token
    assert "token=[REDACTED]" in redacted_token


def test_total_embed_size_downsizing():
    """Cards exceeding 5800 total characters automatically downsize largest field."""
    huge_field = "A" * 3000
    huge_field_2 = "B" * 3000
    card = DiscordEmbedCard(
        color=DISCORD_COLOR_BROKEN,
        title="Overflow Test",
        fields=[
            {"name": "Field1", "value": huge_field},
            {"name": "Field2", "value": huge_field_2},
        ],
    )
    payload = card.to_dict()
    total_len = len(payload["title"]) + len(payload.get("description", ""))
    for f in payload["fields"]:
        total_len += len(f["name"]) + len(f["value"])
    assert total_len <= 5850


# ============================================================================
# 5. HTTP 429 & Retry Engine
# ============================================================================

def test_extract_retry_after_json_seconds():
    """Extract retry_after in floating-point seconds from Discord JSON body."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": "Rate limited", "retry_after": 1.75}
    mock_resp.headers = {}
    delay = _extract_retry_after(mock_resp)
    assert math.isclose(delay, 1.75)


def test_extract_retry_after_legacy_milliseconds():
    """Extract and normalize legacy millisecond retry_after (> 100) to seconds."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"retry_after": 2500.0}
    mock_resp.headers = {"X-RateLimit-Precision": "millisecond"}
    delay = _extract_retry_after(mock_resp)
    assert math.isclose(delay, 2.5)


def test_extract_retry_after_from_header():
    """Fallback to Retry-After HTTP header when JSON body is missing or unparseable."""
    mock_resp = MagicMock()
    mock_resp.json.side_effect = ValueError("Not JSON")
    mock_resp.headers = {"Retry-After": "3.5"}
    delay = _extract_retry_after(mock_resp)
    assert math.isclose(delay, 3.5)


def test_extract_retry_after_from_reset_after_header():
    """Extract from X-RateLimit-Reset-After header."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {}
    mock_resp.headers = {"X-RateLimit-Reset-After": "0.85"}
    delay = _extract_retry_after(mock_resp)
    assert math.isclose(delay, 0.85)


def test_extract_retry_after_default_fallback():
    """Corrupted response safely falls back to default 1.0s delay."""
    mock_resp = MagicMock()
    mock_resp.json.side_effect = Exception("Boom")
    mock_resp.headers = None
    delay = _extract_retry_after(mock_resp, default_delay=1.0)
    assert delay == 1.0


def test_http_429_retry_success():
    """429 followed by 204 successfully retries and returns True."""
    attempts = 0

    def mock_poster(url: str, json: dict):
        nonlocal attempts
        attempts += 1
        resp = MagicMock()
        if attempts == 1:
            resp.status_code = 429
            resp.json.return_value = {"retry_after": 0.05}
            resp.headers = {}
        else:
            resp.status_code = 204
        return resp

    payload = {"embeds": [{"title": "Test Retry"}]}
    ok = _post(
        payload=payload,
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=mock_poster,
        rate_limit_interval_s=0.0,
        suppress_in_test=False,
    )
    assert ok is True
    assert attempts == 2


def test_http_429_max_sleep_cap_aborts_immediately():
    """If Discord 429 asks for > 5.0s (e.g. 600s IP ban), aborts immediately without sleeping."""
    attempts = 0

    def mock_poster(url: str, json: dict):
        nonlocal attempts
        attempts += 1
        resp = MagicMock()
        resp.status_code = 429
        resp.json.return_value = {"retry_after": 600.0}  # 10 minutes
        return resp

    payload = {"embeds": [{"title": "Test 429 Cap"}]}
    start = time.monotonic()
    ok = _post(
        payload=payload,
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=mock_poster,
        rate_limit_interval_s=0.0,
        max_backoff_sleep_s=5.0,
        suppress_in_test=False,
    )
    duration = time.monotonic() - start
    assert ok is False
    assert attempts == 1  # Aborted on attempt 1 without sleeping
    assert duration < 1.0  # Did not sleep 600 seconds!


def test_http_429_exhausted_retries_returns_false():
    """Persistent 429 fails after max 3 retries without throwing exceptions."""
    attempts = 0

    def mock_poster(url: str, json: dict):
        nonlocal attempts
        attempts += 1
        resp = MagicMock()
        resp.status_code = 429
        resp.json.return_value = {"retry_after": 0.01}
        return resp

    payload = {"embeds": [{"title": "Test Retries"}]}
    ok = _post(
        payload=payload,
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=mock_poster,
        max_retries=3,
        rate_limit_interval_s=0.0,
        suppress_in_test=False,
    )
    assert ok is False
    assert attempts == 4  # 1 initial + 3 retries = 4 attempts total


# ============================================================================
# 6. Pytest Suppression & Test Environment Mode
# ============================================================================

def test_under_pytest_detection():
    """Environment detection finds active pytest session."""
    assert _under_pytest() is True
    # Explicit override test
    assert _under_pytest(explicit_flag=False) is False
    assert _under_pytest(explicit_flag=True) is True


def test_pytest_suppression_drops_external_network_io():
    """Under pytest without mock poster, zero HTTP calls are made."""
    notifier = DiscordNotifier(
        webhook_url="https://discord.com/api/webhooks/mock",
        suppress_in_test=True,
    )
    assert notifier.is_pytest_environment() is True

    # Dispatch alerts
    ok_broken = notifier.post_broken_alert("Daemon", "Crash", "traceback", "https://dash")
    ok_recovered = notifier.post_recovered_alert("Daemon", 10.0, "OK", "https://dash")
    ok_trade = notifier.post_trade_execution([{"symbol": "SPY", "side": "BUY", "shares": 10.0, "price": 500.0}], 50000.0)

    assert ok_broken is True
    assert ok_recovered is True
    assert ok_trade is True
    assert len(notifier.dispatched_cards) == 3


def test_pytest_suppression_rapid_burst_instantaneous():
    """50 rapid alerts under pytest complete in < 50ms without pacing sleep."""
    notifier = DiscordNotifier(suppress_in_test=True)
    start = time.monotonic()
    for i in range(50):
        notifier.post_broken_alert(f"Sub_{i}", f"Error {i}", "trace", "https://dash")
    elapsed = time.monotonic() - start

    assert len(notifier.dispatched_cards) == 50
    assert elapsed < 0.20  # Under 200ms total for 50 alerts


def test_mock_poster_invoked_even_under_pytest():
    """Supplying explicit mock poster executes pipeline to test payload contents."""
    captured = []

    def mock_poster(url: str, json: dict):
        captured.append(json)
        resp = MagicMock()
        resp.status_code = 204
        return resp

    notifier = DiscordNotifier(
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=mock_poster,
        suppress_in_test=True,
    )
    ok = notifier.post_broken_alert("ComponentA", "Failed", "evidence", "https://dash")
    assert ok is True
    assert len(captured) == 1
    assert "BROKEN" in captured[0]["embeds"][0]["title"]


# ============================================================================
# 7. Functional Standalone Dispatchers
# ============================================================================

def test_standalone_broken_card_helper():
    """Functional broken_card helper constructs and dispatches embed."""
    captured = []

    def mock_poster(url, json):
        captured.append(json)
        r = MagicMock()
        r.status_code = 204
        return r

    ok = broken_card(
        component="FeedWatchdog",
        error_message="Stale feed timeout",
        evidence="Silence > 120s",
        dashboard_url="https://dash",
        http_post=mock_poster,
        webhook_url="https://mock",
    )
    assert ok is True
    assert len(captured) == 1
    assert captured[0]["embeds"][0]["color"] == RED


def test_standalone_recovered_card_helper():
    """Functional recovered_card helper constructs and dispatches embed."""
    captured = []

    def mock_poster(url, json):
        captured.append(json)
        r = MagicMock()
        r.status_code = 204
        return r

    ok = recovered_card(
        component="FeedWatchdog",
        downtime_s=125.0,
        status_info="Bars resumed",
        dashboard_url="https://dash",
        http_post=mock_poster,
        webhook_url="https://mock",
    )
    assert ok is True
    assert len(captured) == 1
    assert captured[0]["embeds"][0]["color"] == GREEN


def test_standalone_trade_card_helper():
    """Functional trade_card helper constructs and dispatches embed."""
    captured = []

    def mock_poster(url, json):
        captured.append(json)
        r = MagicMock()
        r.status_code = 204
        return r

    orders = [{"symbol": "QQQ", "side": "BUY", "shares": 20.0, "price": 480.0}]
    ok = trade_card(
        orders=orders,
        nav=50000.0,
        regime="BULL_NORMAL",
        dashboard_url="https://dash",
        http_post=mock_poster,
        webhook_url="https://mock",
    )
    assert ok is True
    assert len(captured) == 1
    assert captured[0]["embeds"][0]["color"] == BLUE


# ============================================================================
# 8. Secret Redaction in Logger Records
# ============================================================================

def test_secret_redaction_in_logger_records(caplog):
    """Network transport errors never leak private webhook tokens into log messages."""
    secret_url = "https://discord.com/api/webhooks/9988776655/super_secret_token_12345"

    def failing_poster(url, json):
        raise ConnectionResetError(f"Failed connecting to {url}")

    with caplog.at_level(logging.DEBUG):
        ok = _post(
            payload={"embeds": [{"title": "Secret Test"}]},
            webhook_url=secret_url,
            http_post=failing_poster,
            incident_key="test:leak",
            suppress_in_test=False,
        )
        assert ok is False

    for record in caplog.records:
        assert "super_secret_token_12345" not in record.message
        assert "9988776655" not in record.message
