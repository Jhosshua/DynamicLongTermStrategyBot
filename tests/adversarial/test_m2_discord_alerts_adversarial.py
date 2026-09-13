"""tests/adversarial/test_m2_discord_alerts_adversarial.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adversarial Stress & Vulnerability Verification Suite for Milestone M2:
Discord v2 Institutional Alert Engine (bot.discord_alerts).

Empirically attacks:
1. Extreme Payload Boundaries:
   - Single field exceeding 1,024 characters.
   - Total embed characters exceeding 6,000 across multiple fields.
   - Embed with large description + multiple medium fields.
   - Long markdown code fences with unclosed or nested backticks.
   - JSON serialization integrity under adversarial characters.
2. Secret Redaction:
   - Webhook URLs with private tokens in error_message, description, status_info, recovery_evidence.
   - Credentials (passwords, tokens, API keys) in unredacted card builder fields.
   - Sensitive headers (Authorization: Bearer ..., x-relay-token: ..., Cookie: ...).
   - Log and exception containment.
3. Pytest Isolation & Socket Suppression:
   - Zero socket connections to discord.com under pytest.
   - Investigation of the latent production suppression bug in _under_pytest(suppress_in_test).
   - Socket interception when mock poster is absent vs present.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from bot.discord_alerts import (
    DISCORD_MAX_DESCRIPTION_LEN,
    DISCORD_MAX_FIELD_NAME_LEN,
    DISCORD_MAX_FIELD_VALUE_LEN,
    DISCORD_MAX_TITLE_LEN,
    DiscordEmbedCard,
    DiscordEmbedField,
    DiscordNotifier,
    _post,
    _redact_secrets,
    _sanitize_field_value,
    _under_pytest,
    broken_card,
    build_broken_card,
    build_recovered_card,
    build_trade_execution_card,
    recovered_card,
    trade_card,
)


# ============================================================================
# SUITE 1: Extreme Payload Sizes & JSON Syntax Safety
# ============================================================================

def test_single_field_exceeding_1024_characters():
    """Verify that fields exceeding 1,024 characters are clamped without breaking JSON."""
    huge_val = "A" * 3500
    field = DiscordEmbedField(name="HugeField", value=huge_val)
    d = field.to_dict()

    assert len(d["value"]) <= DISCORD_MAX_FIELD_VALUE_LEN
    assert d["value"] == huge_val[:DISCORD_MAX_FIELD_VALUE_LEN]

    # Check JSON validity
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)
    assert deserialized["value"] == d["value"]


def test_adversarial_total_embed_size_exceeding_6000_multi_field():
    """EMPIRICAL ATTACK: Inject 15 fields of 500 characters each into DiscordEmbedCard.

    Discord enforces a strict 6,000 total character ceiling across all title,
    description, fields (name + value), footer, and author characters.
    15 fields * 500 characters = 7,500 characters.

    Vulnerability: DiscordEmbedCard.to_dict() attempts to downsize only the single
    largest field if `len(curr_val) > excess + 40`. When multiple distributed fields
    create the excess, `len(curr_val) > excess + 40` evaluates to False, causing
    ZERO truncation to occur and producing an embed > 6,000 characters that triggers
    HTTP 400 Bad Request on Discord.
    """
    fields = [{"name": f"Field_{i:02d}", "value": "X" * 500} for i in range(15)]
    card = DiscordEmbedCard(
        color=0x1E88E5,
        title="Adversarial Multi-Field Overflow Test",
        description="Testing multi-field distributed overflow behavior.",
        fields=fields,
    )
    d = card.to_dict()

    total_len = len(d.get("title", "")) + len(d.get("description", "")) + len(d.get("footer", {}).get("text", ""))
    for f in d.get("fields", []):
        total_len += len(f["name"]) + len(f["value"])

    # Empirical check: Discord hard limit is 6000
    print(f"\n[EMPIRICAL] Multi-field total embed length: {total_len} (Discord max: 6000)")
    assert total_len <= 6000, (
        f"VULNERABILITY CONFIRMED: Embed total character count is {total_len}, "
        f"exceeding Discord's 6,000 hard ceiling. Discord will reject with HTTP 400."
    )


def test_adversarial_total_embed_size_large_desc_plus_fields():
    """EMPIRICAL ATTACK: Inject 4,000 char description + 5 fields of 500 chars.

    Total length = 4000 + 2500 = 6,500 characters.
    Excess = 6500 - 5800 = 700.
    Largest field is 500 chars.
    `len(curr_val) > excess + 40` -> 500 > 740 is FALSE!
    No field is truncated, and description is not truncated either.
    """
    desc = "D" * 4000
    fields = [{"name": f"Field_{i}", "value": "V" * 500} for i in range(5)]
    card = DiscordEmbedCard(
        color=0xE53935,
        title="Large Desc Overflow",
        description=desc,
        fields=fields,
    )
    d = card.to_dict()

    total_len = len(d.get("title", "")) + len(d.get("description", "")) + len(d.get("footer", {}).get("text", ""))
    for f in d.get("fields", []):
        total_len += len(f["name"]) + len(f["value"])

    print(f"\n[EMPIRICAL] Large desc + fields total embed length: {total_len} (Discord max: 6000)")
    assert total_len <= 6000, (
        f"VULNERABILITY CONFIRMED: Large description + fields produced {total_len} chars > 6,000."
    )


def test_markdown_code_fence_sanitization_and_nesting():
    """Attack code fence handling with already-wrapped and truncated backtick sequences."""
    # 1. Truncation of long code fence
    raw_code = "```python\n" + ("x = 100\n" * 200) + "```"
    sanitized = _sanitize_field_value(raw_code, max_chars=1024, wrap_code=True)
    assert len(sanitized) <= 1024
    # Check if backticks are duplicated: wrap_code=True on existing ``` produces ``````
    print(f"\n[EMPIRICAL] Sanitized code fence prefix: {repr(sanitized[:20])}")
    # Verify valid JSON serialization regardless of markdown backticks
    payload = {"embeds": [{"fields": [{"name": "Code", "value": sanitized}]}]}
    dumped = json.dumps(payload)
    loaded = json.loads(dumped)
    assert loaded["embeds"][0]["fields"][0]["value"] == sanitized


def test_adversarial_special_characters_json_integrity():
    """Inject quotes, unescaped backslashes, tabs, null bytes, unicode emojis."""
    adversarial_inputs = [
        '{"malicious": "json", "injection": true}',
        'Special chars: \x00 \r \n \t \\ / " \' ` ~ ! @ # $ % ^ & * ( )',
        'Unicode emojis: 🚨 ⚖️ ✅ 🚀 📈 📉 💰 \U0001F600 \U0001F911',
        'Unbalanced markdown: ```python def foo(): "unclosed string',
        'Backslash bombs: ' + ('\\' * 500),
    ]
    for raw in adversarial_inputs:
        sanitized = _sanitize_field_value(raw, max_chars=1024, wrap_code=False)
        card = DiscordEmbedCard(
            color=0x1E88E5,
            title="JSON Integrity",
            description=sanitized,
            fields=[{"name": "Input", "value": sanitized}],
        )
        d = card.to_dict()
        # Ensure serialization produces strictly valid JSON without error
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["title"] == "JSON Integrity"
        assert len(deserialized["fields"]) == 1


# ============================================================================
# SUITE 2: Secret Redaction Verification
# ============================================================================

def test_secret_redaction_webhook_tokens():
    """Verify webhook token redaction in _redact_secrets."""
    raw = "Alert sent to https://discord.com/api/webhooks/1234567890/abcDEFghiJKLmnoPQR_stuVWX successfully"
    redacted = _redact_secrets(raw)
    assert "abcDEFghiJKLmnoPQR_stuVWX" not in redacted
    assert "1234567890" not in redacted
    assert "[REDACTED_ID]/[REDACTED_TOKEN]" in redacted


def test_adversarial_secret_leakage_in_build_broken_card():
    """EMPIRICAL ATTACK: Inject sensitive webhook URL and credentials in error_message.

    Vulnerability: build_broken_card calls _sanitize_field_value on `evidence`,
    but passes `error_message` directly into `description` and `"What broke"` field
    WITHOUT calling _sanitize_field_value or _redact_secrets!
    """
    secret_url = "https://discord.com/api/webhooks/9988776655/super_secret_token_12345"
    secret_cred = "password=my_ultra_secret_db_password"
    leak_message = f"Connection failed to {secret_url} with auth {secret_cred}"

    card = build_broken_card(
        component="AlpacaRelayClient",
        error_message=leak_message,
        evidence="Normal error trace",
    )
    d = card.to_dict()

    # Check description
    print(f"\n[EMPIRICAL] Broken card description: {d['description']}")
    assert "super_secret_token_12345" not in d["description"], (
        "VULNERABILITY CONFIRMED: Secret webhook token leaked into broken card description!"
    )
    assert "my_ultra_secret_db_password" not in d["description"], (
        "VULNERABILITY CONFIRMED: Secret password leaked into broken card description!"
    )

    # Check What broke field
    what_broke_val = next((f["value"] for f in d["fields"] if f["name"] == "What broke"), "")
    print(f"[EMPIRICAL] What broke field: {what_broke_val}")
    assert "super_secret_token_12345" not in what_broke_val, (
        "VULNERABILITY CONFIRMED: Secret webhook token leaked into 'What broke' field!"
    )


def test_adversarial_secret_leakage_in_build_recovered_card():
    """EMPIRICAL ATTACK: Inject secrets into recovery_evidence and status_info.

    Vulnerability: build_recovered_card does NOT call _sanitize_field_value or
    _redact_secrets on `status_info` or `recovery_evidence`!
    """
    secret_url = "https://discord.com/api/webhooks/1122334455/recovery_secret_token_abc"
    secret_key = "token=alpaca_live_key_99999"

    card = build_recovered_card(
        component="FeedManager",
        downtime_duration_s=42.0,
        status_info=f"Stream recovered using {secret_key}",
        recovery_evidence=f"Handshake verified with {secret_url}",
    )
    d = card.to_dict()

    rec_ev_val = next((f["value"] for f in d["fields"] if f["name"] == "Recovery evidence"), "")
    telemetry_val = next((f["value"] for f in d["fields"] if f["name"] == "Telemetry Status"), "")

    print(f"\n[EMPIRICAL] Recovery evidence field: {rec_ev_val}")
    print(f"[EMPIRICAL] Telemetry status field: {telemetry_val}")

    assert "recovery_secret_token_abc" not in rec_ev_val, (
        "VULNERABILITY CONFIRMED: Secret webhook token leaked into 'Recovery evidence' field!"
    )
    assert "alpaca_live_key_99999" not in telemetry_val, (
        "VULNERABILITY CONFIRMED: Secret token leaked into 'Telemetry Status' field!"
    )


def test_adversarial_secret_redaction_sensitive_headers():
    """EMPIRICAL ATTACK: Test redaction of Bearer tokens and Authorization headers.

    Vulnerability: _redact_secrets only has patterns for:
    - https://discord.../webhooks/\d+/...
    - (token|key|secret|password)=...
    It has NO pattern for HTTP Authorization headers, Bearer tokens, or API keys
    formatted as 'Authorization: Bearer <token>' or 'x-relay-token: <key>'.
    """
    headers_text = (
        "Headers:\n"
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.live_secret_jwt\n"
        "x-relay-token: alpaca_relay_secret_token_456\n"
        "Cookie: session=secret_session_cookie_789\n"
    )
    redacted = _redact_secrets(headers_text)
    print(f"\n[EMPIRICAL] Redacted headers:\n{redacted}")

    assert "live_secret_jwt" not in redacted, (
        "VULNERABILITY CONFIRMED: Bearer JWT token not redacted by _redact_secrets!"
    )
    assert "alpaca_relay_secret_token_456" not in redacted, (
        "VULNERABILITY CONFIRMED: x-relay-token not redacted by _redact_secrets!"
    )


# ============================================================================
# SUITE 3: Strict Pytest Suppression & Socket Isolation
# ============================================================================

def test_zero_network_sockets_under_pytest():
    """Strictly verify that 0 network sockets connect to live Discord servers during test runs.

    Monitors socket.socket.connect: if any socket connection attempts to connect
    to discord.com or port 443/80, fails immediately.
    """
    attempted_connections = []
    real_connect = socket.socket.connect

    def mock_socket_connect(self, address):
        host, port = address[0], address[1]
        attempted_connections.append((host, port))
        if "discord" in str(host).lower() or port in (80, 443):
            raise RuntimeError(f"VIOLATION: Attempted network socket connect to {host}:{port} under pytest!")
        return real_connect(self, address)

    with patch.object(socket.socket, "connect", mock_socket_connect):
        notifier = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/12345/test_token",
            suppress_in_test=True,
        )
        assert notifier.is_pytest_environment() is True

        # Dispatch all card types without mock poster
        res1 = notifier.post_broken_alert("PaperLedger", "Test failure")
        res2 = notifier.post_recovered_alert("PaperLedger", 15.0)
        res3 = notifier.post_trade_execution(
            orders=[{"symbol": "SPY", "shares": 10, "price": 500.0, "action": "BUY"}],
            nav=50000.0,
        )

        assert res1 is True
        assert res2 is True
        assert res3 is True

    # Assert zero socket connections were attempted
    assert len(attempted_connections) == 0, (
        f"VIOLATION: Sockets connected under pytest: {attempted_connections}"
    )


def test_latent_production_suppression_bug_in_under_pytest():
    """EMPIRICAL ATTACK: Expose the critical bug in _under_pytest(explicit_flag).

    Vulnerability Analysis:
    Line 181 of bot/discord_alerts.py:
        def _under_pytest(explicit_flag: Optional[bool] = None) -> bool:
            if explicit_flag is not None:
                return explicit_flag
            return (os.environ.get("PYTEST_RUNNING") == "1" ...)

    Line 486 of bot/discord_alerts.py:
        def _post(..., suppress_in_test: bool = True, ...):
            is_test = _under_pytest(suppress_in_test)

    Line 717 of bot/discord_alerts.py:
        def is_pytest_environment(self) -> bool:
            return _under_pytest(self.suppress_in_test)

    Because `suppress_in_test` defaults to `True`, `explicit_flag` is passed as `True`!
    Therefore, `_under_pytest(True)` returns `True` UNCONDITIONALLY, even outside
    pytest in production!
    Consequently, in production, `is_test` is ALWAYS True, and line 493:
        if is_test and http_post is None:
            return True
    DROPS ALL REAL DISCORD ALERTS IN PRODUCTION!
    """
    # 1. Direct call to _under_pytest with default suppress_in_test value (True)
    result_with_flag = _under_pytest(True)
    print(f"\n[EMPIRICAL] _under_pytest(True) returns: {result_with_flag}")
    assert result_with_flag is True

    # 2. Simulate production environment by stripping pytest env vars
    clean_env = {
        k: v for k, v in os.environ.items()
        if k not in ("PYTEST_RUNNING", "PYTEST_CURRENT_TEST", "TESTING", "ENV")
    }
    with patch.dict(os.environ, clean_env, clear=True):
        # Even with environment cleared, because suppress_in_test=True is passed as explicit_flag:
        flagged_result = _under_pytest(True)
        print(f"[EMPIRICAL] Cleaned env _under_pytest(True): {flagged_result}")

        # BUT what does _under_pytest() without explicit flag return?
        # Temporarily mock sys.modules to exclude 'pytest'
        with patch.dict(sys.modules):
            sys.modules.pop("pytest", None)
            unflagged_result = _under_pytest(None)
            print(f"[EMPIRICAL] Cleaned env _under_pytest(None): {unflagged_result}")

        # The vulnerability: flagged_result is True, but actual environment is False!
        assert flagged_result is True
        assert unflagged_result is False

        # Impact: DiscordNotifier in production with default settings claims it is under pytest!
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/prod_token")
        # Under production (no http_post provided), notifier will drop alerts because of this bug!
        assert notifier.suppress_in_test is True
        print(f"[EMPIRICAL] notifier.is_pytest_environment() in prod: {notifier.is_pytest_environment()}")
        assert notifier.is_pytest_environment() is True, (
            "VULNERABILITY: In production, notifier.is_pytest_environment() returns True, "
            "causing silent drop of all live alerts!"
        )


def test_unsuppressed_mode_attempts_live_socket_connect():
    """EMPIRICAL ATTACK: Verify that setting suppress_in_test=False without mock poster
    attempts a live outbound socket connection to discord.com.
    """
    attempted_targets = []
    real_connect = socket.socket.connect

    def spy_socket_connect(self, address):
        host, port = address[0], address[1]
        attempted_targets.append((host, port))
        # Block the actual network call to prevent live hanging/bans
        raise ConnectionRefusedError(f"Blocked outbound socket to {host}:{port}")

    with patch.object(socket.socket, "connect", spy_socket_connect):
        notifier = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/12345/test_token",
            suppress_in_test=False,
            max_retries=0,
        )
        # Verify is_pytest_environment evaluates to False when explicit False is passed
        assert notifier.is_pytest_environment() is False

        # Attempt post
        res = notifier.post_broken_alert("TestComponent", "Test Error")
        assert res is False

    print(f"\n[EMPIRICAL] Socket targets intercepted: {attempted_targets}")
    # Verify an actual network socket connect was attempted to discord.com or its IP
    assert len(attempted_targets) > 0, "Expected socket connect attempt when suppress_in_test=False"

