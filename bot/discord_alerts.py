"""Institutional Discord v2 embed card dispatch engine with rate limiting and pytest suppression.

Fulfills PROJECT.md Features 6, 7, 8, 9 and ORIGINAL_REQUEST.md R3.
- Feature 6: Discord v2 Broken Alert Card (0xE53935)
- Feature 7: Discord v2 Recovered Alert Card (0x43A047)
- Feature 8: Discord v2 Trade Execution Card (0x1E88E5)
- Feature 9: Rate limiting (<= 1 post/2.0s, max sleep 5.0s) & Pytest suppression
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import inspect
import logging
import math
import os
import random
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

logger = logging.getLogger("bot.discord_alerts")

# Institutional Discord Card Color Codes
DISCORD_COLOR_BROKEN: int = 0xE53935     # Hex: #E53935 | Decimal: 15022389 | Red
DISCORD_COLOR_RECOVERED: int = 0x43A047  # Hex: #43A047 | Decimal: 4431943  | Green
DISCORD_COLOR_TRADE: int = 0x1E88E5      # Hex: #1E88E5 | Decimal: 2001125  | Blue

# Color Aliases
RED: int = DISCORD_COLOR_BROKEN
GREEN: int = DISCORD_COLOR_RECOVERED
BLUE: int = DISCORD_COLOR_TRADE


class DiscordColor:
    """Canonical Discord Embed Color Constants."""
    BROKEN: int = DISCORD_COLOR_BROKEN
    RECOVERED: int = DISCORD_COLOR_RECOVERED
    TRADE: int = DISCORD_COLOR_TRADE


class DiscordCardType(str, Enum):
    """Types of institutional Discord cards."""
    BROKEN = "broken"
    RECOVERED = "recovered"
    TRADE_EXECUTION = "trade_execution"


# Timing & Limit Constants
DEFAULT_RATE_LIMIT_INTERVAL_S: float = 2.0
DEFAULT_MAX_BACKOFF_SLEEP_S: float = 5.0
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_HTTP_TIMEOUT_S: float = 10.0
DEFAULT_RETRY_AFTER_SECONDS: float = 1.0
DISCORD_MAX_FIELD_VALUE_LEN: int = 1024
DISCORD_MAX_FIELD_NAME_LEN: int = 256
DISCORD_MAX_TITLE_LEN: int = 256
DISCORD_MAX_DESCRIPTION_LEN: int = 4096
DISCORD_MAX_EMBED_TOTAL_LEN: int = 6000

# Global process-wide rate-limiting synchronization
_process_lock = threading.Lock()
_last_post_monotonic: float = 0.0


@dataclass
class DiscordEmbedField:
    """Single embed field definition."""
    name: str
    value: str
    inline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name[:DISCORD_MAX_FIELD_NAME_LEN] if self.name else "Detail",
            "value": self.value[:DISCORD_MAX_FIELD_VALUE_LEN] if self.value else "(none)",
            "inline": bool(self.inline),
        }


@dataclass
class DiscordEmbedCard:
    """Discord API v10 Embed representation."""
    color: int
    title: str
    description: str = ""
    fields: List[Union[Dict[str, Any], DiscordEmbedField]] = field(default_factory=list)
    url: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    footer: Dict[str, str] = field(
        default_factory=lambda: {"text": "DynamicLongTermStrategyBot · Virtual Paper ($50k) · UTC"}
    )
    components: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Discord API v10 embed structure with limit guarding."""
        serialized_fields: List[Dict[str, Any]] = []
        for f in self.fields:
            if isinstance(f, DiscordEmbedField):
                serialized_fields.append(f.to_dict())
            elif isinstance(f, dict):
                safe_name = str(f.get("name", "Detail"))[:DISCORD_MAX_FIELD_NAME_LEN] or "Detail"
                safe_val = str(f.get("value", "(none)"))[:DISCORD_MAX_FIELD_VALUE_LEN] or "(none)"
                serialized_fields.append({
                    "name": safe_name,
                    "value": safe_val,
                    "inline": bool(f.get("inline", False)),
                })

        safe_title = self.title[:DISCORD_MAX_TITLE_LEN] if self.title else ""
        safe_desc = self.description[:DISCORD_MAX_DESCRIPTION_LEN] if self.description else ""

        embed: Dict[str, Any] = {
            "title": safe_title,
            "color": int(self.color),
            "description": safe_desc,
            "timestamp": self.timestamp,
            "fields": serialized_fields,
            "footer": self.footer or {"text": "DynamicLongTermStrategyBot · Virtual Paper ($50k) · UTC"},
        }
        if self.url:
            embed["url"] = self.url

        # Defensive total character bounding (Discord max 6000 chars, strict ceiling 5800)
        footer_text = str(embed.get("footer", {}).get("text", "")) if isinstance(embed.get("footer"), dict) else ""

        def _calc_total_len(t: str, d: str, ft: str, s_fields: List[Dict[str, Any]]) -> int:
            return len(t) + len(d) + len(ft) + sum(len(f["name"]) + len(f["value"]) for f in s_fields)

        total_len = _calc_total_len(safe_title, safe_desc, footer_text, serialized_fields)
        target_ceiling = 5800

        # Iteratively truncate fields and description until total_len <= target_ceiling
        iteration = 0
        while total_len > target_ceiling and iteration < 50:
            iteration += 1
            excess = total_len - target_ceiling

            max_field_val_len = max((len(f["value"]) for f in serialized_fields), default=0)
            if len(safe_desc) > 300 and len(safe_desc) >= max_field_val_len:
                trim_budget = min(excess + 40, len(safe_desc) - 50)
                safe_desc = safe_desc[: max(10, len(safe_desc) - trim_budget)] + "\n[TRUNCATED TO MEET DISCORD LIMITS]"
                embed["description"] = safe_desc
            elif serialized_fields and max_field_val_len > 30:
                largest_idx = max(range(len(serialized_fields)), key=lambda i: len(serialized_fields[i]["value"]))
                curr_val = serialized_fields[largest_idx]["value"]
                trim_budget = min(excess + 40, len(curr_val) - 30)
                serialized_fields[largest_idx]["value"] = (
                    curr_val[: max(10, len(curr_val) - trim_budget)] + "\n[TRUNCATED TO MEET DISCORD LIMITS]"
                )
            elif len(safe_desc) > 50:
                trim_budget = min(excess + 40, len(safe_desc) - 20)
                safe_desc = safe_desc[: max(10, len(safe_desc) - trim_budget)] + "\n[TRUNCATED TO MEET DISCORD LIMITS]"
                embed["description"] = safe_desc
            elif serialized_fields:
                serialized_fields.pop()
            else:
                safe_title = safe_title[: max(10, len(safe_title) - excess)]
                embed["title"] = safe_title

            total_len = _calc_total_len(safe_title, safe_desc, footer_text, serialized_fields)

        embed["fields"] = serialized_fields
        embed["description"] = safe_desc
        embed["title"] = safe_title
        return embed


@dataclass
class DiscordConfig:
    """Configuration options for Discord alerting subsystem."""
    webhook_url: Optional[str] = None
    rate_limit_interval_s: float = DEFAULT_RATE_LIMIT_INTERVAL_S
    max_backoff_sleep_s: float = DEFAULT_MAX_BACKOFF_SLEEP_S
    max_retries: int = DEFAULT_MAX_RETRIES
    suppress_in_test: bool = True
    default_dashboard_url: str = "https://dynamiclongtermstrategybot-production.up.railway.app"
    http_timeout_s: float = DEFAULT_HTTP_TIMEOUT_S


# Configuration alias
DiscordNotifierConfig = DiscordConfig


# ----------------------------------------------------------------------------
# Formatting & Sanitization Utilities
# ----------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """Format duration in seconds into clean human-readable representation."""
    safe_duration = max(0.0, float(seconds))
    if safe_duration < 60.0:
        return f"{safe_duration:.1f}s"
    minutes = int(safe_duration // 60)
    sec = int(safe_duration % 60)
    if minutes >= 60:
        hours = int(minutes // 60)
        rem_min = int(minutes % 60)
        return f"{hours}h {rem_min}m {sec}s" if sec > 0 else f"{hours}h {rem_min}m"
    return f"{minutes}m {sec}s" if sec > 0 else f"{minutes}m"


def _under_pytest(explicit_flag: Optional[bool] = None) -> bool:
    """Robust tri-layer detection of active pytest test execution."""
    if explicit_flag is not None:
        return explicit_flag
    return (
        os.environ.get("PYTEST_RUNNING") == "1"
        or "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
        or os.environ.get("ENV") == "test"
        or os.environ.get("TESTING") == "1"
    )


def _redact_secrets(text: str) -> str:
    """Mask webhook tokens, private credentials, auth headers, and tokens in messages."""
    if not text:
        return ""
    # 1. Redact full discord webhook URL tokens (including versioned /api/v10/webhooks/... paths)
    text = re.sub(
        r"https://discord(?:app)?\.com/api/(?:v\d+/)?webhooks/\d+/[A-Za-z0-9_\-]+",
        "https://discord.com/api/webhooks/[REDACTED_ID]/[REDACTED_TOKEN]",
        text,
    )
    # 2. Redact Authorization Bearer and Basic headers (JWTs, tokens)
    text = re.sub(
        r"(Authorization:\s*(?:Bearer|Basic)\s+)[A-Za-z0-9_\-\.\+/=]+",
        r"\1[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    # 3. Redact sensitive relay/api headers (e.g. x-relay-token: abc, api-key: abc)
    text = re.sub(
        r"((?:x-relay-token|relay[_-]?token|api[_-]?key)\s*[:=]\s*)[A-Za-z0-9_\-]+",
        r"\1[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    # 4. Redact Cookie session tokens
    text = re.sub(
        r"(Cookie:\s*session=)[A-Za-z0-9_\-]+",
        r"\1[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    # 5. Redact generic inline credentials (token=xxx, key=xxx, secret=xxx, password=xxx)
    text = re.sub(
        r"(token|key|secret|password)=([a-zA-Z0-9_\-]+)",
        r"\1=[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _sanitize_field_value(val: Any, max_chars: int = DISCORD_MAX_FIELD_VALUE_LEN, wrap_code: bool = False) -> str:
    """Safely clamp string, protect against empty values, and bound code fences."""
    if val is None:
        return "```(none)```" if wrap_code else "(none)"
    val_str = str(val).strip()
    if not val_str:
        return "```(none)```" if wrap_code else "(none)"

    val_str = _redact_secrets(val_str)
    budget = max_chars - (6 if wrap_code else 0)
    if len(val_str) > budget:
        val_str = val_str[: budget - 3] + "..."
    return f"```{val_str}```" if wrap_code else val_str


def _extract_retry_after(response: Any, default_delay: float = DEFAULT_RETRY_AFTER_SECONDS) -> float:
    """Parse retry_after in seconds from Discord 429 response body or headers."""
    headers = getattr(response, "headers", None) or {}

    # 1. Try JSON body
    try:
        if hasattr(response, "json"):
            body = response.json()
            if callable(body):
                body = body()
            if isinstance(body, dict):
                if "retry_after_ms" in body:
                    raw = float(body["retry_after_ms"])
                    if math.isfinite(raw) and raw >= 0.0:
                        return raw / 1000.0
                if "retry_after" in body:
                    raw = float(body["retry_after"])
                    if math.isfinite(raw) and raw >= 0.0:
                        is_ms = bool(
                            body.get("is_milliseconds")
                            or str(body.get("unit", "")).lower() in ("ms", "millisecond", "milliseconds")
                            or (hasattr(headers, "get") and "milli" in str(headers.get("X-RateLimit-Precision", "")).lower())
                        )
                        if is_ms and raw >= 1000.0:
                            return raw / 1000.0
                        return raw
    except Exception:
        pass

    # 2. Try Headers
    if headers and hasattr(headers, "get"):
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if ra is not None:
            try:
                ra_str = str(ra).strip()
                if ra_str.lower().endswith("ms"):
                    raw = float(ra_str[:-2])
                    if math.isfinite(raw) and raw >= 0.0:
                        return raw / 1000.0
                raw = float(ra_str)
                if math.isfinite(raw) and raw >= 0.0:
                    is_ms = "milli" in str(headers.get("X-RateLimit-Precision", "")).lower()
                    if is_ms and raw >= 1000.0:
                        return raw / 1000.0
                    return raw
            except (ValueError, TypeError):
                pass

        reset_after = headers.get("X-RateLimit-Reset-After") or headers.get("x-ratelimit-reset-after")
        if reset_after is not None:
            try:
                raw = float(reset_after)
                if math.isfinite(raw) and raw >= 0.0:
                    return raw
            except (ValueError, TypeError):
                pass

    return default_delay


# ----------------------------------------------------------------------------
# Card Builders
# ----------------------------------------------------------------------------

def build_broken_card(
    component: str,
    error_message: str,
    evidence: str = "",
    since: Optional[Any] = None,
    impact: str = "",
    action_required: str = "",
    dashboard_url: str = "",
) -> DiscordEmbedCard:
    """Build institutional red embed card (0xE53935) for broken/failure alerts."""
    url = dashboard_url or "https://dynamiclongtermstrategybot-production.up.railway.app"
    since_str: str
    if since is None:
        since_str = datetime.now(timezone.utc).isoformat()
    elif isinstance(since, datetime):
        since_str = since.isoformat()
    else:
        since_str = str(since)

    safe_component = _redact_secrets(component or "Unknown")
    safe_error_msg = _sanitize_field_value(error_message, max_chars=1024, wrap_code=False) or "System failure detected"
    safe_evidence = _sanitize_field_value(evidence, max_chars=1024, wrap_code=True)
    safe_impact = _sanitize_field_value(
        impact or "Real-time pricing paused. Engine running on cached bars. Order generation frozen in STALE_DATA_HOLD.",
        max_chars=512,
        wrap_code=False,
    )
    safe_action = _sanitize_field_value(
        action_required or "1. Verify AlpacaRelay health on Railway\n2. Inspect upstream stream authentication\n3. Check dashboard alert banner",
        max_chars=512,
        wrap_code=False,
    )

    fields: List[Dict[str, Any]] = [
        {"name": "Component", "value": safe_component, "inline": True},
        {"name": "Since", "value": since_str, "inline": True},
        {"name": "What broke", "value": safe_error_msg, "inline": False},
        {"name": "Evidence", "value": safe_evidence, "inline": False},
        {"name": "Impact", "value": safe_impact, "inline": False},
        {"name": "Action required", "value": safe_action, "inline": False},
        {"name": "What to do now", "value": safe_action, "inline": False},
        {"name": "Operator Dashboard", "value": f"[Open Dashboard]({url})", "inline": True},
        {"name": "Public Dashboard", "value": f"[Open Dashboard]({url})", "inline": True},
    ]

    action_row = {
        "type": 1,
        "components": [
            {
                "type": 2,
                "style": 5,
                "label": "Open Dashboard",
                "url": url,
            }
        ],
    }

    return DiscordEmbedCard(
        color=DISCORD_COLOR_BROKEN,
        title=f"🚨 [BROKEN] Dynamic Long-Term Strategy: {safe_component} BROKEN",
        description=f"**Error**: {_redact_secrets(error_message)}",
        fields=fields,
        url=url,
        components=[action_row],
    )


def build_recovered_card(
    component: str,
    downtime_duration_s: float,
    status_info: str = "",
    recovery_evidence: str = "",
    dashboard_url: str = "",
) -> DiscordEmbedCard:
    """Build institutional green embed card (0x43A047) for recovery notices."""
    url = dashboard_url or "https://dynamiclongtermstrategybot-production.up.railway.app"
    duration_fmt = format_duration(downtime_duration_s)
    safe_component = _redact_secrets(component or "Unknown")
    raw_evidence = recovery_evidence or status_info or "Stream restored and historical bars synchronized."
    raw_telemetry = status_info or "Ingestion state: MONITORING_STREAM · Fallback banner cleared · Rebalancing unlocked"
    safe_evidence = _sanitize_field_value(raw_evidence, max_chars=1024, wrap_code=False)
    safe_telemetry = _sanitize_field_value(raw_telemetry, max_chars=1024, wrap_code=False)

    fields: List[Dict[str, Any]] = [
        {"name": "Component", "value": safe_component, "inline": True},
        {"name": "Downtime Duration", "value": duration_fmt, "inline": True},
        {"name": "Down for", "value": duration_fmt, "inline": True},
        {"name": "What recovered", "value": "WebSocket reconnected, authenticated, and resubscribed to universe symbols.", "inline": False},
        {"name": "Recovery evidence", "value": safe_evidence, "inline": False},
        {"name": "Evidence", "value": safe_evidence, "inline": False},
        {"name": "Telemetry Status", "value": safe_telemetry, "inline": False},
        {"name": "Current Status", "value": safe_telemetry, "inline": False},
        {"name": "Operator Dashboard", "value": f"[Open Dashboard]({url})", "inline": True},
        {"name": "Public Dashboard", "value": f"[Open Dashboard]({url})", "inline": True},
    ]

    action_row = {
        "type": 1,
        "components": [
            {
                "type": 2,
                "style": 5,
                "label": "Open Dashboard",
                "url": url,
            }
        ],
    }

    return DiscordEmbedCard(
        color=DISCORD_COLOR_RECOVERED,
        title=f"✅ [RECOVERED] Dynamic Long-Term Strategy: {safe_component} RECOVERED",
        description=f"Service successfully restored after {duration_fmt} downtime.",
        fields=fields,
        url=url,
        components=[action_row],
    )


def build_trade_execution_card(
    orders: Sequence[Any],
    nav: float,
    cash: Optional[float] = None,
    regime: str = "BULL_NORMAL",
    execution_time: Optional[Any] = None,
    dashboard_url: str = "",
) -> DiscordEmbedCard:
    """Build institutional blue embed card (0x1E88E5) for trade executions."""
    if not orders:
        raise ValueError("Orders list cannot be empty for trade execution notification")

    url = dashboard_url or "https://dynamiclongtermstrategybot-production.up.railway.app"
    trade_url = f"{url}?view=trades" if "?" not in url else url

    order_lines: List[str] = []
    for o in orders[:10]:
        action = getattr(o, "action", getattr(o, "side", "ORDER"))
        if hasattr(action, "value"):
            action = action.value
        act_str = str(action).upper()

        sym = getattr(o, "symbol", "") or (o.get("symbol", "") if isinstance(o, dict) else "")
        shares = getattr(o, "shares", getattr(o, "qty", getattr(o, "delta_shares", 0.0)))
        if isinstance(o, dict):
            shares = o.get("shares", o.get("qty", o.get("delta_shares", 0.0)))
        shares_val = abs(float(shares or 0.0))

        price = getattr(o, "price", getattr(o, "estimated_price", 0.0))
        if isinstance(o, dict):
            price = o.get("price", o.get("estimated_price", 0.0))
        price_val = float(price or 0.0)

        notional = getattr(o, "notional", shares_val * price_val)
        if isinstance(o, dict):
            notional = o.get("notional", shares_val * price_val)
        notional_val = float(notional or 0.0)
        sign = "+" if act_str == "SELL" else "-"

        order_lines.append(f"`{act_str}` **{sym}**: {shares_val:.2f} shs @ ${price_val:.2f} ({sign}${notional_val:,.2f})")

    if len(orders) > 10:
        order_lines.append(f"... and {len(orders) - 10} more orders")

    orders_detail_str = "\n".join(order_lines)

    fields: List[Dict[str, Any]] = [
        {"name": "Regime", "value": regime, "inline": True},
        {"name": "Regime & Cadence", "value": f"Regime: `{regime}`\nTrigger: Cadence / Operator Rebalance", "inline": True},
        {"name": "Portfolio NAV", "value": f"${nav:,.2f}", "inline": True},
    ]

    if cash is not None:
        fields.append({"name": "Cash balance", "value": f"${cash:,.2f}", "inline": True})
        fields.append({
            "name": "Portfolio Account ($50k Base)",
            "value": f"• **Cash Balance**: ${cash:,.2f}\n• **Total NAV**: ${nav:,.2f}",
            "inline": False,
        })

    fields.extend([
        {"name": "Orders Detail", "value": orders_detail_str, "inline": False},
        {"name": "Executed Orders (Fills)", "value": orders_detail_str, "inline": False},
        {"name": "Execution Metadata", "value": "Cadence: `WEEKLY_REBALANCE` · Drift band: ±2.5% · Status: `FILLED` (Simulated MOC)", "inline": False},
        {"name": "Operator Dashboard", "value": f"[Open Dashboard]({url})", "inline": True},
        {"name": "Public Dashboard", "value": f"[View Trades on Dashboard]({trade_url})", "inline": True},
    ])

    action_row = {
        "type": 1,
        "components": [
            {
                "type": 2,
                "style": 5,
                "label": "View Trades on Dashboard",
                "url": trade_url,
            }
        ],
    }

    return DiscordEmbedCard(
        color=DISCORD_COLOR_TRADE,
        title=f"⚖️ [TRADE EXECUTION] Virtual Paper Trade: {regime} Rebalance ({len(orders)} orders)",
        description=f"Executed {len(orders)} rebalance orders at NAV **${nav:,.2f}**.",
        fields=fields,
        url=trade_url,
        components=[action_row],
    )


# ----------------------------------------------------------------------------
# Transport, Rate Limiting & 429 Retry Engine
# ----------------------------------------------------------------------------

def _default_httpx_poster(timeout_s: float = DEFAULT_HTTP_TIMEOUT_S) -> Callable:
    """Produce synchronous httpx post function."""
    import httpx

    def _post(url: str, json: dict):
        return httpx.post(url, json=json, timeout=timeout_s)

    return _post


def _post(
    payload: Dict[str, Any],
    webhook_url: Optional[str] = None,
    http_post: Optional[Callable] = None,
    rate_limit_interval_s: float = DEFAULT_RATE_LIMIT_INTERVAL_S,
    max_backoff_sleep_s: float = DEFAULT_MAX_BACKOFF_SLEEP_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    incident_key: str = "alert",
    suppress_in_test: bool = True,
    lock: Optional[threading.Lock] = None,
) -> bool:
    """Core webhook posting pipeline with process-wide rate limiting and 429 backoff."""
    is_test = bool(suppress_in_test and _under_pytest())

    # If running under pytest and no mock poster is provided, drop network I/O immediately
    if is_test and http_post is None:
        logger.debug("alerts: suppressed Discord webhook under pytest: %s", incident_key)
        return True

    # If not under pytest and no webhook URL configured, drop gracefully
    target_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not target_url and http_post is None:
        logger.warning("alerts: no Discord webhook URL configured; dropping %s", incident_key)
        return False

    poster = http_post or _default_httpx_poster()
    skip_sleep = is_test

    global _last_post_monotonic
    use_lock = lock or _process_lock

    # HTTP Transmission Loop with per-attempt Rate Pacing & 429 Backoff
    for attempt in range(max_retries + 1):
        with use_lock:
            now_mono = time.monotonic()
            elapsed = now_mono - _last_post_monotonic
            sleep_needed = max(0.0, rate_limit_interval_s - elapsed)
            if sleep_needed > 0.0 and not skip_sleep:
                time.sleep(min(sleep_needed, max_backoff_sleep_s))
            _last_post_monotonic = time.monotonic()

        try:
            resp = poster(target_url, json=payload)
        except Exception as exc:
            logger.error("alerts: failed to post %s (%s)", incident_key, type(exc).__name__)
            return False

        status = getattr(resp, "status_code", 0)
        if 200 <= status < 300:
            return True

        if status == 429:
            delay = _extract_retry_after(resp, default_delay=rate_limit_interval_s)

            # Hard ceiling protection: if Discord asks for >= 5.0s, drop to avoid blocking
            if delay >= max_backoff_sleep_s:
                logger.error(
                    "alerts: discord 429 asks for %.1fs (>= %.1fs cap); dropping alert %s",
                    delay,
                    max_backoff_sleep_s,
                    incident_key,
                )
                return False

            if attempt < max_retries:
                jitter = random.uniform(0.1, 0.5)
                backoff_delay = max(delay, rate_limit_interval_s) * (1.5 ** attempt) + jitter
                if backoff_delay >= max_backoff_sleep_s:
                    logger.error(
                        "alerts: discord 429 backoff delay %.1fs >= %.1fs cap; aborting retry for %s",
                        backoff_delay,
                        max_backoff_sleep_s,
                        incident_key,
                    )
                    return False
                clamped_delay = min(backoff_delay, max_backoff_sleep_s)
                logger.warning(
                    "alerts: discord 429 for %s, retry %d/%d in %.2fs",
                    incident_key,
                    attempt + 1,
                    max_retries,
                    clamped_delay,
                )
                if not skip_sleep:
                    t_before = time.monotonic()
                    time.sleep(clamped_delay)
                    t_after = time.monotonic()
                    with use_lock:
                        # If time.monotonic() did not advance (e.g. mock sleep without virtual clock),
                        # advance _last_post_monotonic so backoff sleep satisfies this thread's pacing.
                        if (t_after - t_before) < (clamped_delay * 0.5):
                            _last_post_monotonic = time.monotonic() - rate_limit_interval_s
                continue

            logger.error("alerts: dropped %s after %d retries (429)", incident_key, max_retries)
            return False

        logger.error("alerts: discord returned HTTP %s for %s", status, incident_key)
        return False

    return False


# ----------------------------------------------------------------------------
# Standalone Functional Dispatchers
# ----------------------------------------------------------------------------

def broken_card(
    component: str,
    error_message: str = "",
    evidence: str = "",
    dashboard_url: str = "",
    since: Optional[Any] = None,
    impact: str = "",
    action_required: str = "",
    what_broke: Optional[str] = None,
    webhook_url: Optional[str] = None,
    http_post: Optional[Callable] = None,
    suppress_in_test: bool = True,
    **kwargs,
) -> bool:
    """Build and dispatch institutional broken card."""
    err = what_broke or error_message
    card = build_broken_card(
        component=component,
        error_message=err,
        evidence=evidence,
        since=since,
        impact=impact,
        action_required=action_required,
        dashboard_url=dashboard_url,
    )
    payload_dict = card.to_dict()
    payload = {"embeds": [payload_dict], "allowed_mentions": {"parse": []}}
    if card.components:
        payload["components"] = card.components
    return _post(
        payload=payload,
        webhook_url=webhook_url,
        http_post=http_post,
        incident_key=f"broken:{component}",
        suppress_in_test=suppress_in_test,
    )


def recovered_card(
    component: str,
    downtime_duration_s: float = 0.0,
    status_info: str = "",
    dashboard_url: str = "",
    recovery_evidence: str = "",
    downtime_s: Optional[float] = None,
    webhook_url: Optional[str] = None,
    http_post: Optional[Callable] = None,
    suppress_in_test: bool = True,
    **kwargs,
) -> bool:
    """Build and dispatch institutional recovered card."""
    dt_s = downtime_s if downtime_s is not None else downtime_duration_s
    card = build_recovered_card(
        component=component,
        downtime_duration_s=dt_s,
        status_info=status_info,
        recovery_evidence=recovery_evidence,
        dashboard_url=dashboard_url,
    )
    payload_dict = card.to_dict()
    payload = {"embeds": [payload_dict], "allowed_mentions": {"parse": []}}
    if card.components:
        payload["components"] = card.components
    return _post(
        payload=payload,
        webhook_url=webhook_url,
        http_post=http_post,
        incident_key=f"recovered:{component}",
        suppress_in_test=suppress_in_test,
    )


def trade_card(
    orders: Sequence[Any],
    nav: float,
    cash: Optional[float] = None,
    regime: str = "BULL_NORMAL",
    execution_time: Optional[Any] = None,
    dashboard_url: str = "",
    cash_balance: Optional[float] = None,
    webhook_url: Optional[str] = None,
    http_post: Optional[Callable] = None,
    suppress_in_test: bool = True,
    **kwargs,
) -> bool:
    """Build and dispatch institutional trade execution card."""
    effective_cash = cash if cash is not None else cash_balance
    card = build_trade_execution_card(
        orders=orders,
        nav=nav,
        cash=effective_cash,
        regime=regime,
        execution_time=execution_time,
        dashboard_url=dashboard_url,
    )
    payload_dict = card.to_dict()
    payload = {"embeds": [payload_dict], "allowed_mentions": {"parse": []}}
    if card.components:
        payload["components"] = card.components
    return _post(
        payload=payload,
        webhook_url=webhook_url,
        http_post=http_post,
        incident_key=f"trade:{regime}",
        suppress_in_test=suppress_in_test,
    )


# ----------------------------------------------------------------------------
# DiscordNotifier Class (Dual Synchronous / Asynchronous)
# ----------------------------------------------------------------------------

class DiscordNotifier:
    """Institutional Discord v2 Embed Alert Notifier.

    Provides synchronous and asynchronous compatible dispatch methods,
    process-wide rate limiting (<= 1 post / 2.0s), Discord 429 exponential backoff,
    and automatic pytest suppression with an in-memory card audit list.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        rate_limit_interval_s: float = DEFAULT_RATE_LIMIT_INTERVAL_S,
        max_backoff_sleep_s: float = DEFAULT_MAX_BACKOFF_SLEEP_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        suppress_in_test: bool = True,
        dashboard_url: Optional[str] = None,
        http_post: Optional[Callable] = None,
        http_post_fn: Optional[Callable] = None,
        http_timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
        lock: Optional[threading.Lock] = None,
    ):
        self.webhook_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
        self.rate_limit_interval_s = max(0.0, float(rate_limit_interval_s))
        self.max_backoff_sleep_s = max(0.0, float(max_backoff_sleep_s))
        self.max_retries = max_retries
        self.suppress_in_test = suppress_in_test
        self.dashboard_url = dashboard_url or "https://dynamiclongtermstrategybot-production.up.railway.app"
        self._http_post = http_post or http_post_fn
        self.http_timeout_s = http_timeout_s

        self._lock = lock or _process_lock
        self.last_post_time: float = 0.0
        self.dispatched_cards: List[DiscordEmbedCard] = []

    def is_pytest_environment(self) -> bool:
        """Check if pytest environment is active and suppression is enabled."""
        return bool(self.suppress_in_test and _under_pytest())

    def _prepare_and_post(
        self,
        card: DiscordEmbedCard,
        incident_key: str,
        http_post: Optional[Callable] = None,
    ) -> bool:
        """Store card in internal audit list and dispatch via rate-limited pipeline."""
        self.dispatched_cards.append(card)
        poster = http_post or self._http_post

        # If under pytest and no mock poster is supplied, bypass network I/O
        if self.is_pytest_environment() and poster is None:
            self.last_post_time = time.time()
            return True

        payload_dict = card.to_dict()
        payload = {"embeds": [payload_dict], "allowed_mentions": {"parse": []}}
        if card.components:
            payload["components"] = card.components

        result = _post(
            payload=payload,
            webhook_url=self.webhook_url,
            http_post=poster,
            rate_limit_interval_s=self.rate_limit_interval_s,
            max_backoff_sleep_s=self.max_backoff_sleep_s,
            max_retries=self.max_retries,
            incident_key=incident_key,
            suppress_in_test=self.suppress_in_test,
            lock=self._lock,
        )
        self.last_post_time = time.time()
        return result

    # ------------------------------------------------------------------------
    # Public Synchronous Alert Dispatchers
    # ------------------------------------------------------------------------

    def post_broken_alert(
        self,
        component: str,
        error_message: str,
        evidence: str = "",
        dashboard_url: Optional[str] = None,
        since: Optional[Any] = None,
        impact: str = "",
        action_required: str = "",
        http_post: Optional[Callable] = None,
        **kwargs,
    ) -> bool:
        """Dispatch institutional red embed card (0xE53935) on system failures."""
        url = dashboard_url or self.dashboard_url
        card = build_broken_card(
            component=component,
            error_message=error_message,
            evidence=evidence,
            since=since,
            impact=impact,
            action_required=action_required,
            dashboard_url=url,
        )
        return self._prepare_and_post(card, incident_key=f"broken:{component}", http_post=http_post)

    def post_recovered_alert(
        self,
        component: str,
        downtime_duration_s: float,
        status_info: str = "",
        dashboard_url: Optional[str] = None,
        recovery_evidence: str = "",
        http_post: Optional[Callable] = None,
        **kwargs,
    ) -> bool:
        """Dispatch institutional green embed card (0x43A047) on recovery."""
        url = dashboard_url or self.dashboard_url
        card = build_recovered_card(
            component=component,
            downtime_duration_s=downtime_duration_s,
            status_info=status_info,
            recovery_evidence=recovery_evidence,
            dashboard_url=url,
        )
        return self._prepare_and_post(card, incident_key=f"recovered:{component}", http_post=http_post)

    def post_trade_execution(
        self,
        orders: Sequence[Any],
        nav: float,
        regime: str = "BULL_NORMAL",
        dashboard_url: Optional[str] = None,
        cash: Optional[float] = None,
        execution_time: Optional[Any] = None,
        http_post: Optional[Callable] = None,
        cash_balance: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """Dispatch institutional blue embed card (0x1E88E5) on rebalance execution."""
        url = dashboard_url or self.dashboard_url
        effective_cash = cash if cash is not None else cash_balance
        card = build_trade_execution_card(
            orders=orders,
            nav=nav,
            cash=effective_cash,
            regime=regime,
            execution_time=execution_time,
            dashboard_url=url,
        )
        return self._prepare_and_post(card, incident_key=f"trade:{regime}", http_post=http_post)

    # ------------------------------------------------------------------------
    # Asynchronous Coroutine Counterparts (for async event loops)
    # ------------------------------------------------------------------------

    async def async_post_broken_alert(self, *args, **kwargs) -> bool:
        """Asynchronous wrapper for post_broken_alert."""
        return await asyncio.to_thread(self.post_broken_alert, *args, **kwargs)

    async def async_post_recovered_alert(self, *args, **kwargs) -> bool:
        """Asynchronous wrapper for post_recovered_alert."""
        return await asyncio.to_thread(self.post_recovered_alert, *args, **kwargs)

    async def async_post_trade_execution(self, *args, **kwargs) -> bool:
        """Asynchronous wrapper for post_trade_execution."""
        return await asyncio.to_thread(self.post_trade_execution, *args, **kwargs)
