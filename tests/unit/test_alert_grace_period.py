"""Feed blips shorter than the grace period must not post Discord cards.

On 2026-09-14 the Railway edge cut the relay websocket ~130 times; each cut
recovered in ~2.6s and posted a BROKEN and a RECOVERED card (345 posts/day).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from bot.service import DynamicStrategyService, ServiceConfig


class RecordingNotifier:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def post_broken_alert(self, **kwargs) -> bool:
        self.calls.append("broken")
        return True

    def post_recovered_alert(self, **kwargs) -> bool:
        self.calls.append("recovered")
        return True


def _service(tmp_path: Path, grace: float) -> tuple[DynamicStrategyService, RecordingNotifier]:
    notifier = RecordingNotifier()
    cfg = ServiceConfig(db_path=str(tmp_path / "grace.db"), alert_grace_seconds=grace)
    svc = DynamicStrategyService(config=cfg, discord_notifier=notifier)
    svc.paper_account.init_schema()
    return svc, notifier


@pytest.mark.asyncio
async def test_short_blip_posts_nothing(tmp_path: Path):
    svc, notifier = _service(tmp_path, grace=0.5)
    fm = svc.feed_manager
    await fm._transition_to_live("boot")
    await fm._transition_to_fallback("edge cut")
    await asyncio.sleep(0.05)
    await fm._transition_to_live("reconnected")
    await asyncio.sleep(0.7)  # past the grace window: the cancelled timer must stay silent
    assert notifier.calls == []


@pytest.mark.asyncio
async def test_long_outage_posts_broken_then_recovered_once(tmp_path: Path):
    svc, notifier = _service(tmp_path, grace=0.2)
    fm = svc.feed_manager
    await fm._transition_to_live("boot")
    await fm._transition_to_fallback("relay down")
    await asyncio.sleep(0.35)
    assert notifier.calls == ["broken"]
    # A second disconnect signal during the same outage must not re-post.
    await fm._transition_to_fallback("still down")
    await asyncio.sleep(0.35)
    assert notifier.calls == ["broken"]
    await fm._transition_to_live("back")
    await asyncio.sleep(0.05)
    assert notifier.calls == ["broken", "recovered"]
    # Next blip starts a fresh cycle and again stays silent if short.
    await fm._transition_to_fallback("blip")
    await asyncio.sleep(0.05)
    await fm._transition_to_live("back again")
    await asyncio.sleep(0.3)
    assert notifier.calls == ["broken", "recovered"]


@pytest.mark.asyncio
async def test_zero_grace_keeps_immediate_alerts(tmp_path: Path):
    svc, notifier = _service(tmp_path, grace=0.0)
    fm = svc.feed_manager
    await fm._transition_to_live("boot")
    await fm._transition_to_fallback("down")
    await fm._transition_to_live("up")
    assert notifier.calls == ["broken", "recovered"]
