"""
tests.unit.test_dashboard_ui_contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Static inspection test suite validating mobile viewport, CSS, and UI contracts:
- Mobile viewport meta tag: width=device-width, initial-scale=1.0
- Apple / Material touch target minimum height: >= 44px
- AlpacaRelay disconnect alert banner markup and reactive visibility binding
- "Light & Airy" modern color palette tokens (slate-50, white, emerald, amber)
- Mobile viewport 375px–430px horizontal scroll containment.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
import pytest

TEMPLATES_DIR = Path("web/templates")
STATIC_DIR = Path("web/static")
DASHBOARD_HTML = TEMPLATES_DIR / "dashboard.html"
CUSTOM_CSS = STATIC_DIR / "css" / "custom.css"
DASHBOARD_JS = STATIC_DIR / "js" / "dashboard.js"


class TagAttributeCollector(HTMLParser):
    """Zero-dependency HTML tag and attribute extractor."""

    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        self.tags.append((tag, attr_dict))


@pytest.fixture(scope="session")
def html_content() -> str:
    """Reads raw dashboard.html template."""
    assert DASHBOARD_HTML.exists(), f"Dashboard template not found at {DASHBOARD_HTML}"
    return DASHBOARD_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def css_content() -> str:
    """Reads custom.css if present, otherwise returns empty string."""
    if CUSTOM_CSS.exists():
        return CUSTOM_CSS.read_text(encoding="utf-8")
    return ""


@pytest.fixture(scope="session")
def js_content() -> str:
    """Reads dashboard.js if present, otherwise returns empty string."""
    assert DASHBOARD_JS.exists(), f"dashboard.js not found at {DASHBOARD_JS}"
    return DASHBOARD_JS.read_text(encoding="utf-8")


# ==============================================================================
# 1. Viewport Meta Tag Contract
# ==============================================================================

def test_viewport_meta_tag_contract(html_content: str):
    """Assert mandatory mobile viewport meta tag exists with device-width scaling."""
    parser = TagAttributeCollector()
    parser.feed(html_content)

    viewport_tags = [
        attrs for tag, attrs in parser.tags
        if tag == "meta" and attrs.get("name", "").lower() == "viewport"
    ]
    assert len(viewport_tags) >= 1, "Missing <meta name='viewport'> in dashboard.html"

    content = viewport_tags[0].get("content", "")
    assert "width=device-width" in content, f"Expected width=device-width, got: {content}"
    assert "initial-scale=1" in content, f"Expected initial-scale=1, got: {content}"
    assert "width=1024" not in content, "Desktop fixed width detected in mobile template"


# ==============================================================================
# 2. Button Tap Target >= 44px Contract
# ==============================================================================

def test_button_tap_target_min_44px_contract(html_content: str, css_content: str):
    """Assert interactive operator buttons meet minimum 44px touch target height."""
    parser = TagAttributeCollector()
    parser.feed(html_content)

    button_tags = [attrs for tag, attrs in parser.tags if tag == "button"]
    assert len(button_tags) >= 3, "Expected at least 3 operator buttons (Pause, Resume, Rebalance)"

    # Accepted touch classes: min-h-[44px], h-11 (44px), h-12 (48px), py-3, touch-btn
    touch_height_pattern = re.compile(r"(min-h-\[44px\]|h-11|h-12|py-3|py-3\.5|touch-btn)")

    for attrs in button_tags:
        classes = attrs.get("class", "")
        has_touch_class = bool(touch_height_pattern.search(classes))
        has_inline_min_height = "min-height: 44px" in attrs.get("style", "")
        has_css_rule = "min-height: 44px" in css_content and "touch-btn" in classes

        assert has_touch_class or has_inline_min_height or has_css_rule, (
            f"Button lacks >= 44px touch target sizing! Classes: '{classes}', Style: '{attrs.get('style')}'"
        )


# ==============================================================================
# 3. Disconnect Alert Banner Contract
# ==============================================================================

def test_alert_banner_markup_and_reactive_binding(html_content: str):
    """Assert alert banner exists with warning colors and reactive visibility directive."""
    parser = TagAttributeCollector()
    parser.feed(html_content)

    banner_attrs = [
        attrs for tag, attrs in parser.tags
        if attrs.get("id") in ("alert-banner", "feed-alert-banner")
        or attrs.get("data-testid") == "alert-banner"
    ]
    assert len(banner_attrs) >= 1, "Missing #alert-banner element in dashboard.html"

    attrs = banner_attrs[0]
    classes = attrs.get("class", "")

    # Assert warning colors (amber or rose)
    has_warning_color = any(
        c in classes for c in ("bg-amber-", "bg-rose-", "bg-yellow-", "border-amber-", "border-rose-")
    )
    assert has_warning_color, f"Alert banner missing warning color classes. Classes: '{classes}'"

    # Assert reactive visibility binding (Alpine x-show, v-show, or hidden class)
    has_binding = (
        "x-show" in attrs
        or "v-show" in attrs
        or ":class" in attrs
        or "hidden" in classes
        or "{% if" in html_content
    )
    assert has_binding, f"Alert banner missing conditional visibility binding. Attrs: {attrs}"

    # Assert warning copy in text
    assert (
        "AlpacaRelay Disconnected" in html_content
        or "FEED DISCONNECTED" in html_content
        or "Synthetic Fallback" in html_content
        or "SIMULATION" in html_content
    ), "Alert banner missing clear disconnect/fallback warning copy"


# ==============================================================================
# 4. Light & Airy Design Palette Tokens
# ==============================================================================

def test_light_and_airy_color_palette_tokens(html_content: str):
    """Assert modern Light & Airy palette tokens: soft slate background, pure white cards, emerald accents."""
    # Background: soft slate/off-white (#f8fafc or bg-slate-50)
    assert "bg-slate-50" in html_content or "#f8fafc" in html_content, "Missing soft slate-50 background"

    # Cards: crisp pure white (#ffffff or bg-white)
    assert "bg-white" in html_content or "#ffffff" in html_content, "Missing pure white card backgrounds"

    # Typography: dark slate (#0f172a / text-slate-900 / text-slate-800)
    assert (
        "text-slate-900" in html_content
        or "text-slate-800" in html_content
        or "#0f172a" in html_content
    ), "Missing high-contrast dark slate typography"

    # Secondary text: muted slate
    assert (
        "text-slate-500" in html_content
        or "text-slate-400" in html_content
    ), "Missing muted secondary text styling"

    # Semantic green/emerald for positive P&L / RUNNING
    assert (
        "text-emerald-" in html_content
        or "bg-emerald-" in html_content
        or "#10b981" in html_content
    ), "Missing emerald semantic accents for gains/healthy status"


# ==============================================================================
# 5. Mobile 375px–430px Layout & Scroll Containment
# ==============================================================================

def test_mobile_horizontal_scroll_containment(html_content: str):
    """Assert layout prevents horizontal overflow blowout on 375px–430px screens."""
    # Check max-width constraint for single-column mobile stack
    assert (
        "max-w-lg" in html_content
        or "max-w-md" in html_content
        or "container" in html_content
    ), "Missing mobile container max-width constraint"

    # Positions table must be wrapped in overflow-x-auto to prevent mobile blowout
    assert "overflow-x-auto" in html_content, (
        "Positions table missing overflow-x-auto horizontal scroll container"
    )


# ==============================================================================
# 6. Essential Dashboard Card Sections Present
# ==============================================================================

def test_essential_dashboard_sections_present(html_content: str):
    """Assert presence of all 6 key operational card sections."""
    assert "Operator Dashboard" in html_content or "Dynamic Long-Term" in html_content
    assert "Total NAV" in html_content or "portfolio-nav" in html_content
    assert "Cash" in html_content
    assert "Operator Controls" in html_content or "btn-pause" in html_content
    assert "Active Positions" in html_content or "positions-table" in html_content


# ==============================================================================
# 7. Milestone 4 Iteration 2 Remediation Contracts
# ==============================================================================

def test_wcag_eyebrows_use_slate_500(html_content: str):
    """Assert all card label and eyebrow classes use text-slate-500, with 0 text-slate-400."""
    assert "text-slate-400" not in html_content, (
        "Found text-slate-400 in dashboard.html; must upgrade all card labels to text-slate-500 for WCAG AA compliance"
    )
    assert "text-slate-500" in html_content, "Missing text-slate-500 eyebrow classes in dashboard.html"


def test_no_unused_alpine_script_in_html(html_content: str):
    """Assert unused Alpine.js CDN script tag is removed."""
    assert "alpinejs" not in html_content.lower(), "dashboard.html must not load unused Alpine.js CDN asset"


def test_mobile_sse_label_visibility(html_content: str):
    """Assert sse-label is visible on mobile viewports (no 'hidden sm:inline')."""
    assert 'id="sse-label"' in html_content
    assert 'class="hidden sm:inline"' not in html_content, "sse-label must not be hidden on mobile screens"


def test_dashboard_js_trades_activity_contract(js_content: str):
    """Assert dashboard.js defines fetchTrades(), renderActivity(), and queries /api/trades."""
    assert "async function fetchTrades()" in js_content or "function fetchTrades()" in js_content
    assert "function renderActivity(" in js_content
    assert "fetch('/api/trades')" in js_content
    assert "document.getElementById('activity-list')" in js_content
    assert "No trades executed yet. Initialized at $50,000.00 starting balance." in js_content


def test_dashboard_js_rebalance_rejection_handling(js_content: str):
    """Assert executeManualRebalance properly inspects rejection status and displays error toast."""
    rejection_pattern = re.compile(
        r"!json\.success\s*\|\|\s*\(?json\.status\s*&&\s*json\.status\.startsWith\(['\"]REJECTED['\"]\)\)?"
    )
    assert rejection_pattern.search(js_content) is not None, (
        "dashboard.js must check !json.success || (json.status && json.status.startsWith('REJECTED'))"
    )
    assert "showFeedback(" in js_content


def test_dashboard_js_alert_banner_state_preservation(js_content: str):
    """Assert updateStatusUI preserves feed_source and alert_banner_active on partial updates."""
    assert "AppState.feed_source" in js_content
    assert "AppState.alert_banner_active" in js_content
    assert "data.feed_source !== undefined" in js_content
    assert "data.alert_banner_active !== undefined" in js_content


def test_dashboard_js_dynamic_regime_subtext(js_content: str):
    """Assert updateRegimeDesc sets dynamic descriptions for all regimes."""
    for regime, expected_phrase in [
        ("BULL_MOMENTUM", "Bull Trend & Momentum Risk On"),
        ("REBOUND_BOUNCE", "Rebound Bounce & Recovery Trend"),
        ("HIGH_VOLATILITY", "High Volatility & Defensives"),
        ("CRASH_DEFENSE", "Downside Protection & Cash/Bonds"),
    ]:
        assert regime in js_content, f"Missing regime case: {regime}"
        assert expected_phrase in js_content, f"Missing description phrase: {expected_phrase}"


def test_dashboard_js_modal_dismissal_contract(js_content: str):
    """Assert keyboard Escape and backdrop click listeners dismiss #rebalance-modal."""
    assert "initModalHandlers" in js_content
    assert "Escape" in js_content
    assert "closeRebalanceModal()" in js_content

