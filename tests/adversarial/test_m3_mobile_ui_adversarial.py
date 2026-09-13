"""
tests.adversarial.test_m3_mobile_ui_adversarial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Milestone 3 Empirical Adversarial Test Suite:
Validates mobile ergonomics, touch targets, alert banner dynamics,
and modern 'Light and Airy' design palette contracts.

Requirements Tested:
- ORIGINAL_REQUEST.md R2: Mobile-centric 375px-430px ergonomics, light & airy palette,
  >= 44px tap targets, prominent alert banner on disconnect.
- PROJECT.md: Features 10, 11, 12, 13, 14.
- Target Files: web/templates/dashboard.html, web/static/css/custom.css, web/static/js/dashboard.js
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import bs4
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
import jinja2
import pytest

from bot.feed_manager import ConnectionStatus, FeedSource
from bot.paper_account import PortfolioSummary, PositionDetail
from bot.service import DynamicStrategyService, ServiceConfig, ServiceState, ServiceStatus
from web.app import create_app

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATE_PATH = BASE_DIR / "web" / "templates" / "dashboard.html"
CSS_PATH = BASE_DIR / "web" / "static" / "css" / "custom.css"
JS_PATH = BASE_DIR / "web" / "static" / "js" / "dashboard.js"


@pytest.fixture(autouse=True)
def ensure_event_loop():
    """Ensure a healthy asyncio event loop exists on the current thread for Python 3.9."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield


@pytest.fixture(scope="module")
def raw_html() -> str:
    """Read raw template HTML."""
    assert TEMPLATE_PATH.exists(), f"Missing template at {TEMPLATE_PATH}"
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def raw_css() -> str:
    """Read supplementary custom CSS."""
    assert CSS_PATH.exists(), f"Missing CSS at {CSS_PATH}"
    return CSS_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def raw_js() -> str:
    """Read client-side JavaScript."""
    assert JS_PATH.exists(), f"Missing JS at {JS_PATH}"
    return JS_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def soup(raw_html: str) -> BeautifulSoup:
    """Parsed BeautifulSoup DOM of raw dashboard template."""
    return BeautifulSoup(raw_html, "html.parser")


# ==============================================================================
# Helper Functions for Mathematical Ergonomics & WCAG Color Contrast
# ==============================================================================

def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert hex string (e.g., '#f8fafc') to RGB tuple."""
    hex_clean = hex_str.lstrip("#")
    if len(hex_clean) == 3:
        hex_clean = "".join([c * 2 for c in hex_clean])
    return int(hex_clean[0:2], 16), int(hex_clean[2:4], 16), int(hex_clean[4:6], 16)


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Calculate WCAG 2.1 relative luminance for an sRGB tuple."""
    res = []
    for c in rgb:
        val = c / 255.0
        if val <= 0.03928:
            res.append(val / 12.92)
        else:
            res.append(((val + 0.055) / 1.055) ** 2.4)
    return 0.2126 * res[0] + 0.7152 * res[1] + 0.0722 * res[2]


def contrast_ratio(hex1: str, hex2: str) -> float:
    """Calculate WCAG 2.1 contrast ratio between two hex colors."""
    lum1 = relative_luminance(hex_to_rgb(hex1))
    lum2 = relative_luminance(hex_to_rgb(hex2))
    l_max = max(lum1, lum2)
    l_min = min(lum1, lum2)
    return (l_max + 0.05) / (l_min + 0.05)


# ==============================================================================
# 1. Mobile Viewport (375px–430px) Constraint & Overflow Audit
# ==============================================================================

class TestMobileViewportConstraintsAdversarial:
    """
    Empirical tests validating horizontal scroll containment and viewport constraints
    across target mobile dimensions (375px iPhone SE to 430px iPhone Pro Max).
    """

    def test_viewport_meta_tag_mobile_scaling(self, soup: BeautifulSoup):
        """Assert viewport meta tag strictly sets width=device-width, initial-scale=1.0, and cover."""
        meta_tags = soup.find_all("meta", attrs={"name": "viewport"})
        assert len(meta_tags) >= 1, "Missing <meta name='viewport'> tag"

        content = meta_tags[0].get("content", "").lower()
        assert "width=device-width" in content, f"Missing width=device-width: {content}"
        assert "initial-scale=1" in content, f"Missing initial-scale=1: {content}"
        assert "viewport-fit=cover" in content, f"Missing viewport-fit=cover for iOS safe areas: {content}"

        # Hardcoded desktop viewports are forbidden
        for desktop_dim in ("1024", "1280", "1440", "1920", "960"):
            assert f"width={desktop_dim}" not in content, (
                f"Desktop fixed width width={desktop_dim} detected in viewport tag"
            )

    def test_horizontal_scroll_containment_enforced_in_css_and_body(
        self, raw_html: str, raw_css: str, soup: BeautifulSoup
    ):
        """Assert overflow-x: hidden !important on html/body and overflow-x-hidden on body element."""
        # 1. CSS Rule Inspection
        assert "overflow-x: hidden !important" in raw_css, (
            "custom.css must enforce 'overflow-x: hidden !important' on html, body"
        )
        assert "width: 100%" in raw_css or "max-width: 100%" in raw_css, (
            "custom.css must contain 100% width bounding"
        )

        # 2. Template Body Class Inspection
        body = soup.find("body")
        assert body is not None, "Missing <body> in dashboard.html"
        body_classes = body.get("class", [])
        assert "overflow-x-hidden" in body_classes, (
            f"body tag must include 'overflow-x-hidden' class, got: {body_classes}"
        )

    def test_no_fixed_pixel_width_container_exceeding_375px_in_html(self, soup: BeautifulSoup):
        """
        Adversarially parse all DOM elements in dashboard.html.
        Assert no container has an uncontained fixed width > 375px (e.g. style='width: 400px'
        or w-[400px] without responsive max-width rules).
        """
        fixed_width_style_pattern = re.compile(r"(?:^|;)\s*(?:min-)?width\s*:\s*(\d+)px", re.IGNORECASE)
        tailwind_arbitrary_width_pattern = re.compile(r"\b(?:min-)?w-\[(\d+)px\]\b")
        tailwind_fixed_classes = {
            "w-96": 384,
            "w-80": 320,
            "w-72": 288,
        }

        all_elements = soup.find_all(True)
        for elem in all_elements:
            classes = elem.get("class", [])
            class_str = " ".join(classes) if isinstance(classes, list) else str(classes)
            style = elem.get("style", "")

            # Check inline styles
            style_matches = fixed_width_style_pattern.findall(style)
            for width_px_str in style_matches:
                width_px = int(width_px_str)
                assert width_px <= 375, (
                    f"Element <{elem.name} class='{class_str}'> has fixed inline width {width_px}px > 375px!"
                )

            # Check Tailwind arbitrary pixel widths
            tw_matches = tailwind_arbitrary_width_pattern.findall(class_str)
            for width_px_str in tw_matches:
                width_px = int(width_px_str)
                has_responsive_bounds = "max-w-" in class_str or "w-full" in class_str or "sm:" in class_str
                assert width_px <= 375 or has_responsive_bounds, (
                    f"Element <{elem.name}> specifies w-[{width_px}px] > 375px without responsive bounds!"
                )

            # Check standard fixed Tailwind classes that exceed 375px
            if "w-96" in class_str:
                has_responsive = "max-w-" in class_str or "w-full" in class_str or "sm:" in class_str
                assert has_responsive, (
                    f"Element <{elem.name}> uses fixed 'w-96' (384px) exceeding 375px on mobile!"
                )

    def test_no_fixed_pixel_width_exceeding_375px_in_custom_css(self, raw_css: str):
        """Parse custom.css to verify no stylesheet rules define width or min-width > 375px."""
        width_rule_pattern = re.compile(r"\b(?:min-)?width\s*:\s*(\d+)px\b", re.IGNORECASE)
        matches = width_rule_pattern.findall(raw_css)
        for px_val in matches:
            val = int(px_val)
            assert val <= 375, (
                f"custom.css specifies fixed width of {val}px > 375px! Violates mobile containment."
            )

    def test_main_column_container_responsive_bounds(self, soup: BeautifulSoup):
        """Assert main container enforces w-full, responsive max-w-lg (or max-w-md), and centering."""
        main_div = soup.find("div", class_=lambda c: c and "mx-auto" in c and "w-full" in c)
        assert main_div is not None, "Missing centered 'w-full mx-auto' main column container"
        classes = main_div.get("class", [])
        assert any(c.startswith("max-w-") for c in classes), (
            f"Main container lacks responsive max-w constraint: {classes}"
        )

    def test_positions_table_horizontal_scroll_containment(self, soup: BeautifulSoup):
        """
        Assert desktop tabular view is hidden on mobile screens (< 640px) or enclosed
        in overflow-x-auto, while mobile card stack (#positions-mobile-list) is present.
        """
        # 1. Desktop table container check
        table = soup.find("table", id="positions-table")
        assert table is not None, "Missing #positions-table"

        parent = table.parent
        parent_classes = parent.get("class", []) if parent else []
        is_hidden_on_mobile = "hidden" in parent_classes and "sm:block" in parent_classes
        has_overflow_containment = "overflow-x-auto" in parent_classes
        assert is_hidden_on_mobile or has_overflow_containment, (
            f"Positions table wrapper must have 'hidden sm:block' or 'overflow-x-auto', got: {parent_classes}"
        )

        # 2. Mobile card stack check
        mobile_list = soup.find(id="positions-mobile-list")
        assert mobile_list is not None, "Missing #positions-mobile-list for mobile viewport"
        ml_classes = mobile_list.get("class", [])
        assert "sm:hidden" in ml_classes, (
            f"#positions-mobile-list must be hidden on desktop (sm:hidden), got: {ml_classes}"
        )

    def test_mathematical_operator_grid_clearance_at_375px(self):
        """
        Mathematically verify 3-column operator control grid fits on 375px screen
        without horizontal clipping or touch-target violation.
        Screen: 375px
        Padding: p-4 (16px * 2 = 32px)
        Available width: 343px
        Gap: gap-2.5 (10px * 2 = 20px)
        Width per button: (343 - 20) / 3 = 107.67px >= 44px
        """
        screen_width = 375.0
        padding = 32.0  # p-4
        gaps = 10.0 * 2  # gap-2.5 across 3 columns
        available_width = screen_width - padding
        button_width = (available_width - gaps) / 3.0

        assert button_width >= 44.0, (
            f"Computed button width {button_width:.2f}px violates 44px touch target minimum!"
        )
        assert (button_width * 3 + gaps + padding) <= screen_width, (
            "Computed 3-button grid overflows 375px viewport!"
        )


# ==============================================================================
# 2. Touch Target Audit (Minimum 44px Height & Width)
# ==============================================================================

class TestTouchTargetAuditAdversarial:
    """
    Adversarial audit verifying that EVERY button, link, and interactive control
    satisfies Apple Human Interface Guidelines and WCAG minimum 44px tap targets.
    """

    def test_all_buttons_satisfy_min_44px_height_and_width(self, soup: BeautifulSoup, raw_css: str):
        """Assert every single <button> in dashboard.html specifies min-h-[44px] or touch-btn."""
        buttons = soup.find_all("button")
        assert len(buttons) >= 5, f"Expected at least 5 buttons in template, found {len(buttons)}"

        # Check CSS definition of .touch-btn
        touch_btn_rule = re.search(r"\.touch-btn\s*\{([^}]+)\}", raw_css)
        assert touch_btn_rule is not None, "Missing .touch-btn selector in custom.css"
        css_body = touch_btn_rule.group(1)
        assert "min-height: 44px" in css_body, ".touch-btn missing 'min-height: 44px;'"
        assert "min-width: 44px" in css_body, ".touch-btn missing 'min-width: 44px;'"

        for btn in buttons:
            btn_id = btn.get("id", btn.get("onclick", "unknown_button"))
            classes = btn.get("class", [])
            style = btn.get("style", "")

            has_tailwind_height = any(
                c in classes for c in ("min-h-[44px]", "h-11", "h-12", "h-14", "py-3", "py-3.5")
            )
            has_touch_btn_class = "touch-btn" in classes
            has_inline_height = "min-height: 44px" in style or "height: 44px" in style

            assert (has_tailwind_height or has_touch_btn_class or has_inline_height), (
                f"Button '{btn_id}' fails >= 44px height requirement! Classes: {classes}"
            )

    def test_operator_controls_specific_targets(self, soup: BeautifulSoup):
        """Assert #btn-pause, #btn-resume, and #btn-rebalance meet 44px target requirements."""
        for target_id in ("btn-pause", "btn-resume", "btn-rebalance"):
            btn = soup.find(id=target_id)
            assert btn is not None, f"Required operator button #{target_id} not found in DOM"
            classes = btn.get("class", [])
            assert "touch-btn" in classes, f"#{target_id} missing 'touch-btn' class"
            assert "min-h-[44px]" in classes, f"#{target_id} missing 'min-h-[44px]' class"

    def test_modal_buttons_meet_touch_target_requirements(self, soup: BeautifulSoup):
        """Assert rebalance confirmation modal buttons satisfy 44px target requirements."""
        modal = soup.find(id="rebalance-modal")
        assert modal is not None, "Missing #rebalance-modal in DOM"

        modal_buttons = modal.find_all("button")
        assert len(modal_buttons) >= 2, f"Expected 2 modal action buttons, found {len(modal_buttons)}"

        for btn in modal_buttons:
            classes = btn.get("class", [])
            assert "min-h-[44px]" in classes, f"Modal button missing min-h-[44px]: {classes}"
            assert "touch-btn" in classes, f"Modal button missing touch-btn: {classes}"

    def test_touch_manipulation_prevents_ios_double_tap_delay(self, soup: BeautifulSoup, raw_css: str):
        """Assert touch-action: manipulation is configured in CSS and on interactive controls."""
        assert "touch-action: manipulation" in raw_css, (
            "custom.css must set 'touch-action: manipulation' to prevent 300ms iOS tap delay"
        )
        assert "-webkit-tap-highlight-color: transparent" in raw_css, (
            "custom.css must set '-webkit-tap-highlight-color: transparent'"
        )

        # Operator controls should have touch-manipulation utility
        for target_id in ("btn-pause", "btn-resume", "btn-rebalance"):
            btn = soup.find(id=target_id)
            classes = btn.get("class", [])
            assert "touch-manipulation" in classes, (
                f"Button #{target_id} missing 'touch-manipulation' utility"
            )

    def test_inter_button_clearance_spacing(self, soup: BeautifulSoup):
        """Assert operator button group specifies adequate gap (>= 8px / gap-2.5) to prevent mis-taps."""
        controls_section = soup.find("section", string=re.compile(r"Operator Controls", re.IGNORECASE))
        if controls_section is None:
            # Look for h2 header
            h2 = soup.find("h2", string=re.compile(r"Operator Controls", re.IGNORECASE))
            assert h2 is not None, "Missing 'Operator Controls' heading"
            controls_section = h2.parent

        grid = controls_section.find("div", class_=lambda c: c and "grid" in c)
        assert grid is not None, "Missing grid wrapper for operator controls"
        classes = grid.get("class", [])
        has_valid_gap = any(c in classes for c in ("gap-2", "gap-2.5", "gap-3", "gap-4", "gap-x-3"))
        assert has_valid_gap, f"Operator button grid lacks spacing gap (min 8px/gap-2): {classes}"


# ==============================================================================
# 3. Alert Banner Dynamic Binding Audit
# ==============================================================================

class TestAlertBannerDynamicBindingAdversarial:
    """
    Adversarial verification of top alert banner (#alert-banner):
    - Jinja2 conditional rendering when alert_banner_active is True vs False.
    - Warning color tokens (amber/rose) and mandatory text copy.
    - FastAPI route integration and dynamic DOM manipulation in dashboard.js.
    """

    def test_jinja_render_alert_banner_active_true(self, raw_html: str):
        """Assert alert banner renders visibly with warning styles when alert_banner_active=True."""
        template = jinja2.Template(raw_html)
        ctx = {
            "alert_banner_active": True,
            "daemon_state": "RUNNING",
            "unrealized_pnl": 0.0,
            "portfolio": None,
            "regime": "BULL_NORMAL",
        }
        rendered = template.render(**ctx)
        soup_rendered = BeautifulSoup(rendered, "html.parser")

        banner = soup_rendered.find(id="alert-banner")
        assert banner is not None, "Missing #alert-banner in rendered DOM"
        classes = banner.get("class", [])

        # Must NOT be hidden
        assert "hidden" not in classes, f"Alert banner must not be hidden when active! Classes: {classes}"

        # Must have prominent warning colors (amber or rose)
        class_str = " ".join(classes)
        has_warning_color = any(w in class_str for w in ("bg-amber-", "bg-rose-", "bg-yellow-"))
        assert has_warning_color, f"Alert banner missing prominent warning styling: {class_str}"

        # Must contain prominent disconnect copy
        text = banner.get_text()
        assert "FEED DISCONNECTED" in text or "AlpacaRelay Disconnected" in text, (
            f"Alert banner missing required disconnect warning copy: {text}"
        )
        assert "⚠️" in text, "Alert banner missing ⚠️ emoji icon"

    def test_jinja_render_alert_banner_active_false(self, raw_html: str):
        """Assert alert banner renders with 'hidden' class when alert_banner_active=False."""
        template = jinja2.Template(raw_html)
        ctx = {
            "alert_banner_active": False,
            "daemon_state": "RUNNING",
            "unrealized_pnl": 0.0,
            "portfolio": None,
            "regime": "BULL_NORMAL",
        }
        rendered = template.render(**ctx)
        soup_rendered = BeautifulSoup(rendered, "html.parser")

        banner = soup_rendered.find(id="alert-banner")
        assert banner is not None, "Missing #alert-banner in rendered DOM"
        classes = banner.get("class", [])

        # Must be hidden
        assert "hidden" in classes, f"Alert banner MUST have 'hidden' class when inactive! Classes: {classes}"

    def test_jinja_render_alert_banner_falsy_defaults(self, raw_html: str):
        """Assert alert banner safely defaults to 'hidden' when alert_banner_active is None or absent."""
        template = jinja2.Template(raw_html)
        rendered_none = template.render(alert_banner_active=None)
        soup_none = BeautifulSoup(rendered_none, "html.parser")
        banner_none = soup_none.find(id="alert-banner")
        assert "hidden" in banner_none.get("class", []), "Banner should hide when alert_banner_active=None"

        rendered_empty = template.render()
        soup_empty = BeautifulSoup(rendered_empty, "html.parser")
        banner_empty = soup_empty.find(id="alert-banner")
        assert "hidden" in banner_empty.get("class", []), "Banner should hide when context is empty"

    def test_fastapi_rendered_dashboard_reflects_active_banner_state(self):
        """
        FastAPI integration test: Verify that when FeedManager reports active alert,
        GET / renders dashboard without the 'hidden' class on #alert-banner.
        """
        mock_service = MagicMock()
        mock_service.get_service_status.return_value = ServiceStatus(
            state=ServiceState.RUNNING,
            is_running=True,
            is_paused=False,
            current_regime="BULL_NORMAL",
            total_nav=50000.0,
            cash=50000.0,
            equity=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            active_positions_count=0,
            feed_source="synthetic_fallback",
            alert_banner_active=True,
            last_rebalance_timestamp=None,
            uptime_seconds=100.0,
        )
        mock_conn = ConnectionStatus(
            is_connected=False,
            feed_source="synthetic_fallback",
            alert_banner_active=True,
            last_heartbeat_timestamp=datetime.now(timezone.utc).isoformat(),
            consecutive_reconnect_attempts=3,
            status_message="Upstream disconnected, running on synthetic fallback.",
        )
        mock_service.feed_manager.get_connection_status.return_value = mock_conn
        mock_service.feed_manager.get_latest_prices.return_value = {}
        mock_service.paper_account.get_portfolio_state.return_value = PortfolioSummary(
            cash=50000.0, equity=0.0, total_nav=50000.0, realized_pnl=0.0, unrealized_pnl=0.0, positions=[]
        )
        mock_service.service_config.symbols = ["SPY", "QQQ"]

        app = create_app(service=mock_service)
        with TestClient(app) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            soup_res = BeautifulSoup(resp.text, "html.parser")
            banner = soup_res.find(id="alert-banner")
            assert banner is not None, "Missing #alert-banner"
            classes = banner.get("class", [])
            assert "hidden" not in classes, f"Expected active banner to NOT have 'hidden', got: {classes}"
            assert "bg-amber-50" in classes or any("bg-amber-" in c for c in classes)

    def test_fastapi_rendered_dashboard_reflects_inactive_banner_state(self):
        """
        FastAPI integration test: Verify that when FeedManager reports normal feed,
        GET / renders dashboard with the 'hidden' class on #alert-banner.
        """
        mock_service = MagicMock()
        mock_service.get_service_status.return_value = ServiceStatus(
            state=ServiceState.RUNNING,
            is_running=True,
            is_paused=False,
            current_regime="BULL_NORMAL",
            total_nav=50000.0,
            cash=50000.0,
            equity=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            active_positions_count=0,
            feed_source="alpaca_relay",
            alert_banner_active=False,
            last_rebalance_timestamp=None,
            uptime_seconds=100.0,
        )
        mock_conn = ConnectionStatus(
            is_connected=True,
            feed_source="alpaca_relay",
            alert_banner_active=False,
            last_heartbeat_timestamp=datetime.now(timezone.utc).isoformat(),
            consecutive_reconnect_attempts=0,
            status_message="Connected to AlpacaRelay stream.",
        )
        mock_service.feed_manager.get_connection_status.return_value = mock_conn
        mock_service.feed_manager.get_latest_prices.return_value = {}
        mock_service.paper_account.get_portfolio_state.return_value = PortfolioSummary(
            cash=50000.0, equity=0.0, total_nav=50000.0, realized_pnl=0.0, unrealized_pnl=0.0, positions=[]
        )
        mock_service.service_config.symbols = ["SPY", "QQQ"]

        app = create_app(service=mock_service)
        with TestClient(app) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            soup_res = BeautifulSoup(resp.text, "html.parser")
            banner = soup_res.find(id="alert-banner")
            assert banner is not None, "Missing #alert-banner"
            classes = banner.get("class", [])
            assert "hidden" in classes, f"Expected inactive banner to have 'hidden', got: {classes}"

    def test_javascript_banner_toggle_contract(self, raw_js: str):
        """Inspect web/static/js/dashboard.js to verify client-side reactive banner toggling logic."""
        # Must locate the banner element
        assert "document.getElementById('alert-banner')" in raw_js, (
            "dashboard.js must bind to 'alert-banner' element"
        )
        # Must check feed_source or alert_banner_active
        assert "alert_banner_active" in raw_js or "feed_source" in raw_js, (
            "dashboard.js must check alert_banner_active or feed_source"
        )
        # Must toggle hidden class
        assert "banner.classList.remove('hidden')" in raw_js, (
            "dashboard.js must call remove('hidden') to show alert banner"
        )
        assert "banner.classList.add('hidden')" in raw_js, (
            "dashboard.js must call add('hidden') to hide alert banner"
        )


# ==============================================================================
# 4. "Light and Airy" Palette & Typography Audit
# ==============================================================================

class TestLightAndAiryPaletteAuditAdversarial:
    """
    Empirical audit verifying modern 'Light & Airy' theme:
    - Soft slate background (#f8fafc / bg-slate-50).
    - Crisp white cards (#ffffff / bg-white).
    - Slate typography (#0f172a / text-slate-900).
    - Emerald accents (#10b981).
    - WCAG AA / AAA color contrast validation.
    """

    def test_soft_slate_background_tokens(self, raw_html: str, soup: BeautifulSoup):
        """Assert background uses soft slate-50 (#f8fafc)."""
        html_tag = soup.find("html")
        assert html_tag is not None
        assert "bg-slate-50" in html_tag.get("class", []), "<html> missing 'bg-slate-50'"

        body_tag = soup.find("body")
        assert body_tag is not None
        assert "bg-slate-50" in body_tag.get("class", []), "<body> missing 'bg-slate-50'"

        # Verify Tailwind configuration defines slate.50 as '#f8fafc'
        assert "'#f8fafc'" in raw_html or '"#f8fafc"' in raw_html, (
            "Tailwind config in dashboard.html must map slate.50 to #f8fafc"
        )

    def test_pure_white_card_surfaces_and_borders(self, soup: BeautifulSoup):
        """Assert all operational card sections utilize bg-white with subtle borders."""
        sections = soup.find_all("section")
        assert len(sections) >= 5, f"Expected at least 5 card sections, found {len(sections)}"

        for i, sec in enumerate(sections):
            classes = sec.get("class", [])
            assert "bg-white" in classes, f"Section {i} lacks 'bg-white' card background: {classes}"
            assert any(c.startswith("rounded-") for c in classes), f"Section {i} lacks rounded corners: {classes}"
            assert any(c.startswith("border") for c in classes), f"Section {i} lacks clean border: {classes}"

    def test_high_contrast_slate_typography(self, raw_html: str, soup: BeautifulSoup):
        """Assert primary typography uses crisp dark slate (#0f172a / text-slate-900)."""
        # 1. Verify Tailwind config defines slate.900 as #0f172a
        assert "'#0f172a'" in raw_html or '"#0f172a"' in raw_html, (
            "Tailwind config must define slate.900 as #0f172a"
        )

        # 2. Main Title H1 uses text-slate-900
        h1 = soup.find("h1")
        assert h1 is not None, "Missing <h1> title"
        assert "text-slate-900" in h1.get("class", []), f"<h1> lacks text-slate-900: {h1.get('class')}"

        # 3. Hero NAV metric uses text-slate-900
        nav_el = soup.find(id="portfolio-nav")
        assert nav_el is not None, "Missing #portfolio-nav"
        assert "text-slate-900" in nav_el.get("class", []), f"#portfolio-nav lacks text-slate-900: {nav_el.get('class')}"

        # 4. Rebalance modal heading uses text-slate-900
        modal = soup.find(id="rebalance-modal")
        assert modal is not None, "Missing #rebalance-modal"
        modal_h3 = modal.find("h3")
        assert modal_h3 is not None, "Missing modal <h3>"
        assert "text-slate-900" in modal_h3.get("class", []), f"Modal <h3> lacks text-slate-900: {modal_h3.get('class')}"

    def test_emerald_semantic_accents(self, raw_html: str, soup: BeautifulSoup):
        """Assert emerald accent color #10b981 is defined and used for positive metrics/status."""
        assert "'#10b981'" in raw_html or '"#10b981"' in raw_html, (
            "Tailwind config must define emerald.500 as #10b981"
        )

        # Asset split bar and SSE live indicator must utilize emerald
        split_bar = soup.find(id="bar-equity")
        assert split_bar is not None, "Missing #bar-equity"
        assert "bg-emerald-500" in split_bar.get("class", []), "#bar-equity missing 'bg-emerald-500'"

        sse_dot = soup.find(id="sse-dot")
        assert sse_dot is not None, "Missing #sse-dot"
        assert "bg-emerald-500" in sse_dot.get("class", []), "#sse-dot missing 'bg-emerald-500'"

    def test_wcag_color_contrast_ratios_aa_and_aaa(self):
        """
        Empirically calculate WCAG 2.1 contrast ratios for key dashboard color pairings:
        - Slate 900 (#0f172a) on White (#ffffff): expect >= 7.0 (AAA)
        - Slate 800 (#1e293b) on White (#ffffff): expect >= 7.0 (AAA)
        - Slate 500 (#64748b) on White (#ffffff): expect >= 4.5 (AA)
        - Amber 900 (#78350f) on Amber 50 (#fffbeb): expect >= 4.5 (AA)
        - Emerald 700 (#047857) on Emerald 50 (#ecfdf5): expect >= 4.5 (AA)
        - Rose 700 (#be123c) on Rose 50 (#fff1f2): expect >= 4.5 (AA)
        """
        # 1. Primary Text on White
        cr_primary = contrast_ratio("#0f172a", "#ffffff")
        assert cr_primary >= 7.0, f"Slate 900 on White contrast {cr_primary:.2f} < 7.0 (AAA failed)"

        # 2. Body Text on White
        cr_body = contrast_ratio("#1e293b", "#ffffff")
        assert cr_body >= 7.0, f"Slate 800 on White contrast {cr_body:.2f} < 7.0 (AAA failed)"

        # 3. Secondary Muted Text on White
        cr_secondary = contrast_ratio("#64748b", "#ffffff")
        assert cr_secondary >= 4.5, f"Slate 500 on White contrast {cr_secondary:.2f} < 4.5 (AA failed)"

        # 4. Amber Alert Banner Text on Amber 50
        cr_amber = contrast_ratio("#78350f", "#fffbeb")
        assert cr_amber >= 4.5, f"Amber 900 on Amber 50 contrast {cr_amber:.2f} < 4.5 (AA failed)"

        # 5. Emerald Gain Badge Text on Emerald 50
        cr_emerald = contrast_ratio("#047857", "#ecfdf5")
        assert cr_emerald >= 4.5, f"Emerald 700 on Emerald 50 contrast {cr_emerald:.2f} < 4.5 (AA failed)"

        # 6. Rose Loss Badge Text on Rose 50
        cr_rose = contrast_ratio("#be123c", "#fff1f2")
        assert cr_rose >= 4.5, f"Rose 700 on Rose 50 contrast {cr_rose:.2f} < 4.5 (AA failed)"


# ==============================================================================
# 5. Mobile Edge Cases & Adversarial Stress Scenarios
# ==============================================================================

class TestMobileEdgeCasesAdversarial:
    """
    Stress-tests edge cases:
    - Extreme NAV/PnL numbers (layout overflow prevention).
    - Large positions portfolio stress.
    - Zero positions clean pristine state.
    - ARIA accessibility and semantic role attributes.
    """

    def test_extreme_financial_numbers_rendering(self, raw_html: str):
        """Stress-test template rendering with extreme NAV ($999,999,999.99) and negative PnL."""
        template = jinja2.Template(raw_html)
        huge_portfolio = PortfolioSummary(
            cash=899999999.99,
            equity=100000000.00,
            total_nav=999999999.99,
            realized_pnl=12345678.90,
            unrealized_pnl=-5432109.87,
            positions=[],
        )
        rendered = template.render(
            portfolio=huge_portfolio,
            unrealized_pnl=-5432109.87,
            alert_banner_active=False,
            daemon_state="RUNNING",
            regime="BULL_AGGRESSIVE",
        )
        soup_res = BeautifulSoup(rendered, "html.parser")

        nav_el = soup_res.find(id="portfolio-nav")
        assert nav_el is not None
        assert "999,999,999.99" in nav_el.text

        pnl_el = soup_res.find(id="today-pnl-badge")
        assert pnl_el is not None
        assert "-5432109.87" in pnl_el.text
        # Must reflect rose styling on loss
        assert "bg-rose-50" in pnl_el.get("class", [])

    def test_pristine_initial_state_ergonomics(self, raw_html: str):
        """Verify clean initial $50,000.00 paper balance presentation with zero positions."""
        template = jinja2.Template(raw_html)
        pristine_portfolio = PortfolioSummary(
            cash=50000.00,
            equity=0.00,
            total_nav=50000.00,
            realized_pnl=0.00,
            unrealized_pnl=0.00,
            positions=[],
        )
        rendered = template.render(
            portfolio=pristine_portfolio,
            unrealized_pnl=0.0,
            alert_banner_active=False,
            daemon_state="RUNNING",
            regime="BULL_NORMAL",
        )
        soup_res = BeautifulSoup(rendered, "html.parser")

        # Check empty state message exists
        empty_mobile = soup_res.find(id="positions-empty-mobile")
        assert empty_mobile is not None, "Missing #positions-empty-mobile in DOM"
        assert "$50,000.00 pristine cash balance" in empty_mobile.text

    def test_aria_accessibility_and_semantic_roles(self, soup: BeautifulSoup):
        """Assert accessible semantic roles: role='alert' on banner, aria-hidden on decorative emojis."""
        banner = soup.find(id="alert-banner")
        assert banner is not None
        assert banner.get("role") == "alert", "Top alert banner must have role='alert'"

        # Emojis should have aria-hidden='true'
        warning_icon = banner.find("span", string="⚠️")
        assert warning_icon is not None
        assert warning_icon.get("aria-hidden") == "true", "Warning emoji should have aria-hidden='true'"

        # Rebalance modal dialog accessibility
        modal = soup.find(id="rebalance-modal")
        assert modal is not None
        assert "z-50" in modal.get("class", []), "Modal dialog must have z-50 for viewport overlay isolation"

    def test_in_flight_button_loading_state_preserves_44px_height(self, raw_css: str, soup: BeautifulSoup):
        """
        Adversarial test: Verify that when JS mutates button innerHTML during in-flight actions
        (e.g. '⏳ Pausing...'), the button element cannot collapse below 44px min-height/width
        because min-height: 44px is enforced on the container class (.touch-btn).
        """
        # 1. Check custom.css defines min-height and min-width on .touch-btn
        touch_btn_rule = re.search(r"\.touch-btn\s*\{([^}]+)\}", raw_css)
        assert touch_btn_rule is not None
        css_body = touch_btn_rule.group(1)
        assert "min-height: 44px" in css_body
        assert "min-width: 44px" in css_body

        # 2. Check all operator buttons have .touch-btn
        for btn_id in ("btn-pause", "btn-resume", "btn-rebalance"):
            btn = soup.find(id=btn_id)
            assert "touch-btn" in btn.get("class", [])
            assert "min-h-[44px]" in btn.get("class", [])

    def test_large_positions_portfolio_mobile_stack_stress(self, raw_html: str):
        """Stress-test rendering with 25 diverse positions to verify mobile DOM stability."""
        template = jinja2.Template(raw_html)
        positions = [
            PositionDetail(
                symbol=f"TICK{i:02d}",
                qty=float(i * 10 + 5),
                avg_entry_price=100.0 + i,
                current_price=105.0 + i,
                cost_basis=float((i * 10 + 5) * (100.0 + i)),
                market_value=float((i * 10 + 5) * (105.0 + i)),
                unrealized_pnl=float((i * 10 + 5) * 5.0),
                weight=0.04,
            )
            for i in range(25)
        ]
        total_equity = sum(p.market_value for p in positions)
        total_nav = 10000.0 + total_equity
        portfolio = PortfolioSummary(
            cash=10000.0,
            equity=total_equity,
            total_nav=total_nav,
            realized_pnl=500.0,
            unrealized_pnl=sum(p.unrealized_pnl for p in positions),
            positions=positions,
        )
        rendered = template.render(
            portfolio=portfolio,
            unrealized_pnl=portfolio.unrealized_pnl,
            alert_banner_active=False,
            daemon_state="RUNNING",
            regime="BULL_NORMAL",
        )
        soup_res = BeautifulSoup(rendered, "html.parser")
        # Holdings count badge should render 25 Open
        badge = soup_res.find(id="holdings-badge")
        assert badge is not None
        assert "25 Open" in badge.text

    def test_eyebrow_headings_contrast_assessment(self):
        """
        Empirical evaluation of section label typography:
        Evaluates slate-400 (#94a3b8) vs slate-500 (#64748b) on white surfaces.
        Documents finding that slate-400 provides 2.56:1 contrast (below WCAG AA 4.5:1),
        while slate-500 achieves 4.76:1 (fully WCAG AA compliant).
        """
        slate_400_ratio = contrast_ratio("#94a3b8", "#ffffff")
        slate_500_ratio = contrast_ratio("#64748b", "#ffffff")

        assert slate_400_ratio < 4.5, "slate-400 unexpectedly exceeds 4.5:1"
        assert slate_500_ratio >= 4.5, "slate-500 failed WCAG AA 4.5:1 requirement"
        # Confirm slate-500 provides an 85%+ improvement in contrast ratio
        assert (slate_500_ratio / slate_400_ratio) >= 1.85

    def test_alert_banner_truth_table_in_javascript(self, raw_js: str):
        """
        Verify the JavaScript boolean condition for showing the alert banner:
        data.feed_source === 'synthetic_fallback' || data.alert_banner_active === true
        """
        condition_pattern = re.compile(
            r"feed_source\s*===\s*['\"]synthetic_fallback['\"]\s*\|\|\s*.*alert_banner_active\s*===\s*true"
        )
        assert condition_pattern.search(raw_js) is not None, (
            "dashboard.js must use compound boolean check (synthetic_fallback OR alert_banner_active === true)"
        )

