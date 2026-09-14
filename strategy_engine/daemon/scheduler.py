"""
strategy_engine.daemon.scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

US Equity market calendar and multi-cadence event scheduler for NYSE trading hours.
Supports regular hours (09:30-16:00 ET), scheduled early closes (13:00 ET),
NYSE holiday rules, Easter Computus algorithm, and multi-cadence triggers.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union
from zoneinfo import ZoneInfo

logger = logging.getLogger("strategy_engine.daemon.scheduler")

ET = ZoneInfo("America/New_York")


def western_easter(year: int) -> date:
    """Anonymous Computus algorithm calculating Western Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


class MarketCalendar:
    """NYSE Equity Trading Calendar and Market Hours Calculator."""

    TIMEZONE = ET
    REGULAR_OPEN = time(9, 30)
    REGULAR_CLOSE = time(16, 0)
    EARLY_CLOSE = time(13, 0)

    def to_et(self, dt: datetime) -> datetime:
        """Convert any timezone-aware or UTC datetime to America/New_York."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(self.TIMEZONE)

    def get_holidays(self, year: int) -> Set[date]:
        """Compute all 10 NYSE observed market holidays for a given year."""
        holidays: Set[date] = set()

        # 1. New Year's Day (Jan 1)
        new_year = date(year, 1, 1)
        if new_year.weekday() == 6:  # Sunday -> Monday Jan 2
            holidays.add(date(year, 1, 2))
        elif new_year.weekday() != 5:  # If Saturday, not observed preceding Friday per NYSE Rule 7.2
            holidays.add(new_year)

        # 2. Martin Luther King Jr. Day (3rd Monday in January)
        holidays.add(self._get_nth_weekday_of_month(year, 1, weekday=0, n=3))

        # 3. Washington's Birthday / Presidents' Day (3rd Monday in February)
        holidays.add(self._get_nth_weekday_of_month(year, 2, weekday=0, n=3))

        # 4. Good Friday (Easter Sunday - 2 days)
        easter = western_easter(year)
        holidays.add(easter - timedelta(days=2))

        # 5. Memorial Day (Last Monday in May)
        holidays.add(self._get_last_weekday_of_month(year, 5, weekday=0))

        # 6. Juneteenth National Independence Day (June 19, established 2021)
        if year >= 2021:
            june19 = date(year, 6, 19)
            if june19.weekday() == 5:
                holidays.add(date(year, 6, 18))
            elif june19.weekday() == 6:
                holidays.add(date(year, 6, 20))
            else:
                holidays.add(june19)

        # 7. Independence Day (July 4)
        july4 = date(year, 7, 4)
        if july4.weekday() == 5:
            holidays.add(date(year, 7, 3))
        elif july4.weekday() == 6:
            holidays.add(date(year, 7, 5))
        else:
            holidays.add(july4)

        # 8. Labor Day (1st Monday in September)
        holidays.add(self._get_nth_weekday_of_month(year, 9, weekday=0, n=1))

        # 9. Thanksgiving Day (4th Thursday in November)
        holidays.add(self._get_nth_weekday_of_month(year, 11, weekday=3, n=4))

        # 10. Christmas Day (Dec 25)
        xmas = date(year, 12, 25)
        if xmas.weekday() == 5:
            holidays.add(date(year, 12, 24))
        elif xmas.weekday() == 6:
            holidays.add(date(year, 12, 26))
        else:
            holidays.add(xmas)

        return holidays

    def get_early_closes(self, year: int) -> Set[date]:
        """Compute scheduled early close days (13:00 ET) for a given year."""
        early_closes: Set[date] = set()

        # Day after Thanksgiving (Black Friday: 4th Friday in Nov)
        thanksgiving = self._get_nth_weekday_of_month(year, 11, weekday=3, n=4)
        early_closes.add(thanksgiving + timedelta(days=1))

        # Day before Independence Day (July 3 if weekday, and July 4 is Tue-Fri)
        july4 = date(year, 7, 4)
        if july4.weekday() in (1, 2, 3, 4):  # Tue-Fri
            early_closes.add(date(year, 7, 3))

        # Christmas Eve (Dec 24) if a weekday and not a holiday
        xmas_eve = date(year, 12, 24)
        if xmas_eve.weekday() < 5 and xmas_eve not in self.get_holidays(year):
            early_closes.add(xmas_eve)

        return early_closes

    def is_trading_day(self, dt_or_d: Union[datetime, date]) -> bool:
        """Check if date is an open NYSE trading day (weekday and non-holiday)."""
        d = dt_or_d.date() if isinstance(dt_or_d, datetime) else dt_or_d
        if d.weekday() >= 5:
            return False
        return d not in self.get_holidays(d.year)

    def get_market_hours(
        self, dt_or_d: Union[datetime, date]
    ) -> Optional[Tuple[datetime, datetime]]:
        """Return (market_open, market_close) in America/New_York timezone."""
        d = dt_or_d.date() if isinstance(dt_or_d, datetime) else dt_or_d
        if not self.is_trading_day(d):
            return None

        open_dt = datetime.combine(d, self.REGULAR_OPEN, tzinfo=self.TIMEZONE)
        close_time = self.EARLY_CLOSE if d in self.get_early_closes(d.year) else self.REGULAR_CLOSE
        close_dt = datetime.combine(d, close_time, tzinfo=self.TIMEZONE)
        return open_dt, close_dt

    def is_market_open(self, dt: datetime) -> bool:
        """Check if dt falls strictly within open regular market hours."""
        et_dt = self.to_et(dt)
        hours = self.get_market_hours(et_dt.date())
        if not hours:
            return False
        open_dt, close_dt = hours
        return open_dt <= et_dt < close_dt

    def is_first_trading_day_of_month(self, dt_or_d: Union[datetime, date]) -> bool:
        """Check if date is the very first NYSE trading day of its calendar month."""
        d = dt_or_d.date() if isinstance(dt_or_d, datetime) else dt_or_d
        if not self.is_trading_day(d):
            return False
        first_day = date(d.year, d.month, 1)
        cur = first_day
        while cur <= d:
            if self.is_trading_day(cur):
                return cur == d
            cur += timedelta(days=1)
        return False

    def is_last_trading_day_of_month(self, dt_or_d: Union[datetime, date]) -> bool:
        """Check if date is the last NYSE trading day of its calendar month."""
        d = dt_or_d.date() if isinstance(dt_or_d, datetime) else dt_or_d
        if not self.is_trading_day(d):
            return False
        if d.month == 12:
            next_month = date(d.year + 1, 1, 1)
        else:
            next_month = date(d.year, d.month + 1, 1)
        cur = next_month - timedelta(days=1)
        while cur >= d:
            if self.is_trading_day(cur):
                return cur == d
            cur -= timedelta(days=1)
        return False

    def is_last_trading_day_of_week(self, dt_or_d: Union[datetime, date]) -> bool:
        """Check if date is the last NYSE trading day of its calendar week (Mon-Sun)."""
        d = dt_or_d.date() if isinstance(dt_or_d, datetime) else dt_or_d
        if not self.is_trading_day(d):
            return False
        start_week = d - timedelta(days=d.weekday())
        last_t_day = None
        for i in range(5):  # Mon-Fri
            check_d = start_week + timedelta(days=i)
            if self.is_trading_day(check_d):
                last_t_day = check_d
        return d == last_t_day

    @staticmethod
    def _get_nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
        d = date(year, month, 1)
        while d.weekday() != weekday:
            d += timedelta(days=1)
        return d + timedelta(weeks=n - 1)

    @staticmethod
    def _get_last_weekday_of_month(year: int, month: int, weekday: int) -> date:
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        d = next_month - timedelta(days=1)
        while d.weekday() != weekday:
            d -= timedelta(days=1)
        return d


class CadenceType(str, Enum):
    """Multi-cadence schedule trigger types."""
    INTRADAY_BAR = "INTRADAY_BAR"
    DAILY_CLOSE = "DAILY_CLOSE"
    WEEKLY_REBALANCE = "WEEKLY_REBALANCE"
    MONTHLY_MOMENTUM = "MONTHLY_MOMENTUM"


class MarketScheduler:
    """Multi-cadence market hours event scheduler."""

    def __init__(
        self,
        calendar: Optional[MarketCalendar] = None,
        daily_close_offset_minutes: int = 10,
        check_interval_seconds: float = 1.0,
        retry_interval_seconds: float = 60.0,
        on_executed: Optional[Callable[[CadenceType, date], Any]] = None,
    ):
        self.calendar = calendar or MarketCalendar()
        self.daily_close_offset = timedelta(minutes=daily_close_offset_minutes)
        self.check_interval = check_interval_seconds
        self._handlers: Dict[CadenceType, List[Callable[[datetime], Any]]] = {
            c: [] for c in CadenceType
        }
        self._last_executed: Dict[CadenceType, Optional[date]] = {
            c: None for c in CadenceType
        }
        # A failed or skipped cadence is retried, but at most once per
        # retry_interval so a dead feed doesn't hammer handlers every tick.
        self.retry_interval = timedelta(seconds=retry_interval_seconds)
        self._next_retry_at: Dict[CadenceType, Optional[datetime]] = {
            c: None for c in CadenceType
        }
        # Called after a cadence completes so the caller can persist it
        # (in-memory state alone re-ran cadences after a mid-window restart).
        self._on_executed = on_executed
        self._is_running: bool = False
        self._stop_event = asyncio.Event()

    def restore_last_executed(self, state: Dict[CadenceType, date]) -> None:
        """Seed completed-cadence dates loaded from persistent storage."""
        for cadence, day in state.items():
            self._last_executed[cadence] = day

    def on_cadence(self, cadence: CadenceType, handler: Callable[[datetime], Any]) -> None:
        """Register async or sync callback for a specific cadence."""
        self._handlers[cadence].append(handler)

    async def _dispatch(self, cadence: CadenceType, dt: datetime) -> bool:
        """Dispatch registered handlers for cadence.

        Returns True only if every handler finished. A handler that raises, or
        returns exactly False (it skipped, e.g. no live data), leaves the
        cadence not done so it is retried before the close.
        """
        all_succeeded = True
        for handler in self._handlers[cadence]:
            try:
                res = handler(dt)
                if asyncio.iscoroutine(res):
                    res = await res
                if res is False:
                    logger.warning("Cadence %s handler did not complete; will retry", cadence.value)
                    all_succeeded = False
            except Exception as e:
                logger.error("Error executing %s handler %s: %s", cadence.value, handler, e)
                all_succeeded = False
        return all_succeeded

    async def tick(self, current_dt: Optional[datetime] = None) -> List[CadenceType]:
        """Execute one evaluation tick at current_dt (or now ET). Returns triggered cadences."""
        now_et = self.calendar.to_et(current_dt or datetime.now(timezone.utc))
        today = now_et.date()
        triggered: List[CadenceType] = []

        if not self.calendar.is_trading_day(today):
            return triggered

        market_hours = self.calendar.get_market_hours(today)
        if not market_hours:
            return triggered

        market_open, market_close = market_hours
        daily_eval_time = market_close - self.daily_close_offset

        in_window = daily_eval_time <= now_et < market_close

        # 1. Monthly Momentum: 1st trading day of month, 5 min before daily eval
        if self.calendar.is_first_trading_day_of_month(today):
            if daily_eval_time - timedelta(minutes=5) <= now_et < market_close:
                await self._run_cadence(CadenceType.MONTHLY_MOMENTUM, now_et, today, triggered)

        # 2. Daily Close Evaluation: 10 min before close (15:50 ET regular or 12:50 ET early close)
        if in_window:
            await self._run_cadence(CadenceType.DAILY_CLOSE, now_et, today, triggered)

        # 3. Weekly Rebalance: Last trading day of week at 15:50 ET (or 12:50 ET early close)
        if in_window and self.calendar.is_last_trading_day_of_week(today):
            await self._run_cadence(CadenceType.WEEKLY_REBALANCE, now_et, today, triggered)

        return triggered

    async def _run_cadence(
        self, cadence: CadenceType, now_et: datetime, today: date, triggered: List[CadenceType]
    ) -> None:
        """Run a cadence once per day, retrying failures no faster than retry_interval."""
        if self._last_executed[cadence] == today:
            return
        next_retry = self._next_retry_at[cadence]
        if next_retry is not None and now_et < next_retry:
            return
        if await self._dispatch(cadence, now_et):
            self._last_executed[cadence] = today
            self._next_retry_at[cadence] = None
            triggered.append(cadence)
            if self._on_executed:
                try:
                    self._on_executed(cadence, today)
                except Exception as e:
                    logger.error("Failed to persist %s completion: %s", cadence.value, e)
        else:
            self._next_retry_at[cadence] = now_et + self.retry_interval
            logger.warning(
                "Cadence %s not completed; retrying after %s until market close",
                cadence.value,
                self._next_retry_at[cadence].strftime("%H:%M:%S"),
            )

    async def run(self) -> None:
        """Run the periodic scheduler tick loop until stop() is called."""
        self._is_running = True
        self._stop_event.clear()
        logger.info("MarketScheduler loop started (tick: %.1fs)", self.check_interval)
        while not self._stop_event.is_set():
            try:
                await self.tick()
            except Exception as e:
                logger.error("Error in scheduler tick: %s", e)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.check_interval)
            except asyncio.TimeoutError:
                pass
        logger.info("MarketScheduler loop stopped")

    def stop(self) -> None:
        """Stop the scheduler loop gracefully."""
        self._is_running = False
        self._stop_event.set()
