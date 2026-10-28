"""CFTC Commitments of Traders: observation date vs release date.

The CFTC measures positions as of Tuesday close and normally publishes them on
the Friday at 15:30 US Eastern. A US federal holiday in the release week
usually pushes publication back by one business day. The CFTC's own schedule
is the authority; this module applies that general rule and labels the result
as an estimate unless an explicit override is supplied.

Two consequences for BHWS:
  * Any chart or score history indexed on the report date is dated three days
    before anyone could have known the number.
  * Many futures settle before 15:30 ET on Friday (energy 14:30, metals 13:30,
    grains 14:20, Treasuries 15:00), so the Friday settlement of the release
    week is not a tradeable price for a signal released that afternoon. The
    conservative entry is the next session.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, Iterable, Optional, Sequence

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET = timezone(timedelta(hours=-5))

RELEASE_TIME_ET = time(15, 30)

# Manual overrides, keyed by observation (Tuesday) date, for releases the
# general rule cannot know about (for example a government shutdown backlog).
# Values are the actual public release date. Add rows only from the CFTC
# release schedule; never guess.
RELEASE_OVERRIDES: Dict[date, date] = {}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def us_federal_holidays(year: int) -> set:
    """US federal holidays (observed dates). Juneteenth from 2021."""
    h = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),     # MLK
        _nth_weekday(year, 2, 0, 3),     # Presidents
        _last_weekday(year, 5, 0),       # Memorial
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),     # Labor
        _nth_weekday(year, 10, 0, 2),    # Columbus
        _observed(date(year, 11, 11)),
        _nth_weekday(year, 11, 3, 4),    # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    if year >= 2021:
        h.add(_observed(date(year, 6, 19)))
    # New Year falling on Saturday is observed on 31 Dec of the prior year.
    nxt = date(year + 1, 1, 1)
    if nxt.weekday() == 5:
        h.add(date(year, 12, 31))
    return h


def _is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in us_federal_holidays(d.year)


@dataclass(frozen=True)
class ReleaseInfo:
    observed: date          # Tuesday the positions are measured
    released: date          # date the CFTC publishes
    released_at_utc: datetime
    rule: str               # "standard", "holiday_shift" or "override"

    def as_dict(self) -> dict:
        return {
            "observed": self.observed.isoformat(),
            "released": self.released.isoformat(),
            "released_at_utc": self.released_at_utc.isoformat(),
            "release_rule": self.rule,
        }


def release_for(observed: date) -> ReleaseInfo:
    """Public release date for a COT report observed on `observed`."""
    if isinstance(observed, datetime):
        observed = observed.date()
    if observed in RELEASE_OVERRIDES:
        rel = RELEASE_OVERRIDES[observed]
        rule = "override"
    else:
        # Normal release is the Friday of the observation week. (When Tuesday
        # is a holiday the CFTC observes on the Monday; the release is still
        # that week's Friday unless a later holiday pushes it.)
        rel = observed + timedelta(days=(4 - observed.weekday()) % 7)
        rule = "standard"
        # A federal holiday after the observation and up to the normal release
        # day delays publication; in practice it moves to the next business
        # day after the weekend (CFTC 2026 schedule: 5 Jan, 22 Jun, 6 Jul,
        # 16 Nov, 30 Nov, 28 Dec). Monday holidays before the observation do
        # not delay the Friday release.
        span = (rel - observed).days
        holidays_in_window = [
            observed + timedelta(days=k) for k in range(1, span + 1)
            if (observed + timedelta(days=k)) in us_federal_holidays((observed + timedelta(days=k)).year)
        ]
        for _ in holidays_in_window:
            rel += timedelta(days=1)
            rule = "holiday_shift"
        while not _is_business_day(rel):
            rel += timedelta(days=1)
            rule = "holiday_shift"
    at = datetime.combine(rel, RELEASE_TIME_ET, tzinfo=_ET).astimezone(timezone.utc)
    return ReleaseInfo(observed, rel, at, rule)


def latest_expected_observation(now_utc: datetime) -> date:
    """The most recent Tuesday whose report should already be public."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    d = now_utc.astimezone(_ET).date()
    # walk back to a Tuesday
    while d.weekday() != 1:
        d -= timedelta(days=1)
    for _ in range(6):
        if release_for(d).released_at_utc <= now_utc:
            return d
        d -= timedelta(days=7)
    return d


def freshness(latest_observed: Optional[date], now_utc: datetime) -> dict:
    """Classify how current a COT series is.

    fresh    latest report is the one that should be public now
    late     one report behind (common around release time or holidays)
    stale    two or more reports behind; downstream confidence must drop
    missing  no data at all
    """
    expected = latest_expected_observation(now_utc)
    if latest_observed is None:
        return {"status": "missing", "expected_observed": expected.isoformat(),
                "reports_behind": None}
    if isinstance(latest_observed, datetime):
        latest_observed = latest_observed.date()
    behind = max(0, round((expected - latest_observed).days / 7))
    status = "fresh" if behind == 0 else "late" if behind == 1 else "stale"
    rel = release_for(latest_observed)
    return {
        "status": status,
        "reports_behind": behind,
        "expected_observed": expected.isoformat(),
        **rel.as_dict(),
    }


def available_observations(obs_dates: Iterable[date], as_of_utc: datetime) -> list:
    """Observation dates that were public at `as_of_utc` (point-in-time)."""
    return [d for d in obs_dates if release_for(d).released_at_utc <= as_of_utc]
