from datetime import date, datetime, timezone

from bhws_core.cot_calendar import (release_for, freshness, available_observations,
                                    latest_expected_observation)

# CFTC 2026 release schedule, holiday-delayed rows (cftc.gov ReleaseSchedule).
CFTC_2026_DELAYED = {
    date(2025, 12, 30): date(2026, 1, 5),
    date(2026, 6, 16): date(2026, 6, 22),
    date(2026, 6, 30): date(2026, 7, 6),
    date(2026, 11, 10): date(2026, 11, 16),
    date(2026, 11, 24): date(2026, 11, 30),
    date(2026, 12, 22): date(2026, 12, 28),
}


def test_standard_release_is_friday_1530_et():
    r = release_for(date(2026, 10, 20))
    assert r.released == date(2026, 10, 23)
    assert r.rule == "standard"
    assert r.released_at_utc == datetime(2026, 10, 23, 19, 30, tzinfo=timezone.utc)  # EDT


def test_matches_cftc_2026_holiday_schedule():
    for obs, rel in CFTC_2026_DELAYED.items():
        assert release_for(obs).released == rel, obs


def test_monday_holidays_do_not_delay():
    # Labor Day 7 Sep 2026, Columbus Day 12 Oct 2026: CFTC released 11 Sep, 16 Oct.
    assert release_for(date(2026, 9, 8)).released == date(2026, 9, 11)
    assert release_for(date(2026, 10, 13)).released == date(2026, 10, 16)


def test_no_false_holiday_shifts_in_2026():
    from datetime import timedelta
    d, shifted = date(2026, 1, 6), []
    while d.year == 2026:
        if release_for(d).released.weekday() != 4:
            shifted.append(d)
        d += timedelta(days=7)
    assert set(shifted) <= set(CFTC_2026_DELAYED) | {date(2026, 12, 29)}


def test_point_in_time_availability():
    obs = [date(2026, 10, 13), date(2026, 10, 20)]
    # Friday 23 Oct 15:00 ET = before release of the 20 Oct report
    before = datetime(2026, 10, 23, 19, 0, tzinfo=timezone.utc)
    after = datetime(2026, 10, 23, 19, 31, tzinfo=timezone.utc)
    assert available_observations(obs, before) == [date(2026, 10, 13)]
    assert available_observations(obs, after) == obs


def test_freshness_states():
    now = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
    assert latest_expected_observation(now) == date(2026, 10, 20)
    assert freshness(date(2026, 10, 20), now)["status"] == "fresh"
    assert freshness(date(2026, 10, 13), now)["status"] == "late"
    assert freshness(date(2026, 9, 29), now)["status"] == "stale"
    assert freshness(None, now)["status"] == "missing"
