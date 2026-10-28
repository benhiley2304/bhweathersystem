"""Provenance and freshness for every input that can move a conclusion.

Each input carries: source, observed_at (what period the number describes),
released_at (when it became public), retrieved_at (when BHWS fetched it),
expected frequency, transformations applied, and whether it is a fallback.

The rule from the brief: a failed or stale input must not produce a
normal-looking high-confidence conclusion. `data_confidence()` turns the set
of inputs into a separate confidence grade with reasons. It never changes the
direction of anything; it caps how much weight the read deserves.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

# Days after which an input is stale, by expected frequency. Generous on
# purpose: monthly macro data is "late" well before it is stale.
STALE_AFTER_DAYS = {
    "intraday": 1,
    "daily": 4,        # covers a long weekend
    "weekly": 10,      # one missed COT release
    "monthly": 45,
    "quarterly": 120,
    "static": None,    # e.g. a seasonal profile rebuilt yearly
}

FALLBACK_KINDS = ("timeout_neutral", "snapshot", "sample", "carried_forward", "default")


@dataclass
class InputStamp:
    name: str
    source: str
    expected_frequency: str
    observed_at: Optional[str] = None     # ISO date/time the value describes
    released_at: Optional[str] = None     # ISO date/time it became public
    retrieved_at: Optional[str] = None    # ISO date/time BHWS fetched it
    transformations: List[str] = field(default_factory=list)
    fallback: Optional[str] = None        # one of FALLBACK_KINDS, or None
    critical: bool = True                 # does the read depend on it?
    note: Optional[str] = None

    def status(self, now_utc: datetime) -> str:
        if self.fallback:
            return "fallback"
        ref = self.released_at or self.observed_at
        if ref is None:
            return "missing"
        lim = STALE_AFTER_DAYS.get(self.expected_frequency)
        if lim is None:
            return "fresh"
        ts = _parse(ref)
        if ts is None:
            return "unknown"
        age = (now_utc - ts).total_seconds() / 86400.0
        return "fresh" if age <= lim else "stale"

    def as_dict(self, now_utc: datetime) -> dict:
        d = asdict(self)
        d["status"] = self.status(now_utc)
        return d


def _parse(s: str) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def data_confidence(stamps: List[InputStamp], now_utc: Optional[datetime] = None) -> dict:
    """Grade the evidence base: High / Medium / Low / Insufficient.

    Insufficient  a critical input is missing or a fallback
    Low           a critical input is stale, or two non-critical problems
    Medium        one non-critical input is stale, missing or a fallback
    High          everything current and real
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    reasons = []
    crit_bad = crit_stale = soft_bad = 0
    for s in stamps:
        st = s.status(now_utc)
        if st == "fresh":
            continue
        reasons.append(f"{s.name}: {st}" + (f" ({s.fallback})" if s.fallback else ""))
        if s.critical and st in ("missing", "fallback", "unknown"):
            crit_bad += 1
        elif s.critical and st == "stale":
            crit_stale += 1
        else:
            soft_bad += 1
    if crit_bad:
        grade = "Insufficient"
    elif crit_stale or soft_bad >= 2:
        grade = "Low"
    elif soft_bad == 1:
        grade = "Medium"
    else:
        grade = "High"
    return {"grade": grade, "reasons": reasons,
            "inputs": [s.as_dict(now_utc) for s in stamps]}
