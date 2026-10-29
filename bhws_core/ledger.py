"""Trade ledger and weekly grade record.

The ledger is Ben's: every level in it (zone, entry, stop, target) is his.
The system only freezes the context it saw when the plan was written
(grade under both rule sets, setup score, archetype, confirmation state,
bias) so a closed trade can later be compared with what the replay said
about setups of that class. Nothing here estimates a probability.

Entries live in state["ledger"] = {"entries": [...]}.
The weekly grade record lives in state["ledger_grades"] = {"rows": [...]},
one row per market per build date, so rules v1 and v2 can be compared
forward without re-running anything.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

ZONE_TYPES = ("RBR", "DBD", "DBR", "RBD")   # rally-base-rally, drop-base-drop, drop-base-rally, rally-base-drop
STATUSES = ("planned", "open", "closed", "cancelled")
GRADES_KEEP_DAYS = 3 * 366


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def derive(plan: Dict[str, Any], side: int, sd_weekly: Optional[float] = None) -> Dict[str, Any]:
    """Reward:risk and stop geometry from Ben's levels. Returns only what the
    levels support; missing inputs give None, never a default."""
    e, s, t = _f(plan.get("entry")), _f(plan.get("stop")), _f(plan.get("target"))
    hi, lo = _f(plan.get("zone_hi")), _f(plan.get("zone_lo"))
    out: Dict[str, Any] = {"r_multiple": None, "stop_distance": None, "target_distance": None,
                           "stop_sd": None, "target_sd": None, "zone_width": None,
                           "stop_outside_zone": None, "warnings": []}
    if hi is not None and lo is not None:
        if hi < lo:
            hi, lo = lo, hi
        out["zone_width"] = hi - lo
        if s is not None:
            # stop belongs on the far side of the zone from the entry
            out["stop_outside_zone"] = (s < lo) if side > 0 else (s > hi)
            if out["stop_outside_zone"] is False:
                out["warnings"].append("stop sits inside or on the wrong side of the zone")
    if e is not None and s is not None:
        risk = (e - s) if side > 0 else (s - e)
        if risk <= 0:
            out["warnings"].append("stop is on the wrong side of the entry for this side")
        else:
            out["stop_distance"] = risk
            if t is not None:
                reward = (t - e) if side > 0 else (e - t)
                if reward <= 0:
                    out["warnings"].append("target is on the wrong side of the entry")
                else:
                    out["target_distance"] = reward
                    out["r_multiple"] = round(reward / risk, 2)
            if sd_weekly and sd_weekly > 0 and e:
                out["stop_sd"] = round((risk / e) / sd_weekly, 2)
                if out["target_distance"] is not None:
                    out["target_sd"] = round((out["target_distance"] / e) / sd_weekly, 2)
    return out


def new_entry(payload: Dict[str, Any], context: Optional[Dict[str, Any]] = None,
              sd_weekly: Optional[float] = None, now: Optional[float] = None) -> Dict[str, Any]:
    now = now or time.time()
    side = int(payload.get("side") or 0)
    if side not in (1, -1):
        raise ValueError("side must be 1 (long) or -1 (short)")
    market = str(payload.get("market") or "").upper().strip()
    if not market:
        raise ValueError("market required")
    plan = dict(payload.get("plan") or {})
    zt = plan.get("zone_type")
    if zt is not None and str(zt).upper() not in ZONE_TYPES:
        raise ValueError(f"zone_type must be one of {ZONE_TYPES}")
    if zt is not None:
        plan["zone_type"] = str(zt).upper()
    status = payload.get("status") or "planned"
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    ctx = dict(context or {})
    return {
        "id": uuid.uuid4().hex[:12],
        "created_at": now, "updated_at": now,
        "market": market, "side": side, "status": status,
        "plan": plan,
        "derived": derive(plan, side, sd_weekly),
        "context": ctx,              # frozen system read at plan time
        "fills": list(payload.get("fills") or []),
        "outcome": None,
        "notes": str(payload.get("notes") or ""),
    }


def update_entry(entry: Dict[str, Any], patch: Dict[str, Any], sd_weekly: Optional[float] = None,
                 now: Optional[float] = None) -> Dict[str, Any]:
    now = now or time.time()
    e = dict(entry)
    if "plan" in patch and isinstance(patch["plan"], dict):
        plan = dict(e.get("plan") or {})
        plan.update(patch["plan"])
        zt = plan.get("zone_type")
        if zt is not None and str(zt).upper() not in ZONE_TYPES:
            raise ValueError(f"zone_type must be one of {ZONE_TYPES}")
        e["plan"] = plan
        e["derived"] = derive(plan, int(e["side"]), sd_weekly)
    if "status" in patch:
        if patch["status"] not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        e["status"] = patch["status"]
    if "notes" in patch:
        e["notes"] = str(patch["notes"] or "")
    if "fills" in patch and isinstance(patch["fills"], list):
        e["fills"] = patch["fills"]
    e["updated_at"] = now
    return e


def realised_r(entry: Dict[str, Any], exit_px: float, entry_px: Optional[float] = None) -> Optional[float]:
    """R multiple realised against the planned risk (entry to stop)."""
    side = int(entry["side"])
    plan = entry.get("plan") or {}
    e = _f(entry_px) if entry_px is not None else _f(plan.get("entry"))
    s = _f(plan.get("stop"))
    x = _f(exit_px)
    if e is None or s is None or x is None:
        return None
    risk = (e - s) if side > 0 else (s - e)
    if risk <= 0:
        return None
    pnl = (x - e) if side > 0 else (e - x)
    return round(pnl / risk, 3)


def close_entry(entry: Dict[str, Any], outcome: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    now = now or time.time()
    e = dict(entry)
    x = _f(outcome.get("exit_px"))
    if x is None:
        raise ValueError("exit_px required")
    fill = _f(outcome.get("entry_px"))
    r = realised_r(e, x, fill)
    e["outcome"] = {
        "exit_at": outcome.get("exit_at") or now,
        "exit_px": x, "entry_px": fill,
        "r_realised": r,
        "mfe_px": _f(outcome.get("mfe_px")), "mae_px": _f(outcome.get("mae_px")),
        "note": str(outcome.get("note") or ""),
    }
    e["status"] = "closed"
    e["updated_at"] = now
    return e


def calibration(entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Realised R by archetype and by rules-v2 grade, closed trades only.
    Reported with n; no inference is drawn for n below 10."""
    closed = [e for e in entries if e.get("status") == "closed" and (e.get("outcome") or {}).get("r_realised") is not None]
    def _agg(rows):
        rs = sorted(x["outcome"]["r_realised"] for x in rows)
        n = len(rs)
        if not n:
            return {"n": 0}
        mean = sum(rs) / n
        med = rs[n // 2] if n % 2 else (rs[n // 2 - 1] + rs[n // 2]) / 2
        return {"n": n, "mean_r": round(mean, 3), "median_r": round(med, 3),
                "hit_rate": round(sum(1 for r in rs if r > 0) / n, 3),
                "sum_r": round(sum(rs), 2), "worst": rs[0], "best": rs[-1],
                "enough": n >= 10}
    by_arch: Dict[str, List[Dict[str, Any]]] = {}
    by_grade: Dict[str, List[Dict[str, Any]]] = {}
    for e in closed:
        c = e.get("context") or {}
        by_arch.setdefault(str(c.get("archetype") or "n/a"), []).append(e)
        by_grade.setdefault(str(c.get("grade") or "n/a"), []).append(e)
    return {"all": _agg(closed),
            "by_archetype": {k: _agg(v) for k, v in sorted(by_arch.items())},
            "by_grade": {k: _agg(v) for k, v in sorted(by_grade.items())},
            "open": sum(1 for e in entries if e.get("status") == "open"),
            "planned": sum(1 for e in entries if e.get("status") == "planned")}


def append_grades(store: Optional[Dict[str, Any]], day: str, rows: List[Dict[str, Any]],
                  keep_days: int = GRADES_KEEP_DAYS) -> Dict[str, Any]:
    """Record one row per market for `day` (ISO date). Re-running on the same
    day replaces that day's rows. Old rows beyond keep_days are pruned."""
    store = dict(store or {})
    old = [r for r in (store.get("rows") or []) if r.get("day") != day]
    cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - keep_days * 86400))
    old = [r for r in old if str(r.get("day") or "") >= cutoff]
    new = []
    for r in rows:
        if not r.get("market"):
            continue
        new.append({"day": day, **r})
    store["rows"] = old + new
    store["updated_at"] = time.time()
    return store


def grade_row(market: str, setup: Dict[str, Any], score: Optional[Dict[str, Any]],
              close: Optional[float]) -> Dict[str, Any]:
    """Flatten one Desk market read into a forward-record row."""
    g = (setup.get("grade") or {})
    cf = setup.get("confirm") or {}
    return {
        "market": market,
        "side": int(setup.get("side") or 0),
        "lean": int(setup.get("lean") or 0),
        "grade": g.get("key"),
        "grade_v1": setup.get("grade_v1"),
        "archetype": setup.get("archetype"),
        "confirm": cf.get("state") if isinstance(cf, dict) else None,
        "caps": list(setup.get("caps") or []),
        "setup_score": (score or {}).get("score"),
        "ready": ((score or {}).get("ready") or {}).get("key"),
        "timing": ((score or {}).get("timing") or {}).get("state"),
        "close": close,
    }
