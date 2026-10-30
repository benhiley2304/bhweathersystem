"""Payoff profiles ("does it pay"): replay history of first-passage outcomes and excursions for a
class of setups, plus a stress test of Ben's own levels against that history.

The engine evaluates levels, it never generates them. Everything it returns is replay history for
setups like the current one (research/payoff/build.py, point-in-time Desk replay 2012-2026).
Nothing here is a probability for a live trade and no expectancy is implied.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "research", "payoff", "payoff_profiles.json")
_DATA: Optional[Dict[str, Any]] = None

MIN_USE = 120          # smallest cell the panel will lead with; narrower cells are listed, not used
SIDE_WORD = {1: "long", -1: "short"}


def load(path: Optional[str] = None) -> Dict[str, Any]:
    global _DATA
    if _DATA is None or path:
        with open(path or _PATH, "r") as fh:
            _DATA = json.load(fh)
    return _DATA


def available() -> bool:
    try:
        return bool(load().get("cells"))
    except Exception:
        return False


def _row(cell: Dict[str, Any], k: float) -> Optional[Dict[str, Any]]:
    for r in cell.get("first_passage") or []:
        if abs(float(r["k"]) - k) < 1e-9:
            return r
    return None


def lookup(arch: Optional[str], side: Optional[int], confirm: Optional[str], cat: Optional[str],
           market: Optional[str] = None) -> Dict[str, Any]:
    """Most specific cell with n >= MIN_USE for this setup class, with the narrower cells listed,
    the market's own history (if any) and the unconditional baseline for the asset class."""
    D = load()
    cells = D.get("cells") or {}
    out: Dict[str, Any] = {"available": False, "primary": None, "chain": [], "own": None,
                           "baseline": None, "meta": D.get("meta") or {}}
    sw = SIDE_WORD.get(int(side)) if side in (1, -1) else None
    base = (D.get("baseline") or {})
    out["baseline"] = base.get(cat) or base.get(sw) or base.get("all")
    if not arch or not sw:
        out["reason"] = "No directional archetype to match against the replay."
        return out
    keys: List[str] = []
    if confirm and cat:
        keys.append(f"{arch}|{sw}|{confirm}|{cat}")
    if confirm:
        keys.append(f"{arch}|{sw}|{confirm}")
    keys.append(f"{arch}|{sw}")
    keys.append(f"{arch}")
    keys.append("all")
    chain = []
    primary = None
    for key in keys:
        c = cells.get(key)
        if not c:
            chain.append({"key": key, "n": 0, "label": key.replace("|", ", "), "used": False, "missing": True})
            continue
        r1 = _row(c, 1.0) or {}
        item = {"key": key, "label": c["label"], "n": c["n"], "level": c["level"], "used": False,
                "target2_k1": r1.get("target2"), "stop_k1": r1.get("stop")}
        if primary is None and c["n"] >= MIN_USE:
            primary = dict(c, key=key)
            item["used"] = True
        chain.append(item)
    out["chain"] = chain
    if primary is None:
        out["reason"] = "No cell of this class has enough replay rows."
        return out
    out["available"] = True
    out["primary"] = primary
    own = D.get("own") or {}
    if market:
        o = own.get(f"{market}|{arch}|{sw}") or own.get(f"{market}|{arch}")
        if o:
            out["own"] = dict(o, key=f"{market}|{arch}")
    return out


def _xrow(cell: Optional[Dict[str, Any]], k: float, T: int) -> Optional[Dict[str, Any]]:
    for r in (cell or {}).get("execution") or []:
        if abs(float(r["k"]) - k) < 1e-9 and int(r["T"]) == int(T):
            return r
    return None


PF_DEFINITION = ("Profit factor = gross R won \u00f7 gross R lost across all trades in the cell, time-outs marked at the final close; "
                 "1.0 is break-even before costs. With win rate w and a fixed T:1 payoff, PF = wT \u00f7 (1 \u2212 w): "
                 "34% at 2:1 gives 1.03, 24% at 3:1 gives 0.95. Expectancy is the mean R per trade.")
SCALE_DEFINITION = ("Half then add: half a unit at entry; when a close reaches +1R before \u22121R the other half is added at that close and the stop "
                    "for the whole position moves to the lowest close of the prior ten sessions (never below the original stop); the target stays at "
                    "the original 2R or 3R. Full: one unit at entry, stop \u22121R, target as set.")


def _nearest_k(stop_sd: float, ks: List[float]) -> float:
    return min(ks, key=lambda k: abs(k - stop_sd))


def _pct(x: Optional[float]) -> Optional[int]:
    return None if x is None else int(round(100 * float(x)))


def stress(profile: Dict[str, Any], stop_sd: Optional[float], target_sd: Optional[float],
           r_multiple: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Evaluate Ben's stop and target, expressed in weekly sigma, against the class history.
    Returns sentences and flags. Never proposes a level."""
    if not profile or not profile.get("available") or stop_sd is None or stop_sd <= 0:
        return None
    cell = profile["primary"]
    base = profile.get("baseline") or {}
    ks = [float(r["k"]) for r in cell["first_passage"]]
    k = _nearest_k(stop_sd, ks)
    row = _row(cell, k) or {}
    brow = _row(base, k) if base else None
    exc = cell.get("excursion") or {}
    e4 = exc.get("4") or {}
    e13 = exc.get("13") or {}
    flags: List[Dict[str, str]] = []
    notes: List[str] = []
    out_of_range = stop_sd > 2.0 or stop_sd < 0.35
    n = cell["n"]
    lab = cell["label"]
    if r_multiple is None and target_sd and stop_sd:
        r_multiple = round(target_sd / stop_sd, 2)
    use3 = r_multiple is not None and r_multiple >= 2.75
    tkey = "target3" if use3 else "target2"
    tword = "3R" if use3 else "2R"
    skey = "stop3" if use3 else "stop"
    s1 = (f"In {n:,} past {lab} setups, a stop {stop_sd:.2f}\u03c3 from entry sits nearest the {k:g}\u03c3 row: "
          f"the stop was hit before a {tword} target in {_pct(row.get(skey, row.get('stop')))}% of cases and the target came first in "
          f"{_pct(row.get(tkey))}%")
    if brow and brow.get(tkey) is not None:
        s1 += f" (any week in this asset class, same rules: {_pct(brow.get(tkey))}%)"
    s1 += "."
    sentences = [s1]
    wt = row.get("weeks_to_target")
    if wt:
        sentences.append(f"When the 2R level was reached it took a median {wt[1]:g} weeks (quartiles {wt[0]:g} to {wt[2]:g}).")
    be = row.get("better_entry_2w")
    if be is not None:
        sentences.append(f"{_pct(be)}% of these setups traded at least half the stop distance against the entry inside two weeks, before any target.")
    if out_of_range:
        notes.append(f"Your stop ({stop_sd:.2f}\u03c3) is outside the tested range (0.5\u03c3 to 1.5\u03c3); the nearest row is shown.")
    if e4.get("mae"):
        mae25, mae50 = e4["mae"][0], e4["mae"][1]
        if stop_sd < mae25:
            flags.append({"v": "neg", "t": f"Stop inside the 4-week noise band: three quarters of past setups in this class ran at least {mae25:.2f}\u03c3 against the entry within 4 weeks (median {mae50:.2f}\u03c3). A stop at {stop_sd:.2f}\u03c3 would have been hit first in {_pct(row.get(skey, row.get('stop')))}% of them."})
        elif stop_sd < mae50:
            flags.append({"v": "mid", "t": f"Stop inside the median 4-week adverse run ({mae50:.2f}\u03c3): more than half of past setups in this class traded through it inside 4 weeks."})
        else:
            flags.append({"v": "pos", "t": f"Stop beyond the median 4-week adverse run ({mae50:.2f}\u03c3). Whether it belongs there is the zone's call, not the statistic's."})
    if target_sd is not None and target_sd > 0 and e13.get("mfe"):
        mfe25, mfe50, mfe75 = e13["mfe"]
        if target_sd > mfe75:
            flags.append({"v": "neg", "t": f"Target beyond the 75th percentile favourable run at 13 weeks ({mfe75:.2f}\u03c3): fewer than a quarter of past setups in this class travelled {target_sd:.2f}\u03c3 inside the window."})
        elif target_sd > mfe50:
            flags.append({"v": "mid", "t": f"Target between the median ({mfe50:.2f}\u03c3) and 75th percentile ({mfe75:.2f}\u03c3) favourable run at 13 weeks: reached by fewer than half of past setups inside the window."})
        else:
            flags.append({"v": "pos", "t": f"Target inside the median 13-week favourable run ({mfe50:.2f}\u03c3): more than half of past setups in this class travelled that far, though not always before the stop."})
    T = 3 if use3 else 2
    xr = _xrow(cell, k, T); bx = _xrow(base, k, T) if base else None
    execution = None
    if xr and xr.get("full") and xr.get("scale"):
        f, sc = xr["full"], xr["scale"]
        def _pf(v): return "n/a" if v is None else f"{v:.2f}"
        s_ex = (f"On a {T}R target at the {k:g}\u03c3 row, full size at entry: win rate {_pct(f.get('win'))}%, profit factor {_pf(f.get('pf'))}, "
                f"expectancy {f.get('exp'):+.2f}R per trade. Half at entry and the other half added at 1:1 with the stop moved under the prior ten-session low: "
                f"win rate {_pct(sc.get('win'))}%, profit factor {_pf(sc.get('pf'))}, expectancy {sc.get('exp'):+.2f}R; the add was triggered in {_pct(sc.get('added'))}% of cases")
        if bx and bx.get("full") and bx.get("scale"):
            s_ex += f" (any week, same rules: {_pf(bx['full'].get('pf'))} full, {_pf(bx['scale'].get('pf'))} scaled)"
        s_ex += "."
        sentences.append(s_ex)
        execution = {"k": k, "T": T, "full": f, "scale": sc, "baseline_full": (bx or {}).get("full"), "baseline_scale": (bx or {}).get("scale"),
                     "full_pf_h": xr.get("full_pf_h"), "scale_pf_h": xr.get("scale_pf_h"), "full_drift_pf": xr.get("full_drift_pf"), "scale_drift_pf": xr.get("scale_drift_pf")}
    stab = None
    if row.get("target2_h1") is not None and row.get("target2_h2") is not None:
        stab = {"h1": row["target2_h1"], "h2": row["target2_h2"], "drift": row.get("target2_drift"),
                "stop_h1": row.get("stop_h1"), "stop_h2": row.get("stop_h2")}
    return {"k": k, "row": row, "baseline_row": brow, "stop_sd": round(stop_sd, 2),
            "target_sd": (round(target_sd, 2) if target_sd else None), "r_multiple": r_multiple,
            "uses": tword, "sentences": sentences, "flags": flags, "notes": notes, "stability": stab, "execution": execution,
            "definitions": {"profit_factor": PF_DEFINITION, "scale": SCALE_DEFINITION},
            "caveat": "Replay history for setups like this one, tested on daily closes with no costs. Not a probability for this trade; it does not say whether the zone will hold."}


def table(level_max: int = 2, k: float = 1.0) -> List[Dict[str, Any]]:
    """Compact cross-archetype table for the Edge lab: every cell up to level_max at one stop size."""
    D = load()
    rows = []
    for key, c in (D.get("cells") or {}).items():
        if c["level"] > level_max:
            continue
        r = _row(c, k) or {}
        e13 = (c.get("excursion") or {}).get("13") or {}
        x2 = _xrow(c, k, 2) or {}; x3 = _xrow(c, k, 3) or {}
        rows.append({"key": key, "label": c["label"], "level": c["level"], "n": c["n"], "markets": c["markets"],
                     "exec": {"2": x2, "3": x3},
                     "target2": r.get("target2"), "stop": r.get("stop"), "neither": r.get("neither"), "target3": r.get("target3"),
                     "target2_drift": r.get("target2_drift"), "h1": r.get("target2_h1"), "h2": r.get("target2_h2"),
                     "weeks": (r.get("weeks_to_target") or [None, None, None])[1], "better_entry": r.get("better_entry_2w"),
                     "skew13": e13.get("skew"), "mfe13": (e13.get("mfe") or [None, None, None])[1], "mae13": (e13.get("mae") or [None, None, None])[1]})
    rows.sort(key=lambda x: (x["level"], -(x["target2"] or 0)))
    base = ((D.get("baseline") or {}).get("all") or {})
    br = _row(base, k) if base else None
    bx = {"2": _xrow(base, k, 2), "3": _xrow(base, k, 3)} if base else None
    return {"k": k, "rows": rows, "baseline": br, "baseline_exec": bx, "definitions": {"profit_factor": PF_DEFINITION, "scale": SCALE_DEFINITION},
            "meta": D.get("meta") or {}, "modifiers": {
        name: {"n": m["n"], "target2": (_row(m, k) or {}).get("target2"), "stop": (_row(m, k) or {}).get("stop"),
               "skew13": ((m.get("excursion") or {}).get("13") or {}).get("skew")}
        for name, m in (D.get("modifiers") or {}).items()}}
