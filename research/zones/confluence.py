"""Does a Desk setup pay better when the signal week is sitting in a weekly zone?

Joins the payoff replay (research/payoff, /tmp/bt/payoff_paths.pkl: every graded setup week with its
R-unit outcome at a 1-sigma stop and 2R/3R targets, entry at the Monday close after the signal) with the
weekly zones detected on the same market. A setup week is tagged:

  at_zone      the week's range reached the proximal edge of a weekly zone of the setup's kind
               (demand for longs, supply for shorts) that had formed at least three weeks earlier and had
               not closed through its distal edge before this week; the week itself did not close through it
  fresh        as above and the zone had not been touched before this week
  against      the week reached a zone of the opposite kind (long into supply, short into demand)
  none         neither

Two executions are compared for the tagged weeks:
  engine   the payoff engine's own trade: Monday close entry, 1-sigma stop, 2R/3R targets on closes
           (columns xf_1.0_2 / xf_1.0_3, plus the scaled execution)
  zone     Ben's convention made point in time: entry at the signal Friday close, taken only when that
           close still sits inside the zone or within a quarter height of the proximal edge; stop a quarter
           height beyond the distal edge; 2R and 3R targets from the entry; weekly bars for 13 weeks;
           when stop and target both print in one week the stop is taken
Point in time: zones are built from bars before the signal week; the dedupe step uses the full history so
a few zones could differ from a true as-of run.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bhws_core.zones import detect  # noqa: E402

SETTLE = 3  # weeks after the leg out before a zone is considered known


def tag_market(wk: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    zs = detect(wk, "W")
    H, L, C = wk["High"].to_numpy(float), wk["Low"].to_numpy(float), wk["Close"].to_numpy(float)
    idx = wk.index
    # per zone: formation bar, first-touch bar, broken bar
    info = []
    for z in zs:
        f = idx.searchsorted(z.legout_end, side="right") + SETTLE - 1
        sgn = 1 if z.kind == "demand" else -1
        touch = broken = None
        for u in range(f, len(wk)):
            if broken is None and ((C[u] < z.distal) if sgn > 0 else (C[u] > z.distal)):
                broken = u
                break
            if touch is None and ((L[u] <= z.proximal) if sgn > 0 else (H[u] >= z.proximal)):
                touch = u
        info.append((z, f, touch, broken, sgn))
    out = {}
    for i, r in rows.iterrows():
        w = idx.searchsorted(pd.Timestamp(r.t), side="right") - 1
        if w < 0:
            out[i] = ("none", None, None, None, None)
            continue
        side = int(r.side)
        tag, zz = "none", None
        for z, f, touch, broken, sgn in info:
            if w < f or (broken is not None and broken < w):
                continue
            hit = (L[w] <= z.proximal) if sgn > 0 else (H[w] >= z.proximal)
            if not hit:
                continue
            closed_through = (C[w] < z.distal) if sgn > 0 else (C[w] > z.distal)
            if closed_through:
                continue
            if sgn == side:
                new = "fresh" if (touch is None or touch >= w) else "at_zone"
                if tag != "fresh":
                    tag, zz = new, z
                if tag == "fresh":
                    break
            elif tag == "none":
                tag = "against"
        if zz is None:
            out[i] = (tag, None, None, None, None)
        else:
            # Ben's execution: entry at the proximal edge, stop a quarter height beyond the distal edge,
            # weekly bars from the signal week for 13 weeks; same-bar ambiguity goes against the trade.
            # Point in time: the setup is read at the Friday close, so the entry is that close, taken only
            # when it still sits inside the zone or within a quarter height of the proximal edge. The stop
            # is a quarter height beyond the distal edge; targets are 2R and 3R from the entry; weekly bars
            # for the next 13 weeks; stop before target when both print in one week.
            h = zz.hi - zz.lo
            stop = zz.distal - side * 0.25 * h
            entry = C[w]
            near = (entry <= zz.proximal + 0.25 * h) if side > 0 else (entry >= zz.proximal - 0.25 * h)
            risk = abs(entry - stop)
            if not near or risk <= 0:
                out[i] = (tag, None, None, None, zz.scale)
                continue
            res = {}
            for k in (2, 3):
                outcome = None
                for u in range(w + 1, min(len(wk), w + 14)):
                    stop_hit = (L[u] <= stop) if side > 0 else (H[u] >= stop)
                    best = (H[u] - entry) / risk if side > 0 else (entry - L[u]) / risk
                    if stop_hit:
                        outcome = -1.0
                        break
                    if best >= k:
                        outcome = float(k)
                        break
                if outcome is None:
                    u = min(len(wk) - 1, w + 13)
                    outcome = side * (C[u] - entry) / risk
                res[k] = outcome
            out[i] = (tag, res[2], res[3], risk / float(zz.atr) if zz.atr else None, zz.scale)
    return pd.Series(out)


def resting_market(wk: pd.DataFrame, rows: pd.DataFrame, wait=4, max_atr=2.0) -> pd.Series:
    """Ben's wait-for-the-zone process, point in time. At the signal Friday the nearest fresh weekly zone of
    the setup's kind that price has not yet reached (within ``max_atr`` ATR of the close) gets a resting order
    at its proximal edge. If the order is touched in the next ``wait`` weeks the trade is on at the proximal
    edge with the stop a quarter height beyond the distal edge; 2R/3R targets, 13 weekly bars from the fill,
    stop first when both print in a week. Returns (filled, r2, r3, risk_atr, scale) or Nones when no zone."""
    zs = detect(wk, "W")
    H, L, C = wk["High"].to_numpy(float), wk["Low"].to_numpy(float), wk["Close"].to_numpy(float)
    idx = wk.index
    tr = np.maximum(H - L, np.maximum(abs(H - np.roll(C, 1)), abs(L - np.roll(C, 1))))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    info = []
    for z in zs:
        f = idx.searchsorted(z.legout_end, side="right") + SETTLE - 1
        sgn = 1 if z.kind == "demand" else -1
        touch = None
        for u in range(f, len(wk)):
            if (L[u] <= z.proximal) if sgn > 0 else (H[u] >= z.proximal):
                touch = u
                break
        info.append((z, f, touch, sgn))
    out = {}
    for i, r in rows.iterrows():
        w = idx.searchsorted(pd.Timestamp(r.t), side="right") - 1
        side = int(r.side)
        best = None
        if w >= 14 and atr[w] > 0:
            for z, f, touch, sgn in info:
                if sgn != side or w < f or (touch is not None and touch <= w):
                    continue
                dist = (C[w] - z.proximal) if side > 0 else (z.proximal - C[w])
                if dist <= 0 or dist > max_atr * atr[w]:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, z)
        if best is None:
            out[i] = (None, None, None, None, None)
            continue
        z = best[1]
        h = z.hi - z.lo
        stop = z.distal - side * 0.25 * h
        risk = abs(z.proximal - stop)
        fill = None
        for u in range(w + 1, min(len(wk), w + 1 + wait)):
            if (L[u] <= z.proximal) if side > 0 else (H[u] >= z.proximal):
                fill = u
                break
        if fill is None:
            out[i] = (False, None, None, risk / atr[w], z.scale)
            continue
        res = {}
        for k in (2, 3):
            outcome = None
            for u in range(fill, min(len(wk), fill + 14)):
                stop_hit = (L[u] <= stop) if side > 0 else (H[u] >= stop)
                reach = (H[u] - z.proximal) / risk if side > 0 else (z.proximal - L[u]) / risk
                if stop_hit:
                    outcome = -1.0
                    break
                if reach >= k:
                    outcome = float(k)
                    break
            if outcome is None:
                u = min(len(wk) - 1, fill + 13)
                outcome = side * (C[u] - z.proximal) / risk
            res[k] = outcome
        out[i] = (True, res[2], res[3], risk / atr[w], z.scale)
    return pd.Series(out)


def stats(g, col):
    x = g[col].dropna()
    n = len(x)
    if n < 30:
        return {"n": int(n)}
    gp = float(x[x > 0].sum())
    gl = float(-x[x < 0].sum())
    return {"n": int(n), "win": round(float((x > 0).mean()), 3), "pf": round(gp / gl, 2) if gl > 0 else None,
            "exp": round(float(x.mean()), 3)}


def run(ohlc="/tmp/zones/ohlc.pkl", paths="/tmp/bt/payoff_paths.pkl", out="research/zones/zone_confluence.json"):
    data = pickle.load(open(ohlc, "rb"))
    P = pd.read_pickle(paths)
    P = P[P.id.isin(data.keys())].copy()
    tags = []
    for mkt, g in P.groupby("id"):
        tags.append(tag_market(data[mkt]["W"], g))
    rest = []
    for mkt, g in P.groupby("id"):
        rest.append(resting_market(data[mkt]["W"], g))
    RT = pd.concat(rest)
    P["ro_has"] = RT.map(lambda x: x[0] is not None)
    P["ro_filled"] = RT.map(lambda x: bool(x[0]))
    P["ro_2"] = RT.map(lambda x: x[1])
    P["ro_3"] = RT.map(lambda x: x[2])
    P["ro_risk_atr"] = RT.map(lambda x: x[3])
    P["ro_scale"] = RT.map(lambda x: x[4])
    T = pd.concat(tags)
    P["zone"] = T.map(lambda x: x[0])
    P["ze_2"] = T.map(lambda x: x[1])
    P["ze_3"] = T.map(lambda x: x[2])
    P["ze_risk_atr"] = T.map(lambda x: x[3])
    P["ze_scale"] = T.map(lambda x: x[4])
    P["zone_any"] = P.zone.isin(["fresh", "at_zone"])
    P.to_pickle("/tmp/zones/confluence.pkl")
    res = {"meta": {"rows": int(len(P)), "markets": int(P.id.nunique()), "settle_weeks": SETTLE,
                    "built": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
                    "note": "entry at the Monday close after the signal, 1-sigma stop, closes only; "
                            "tags describe the signal week's position against weekly zones"},
           "cells": {}}

    def block(name, g):
        res["cells"][name] = {"tags": {k: int(v) for k, v in g.zone.value_counts().items()}}
        for tag in ("fresh", "at_zone", "against", "none"):
            s = g[g.zone == tag]
            res["cells"][name][tag] = {"2R_full": stats(s, "xf_1.0_2"), "3R_full": stats(s, "xf_1.0_3"),
                                       "2R_scale": stats(s, "xs_1.0_2"), "3R_scale": stats(s, "xs_1.0_3")}
        ro = g[g.ro_has]
        res["cells"][name]["resting"] = {"orders": int(len(ro)), "filled": int(ro.ro_filled.sum()),
                                         "fill_rate": (round(float(ro.ro_filled.mean()), 3) if len(ro) else None),
                                         "2R": stats(ro, "ro_2"), "3R": stats(ro, "ro_3"),
                                         "risk_atr_med": (round(float(ro.ro_risk_atr.median()), 2) if len(ro) else None)}
        for sc in ("tight", "wide"):
            rs = ro[ro.ro_scale == sc]
            res["cells"][name]["resting"][sc] = {"2R": stats(rs, "ro_2"), "3R": stats(rs, "ro_3")}
        s = g[g.zone_any]
        res["cells"][name]["at_or_fresh"] = {"2R_full": stats(s, "xf_1.0_2"), "3R_full": stats(s, "xf_1.0_3"),
                                             "2R_edge": stats(s, "ze_2"), "3R_edge": stats(s, "ze_3"),
                                             "risk_atr_med": (round(float(s.ze_risk_atr.median()), 2) if len(s) else None)}
        for tag in ("fresh", "at_zone"):
            s = g[g.zone == tag]
            res["cells"][name][tag]["2R_edge"] = stats(s, "ze_2")
            res["cells"][name][tag]["3R_edge"] = stats(s, "ze_3")

    block("all graded", P)
    for cls in ("prime", "setup", "developing"):
        block(f"cls={cls}", P[P.cls == cls])
    block("prime+setup", P[P.cls.isin(["prime", "setup"])])
    for side, lab in ((1, "long"), (-1, "short")):
        block(f"side={lab}", P[P.side == side])
        block(f"prime+setup {lab}", P[(P.side == side) & P.cls.isin(["prime", "setup"])])
    for per in sorted(P.period.unique()):
        block(f"period={per}", P[P.period == per])
    for arch, g in P.groupby("arch"):
        if len(g) >= 300:
            block(f"arch={arch}", g)
    for cat, g in P.groupby("cat"):
        block(f"cat={cat}", g)
    for sc in ("tight", "wide"):
        block(f"scale={sc}", P[(P.ze_scale == sc) | P.ze_scale.isna()])
    # The waiting trade-off: engine outcome of the same setups split by whether the resting order was filled.
    ro = P[P.ro_has]
    res["waiting_tradeoff"] = {
        "orders": int(len(ro)), "fill_rate": round(float(ro.ro_filled.mean()), 3),
        "engine_2R_when_filled": stats(ro[ro.ro_filled], "xf_1.0_2"),
        "engine_2R_when_not_filled": stats(ro[~ro.ro_filled], "xf_1.0_2"),
        "engine_2R_no_zone_ahead": stats(P[~P.ro_has], "xf_1.0_2"),
        "resting_2R_when_filled": stats(ro, "ro_2"),
        "engine_exp_per_order": round(float(ro["xf_1.0_2"].mean()), 3),
        "resting_exp_per_order": round(float(ro.ro_2.fillna(0).mean()), 3),
        "resting_risk_atr_quartiles": [round(float(q), 2) for q in ro[ro.ro_filled].ro_risk_atr.quantile([.25, .5, .75])],
        "note": "Split by a future event (whether price came back to the zone), so it describes the trade-off of waiting, "
                "not a rule. The engine's Monday-close entry wins big when price never returns and loses badly when it does; "
                "the resting order rescues the returning subset and forgoes the runners."}
    Path(out).write_text(json.dumps(res, indent=1))

    print(f"{len(P)} graded setup weeks on {P.id.nunique()} markets")
    print("Resting order at the nearest fresh weekly zone ahead of price (filled within 4 weeks), stop a quarter height beyond the distal, 13 weekly bars from the fill:")
    print(f"{'cell':30s} {'orders':>6s} {'fill':>5s} {'n':>6s} {'w2R':>5s} {'PF2':>5s} {'e2R':>6s} | {'w3R':>5s} {'PF3':>5s} {'e3R':>6s} | tight PF2 / wide PF2")
    for name, c in res["cells"].items():
        r = c["resting"]
        a, b = r["2R"], r["3R"]
        if a["n"] < 30:
            continue
        tp, wp = r["tight"]["2R"].get("pf"), r["wide"]["2R"].get("pf")
        print(f"{name:30s} {r['orders']:6d} {r['fill_rate']:5.0%} {a['n']:6d} {a['win']:5.0%} {a['pf']:5} {a['exp']:+6.2f} | {b['win']:5.0%} {b['pf']:5} {b['exp']:+6.2f} | {tp} ({r['tight']['2R']['n']}) / {wp} ({r['wide']['2R']['n']})")
    print("\nZone execution (entry at the signal Friday close while inside or within a quarter height of the zone, stop a quarter height beyond the distal, weekly bars, 13 weeks):")
    print(f"{'cell':30s} {'tag':10s} {'n':>6s} {'w2R':>5s} {'PF2':>5s} {'e2R':>6s} | {'w3R':>5s} {'PF3':>5s} {'e3R':>6s}")
    for name, c in res["cells"].items():
        for tag in ("fresh", "at_zone", "at_or_fresh"):
            a, b = c[tag]["2R_edge"], c[tag]["3R_edge"]
            if a["n"] < 30:
                continue
            print(f"{name:30s} {tag:10s} {a['n']:6d} {a['win']:5.0%} {a['pf']:5} {a['exp']:+6.2f} | {b['win']:5.0%} {b['pf']:5} {b['exp']:+6.2f}")
    print("\nEngine execution (Monday close entry, 1-sigma stop, closes only):")
    print(f"{'cell':30s} {'tag':10s} {'n':>6s} {'w2R':>5s} {'PF2':>5s} {'e2R':>6s} | {'w3R':>5s} {'PF3':>5s} {'e3R':>6s}")
    for name, c in res["cells"].items():
        for tag in ("fresh", "at_zone", "against", "none"):
            a, b = c[tag]["2R_full"], c[tag]["3R_full"]
            if a["n"] < 30:
                print(f"{name:30s} {tag:10s} {a['n']:6d}")
                continue
            print(f"{name:30s} {tag:10s} {a['n']:6d} {a['win']:5.0%} {a['pf']:5} {a['exp']:+6.2f} | {b['win']:5.0%} {b['pf']:5} {b['exp']:+6.2f}")
    return res


if __name__ == "__main__":
    run(*sys.argv[1:])
