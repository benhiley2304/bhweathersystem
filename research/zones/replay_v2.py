"""Point-in-time replay of the second supply and demand detector (bhws_core.zones_v2).

For every zone the detector finds on 58 markets (weekly 2011-2026, two-day and daily 2020-2026) the first visit
after the zone became knowable (``known_at``: the bar at whose close the leg out qualified) is marked as a
resting order at the proximal edge with the stop a quarter height beyond the distal edge. Two execution models
(full; half then add at a +1R close with the stop moved to entry) at 1R, 2R and 3R targets, time-outs marked at
the final close of the horizon (weekly 12 bars, two-day 20, daily 25). A control repeats the full model on the
same band moved 1.5 heights away from the base.

Cells split the visits by timeframe, kind, pattern, how the leg out qualified, Yahoo volume tier where the volume
series is coherent, base length, originality, coverage by a weekly zone, the location on the weekly curve and
the method's action matrix at the time of the visit, score quartile, and first versus second half of the sample.

Replay history, not a track record. The detector's parameters were tuned so that it reproduces nine zones Ben
drew on 30 Oct 2026; those nine are in-sample. Everything else here shares the parameters but not the charts.
Yahoo volume is coherent on 45 of 116 series only (contract switches elsewhere); volume cells cover those.

Usage: python3 research/zones/replay_v2.py [/tmp/zones/ohlc.pkl] [research/zones/zone_replay_v2.json]
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bhws_core import zones_v2 as Z  # noqa: E402

HORIZON = {"W": 12, "2D": 20, "D": 25}
FX_SPOT = {"USDJPY", "EURJPY", "EURGBP", "EURAUD", "EURCAD", "EURNZD", "EURCHF", "GBPJPY", "GBPAUD", "GBPCAD", "GBPNZD",
           "GBPCHF", "AUDJPY", "AUDNZD", "AUDCAD", "NZDJPY", "NZDCAD", "CADJPY", "CHFJPY", "AUDCHF"}


def _frames(d):
    out = {}
    daily = d.get("D")
    if daily is not None and len(daily) >= 120:
        if Z.looks_like_spot_fx_defect(daily):
            daily = Z.repair_spot_fx(daily)
        out["D"] = daily
        out["2D"] = Z.resample_2d(daily)
    if d.get("W") is not None and len(d["W"]) >= 120:
        out["W"] = d["W"]
    return out


def _curve_at(price, weekly, when):
    live = [w for w in weekly if w.known_at is not None and w.known_at < when and w.legout_end < when
            and (w.broken is None or w.broken > when)]
    dem = [w for w in live if w.kind == "demand" and w.proximal <= price]
    sup = [w for w in live if w.kind == "supply" and w.proximal >= price]
    ind = [w for w in live if w.kind == "demand" and w.lo <= price <= w.hi]
    ins = [w for w in live if w.kind == "supply" and w.lo <= price <= w.hi]
    if ind:
        return "very low"
    if ins:
        return "very high"
    if dem and sup:
        dz, sz = max(dem, key=lambda w: w.proximal), min(sup, key=lambda w: w.proximal)
        span = sz.proximal - dz.proximal
        f = (price - dz.proximal) / span if span > 0 else 0.5
        return "low" if f < 0.33 else "high" if f > 0.66 else "equilibrium"
    return "open"


def _covered_at(z, weekly, when):
    for w in weekly:
        if w.kind != z.kind or w.known_at is None or w.known_at >= when or (w.broken is not None and w.broken <= when):
            continue
        ov = min(z.hi, w.hi) - max(z.lo, w.lo)
        if ov >= 0.5 * (z.hi - z.lo):
            return True
    return False


def _pit_score(z):
    return Z.score(Z.Zone(**{**Z.asdict(z), "fresh_state": "fresh", "tests": 0, "broken": None}))


def collect(data):
    rows = []
    for mkt, d in sorted(data.items()):
        frames = _frames(d)
        if "W" not in frames:
            continue
        weekly = Z.detect(frames["W"], "W")
        wdf = frames["W"]
        trend_cache = {}
        for tf, df in frames.items():
            zs = weekly if tf == "W" else Z.detect(df, tf)
            vol_ok = Z.volume_reference(df)[0] is not None
            for z in zs:
                r = Z.first_visit(df, z, HORIZON[tf], z.known_at)
                c = Z.first_visit(df, Z.shifted(z), HORIZON[tf], z.known_at)
                if r is None:
                    continue
                when = r["touch"]
                wk = wdf.index.searchsorted(when)
                key = int(wk)
                if key not in trend_cache:
                    sub = wdf.iloc[:max(30, wk)]
                    trend_cache[key] = Z.direction(sub)["label"] if len(sub) >= 30 else "unknown"
                trend = trend_cache[key]
                curve = _curve_at(z.proximal, weekly, when) if tf != "W" else _curve_at(z.proximal, [w for w in weekly if w is not z], when)
                act = Z.action(z.kind, curve, trend)
                path = "explosive" if z.explosive else ("sustained" if any("sustained" in n for n in z.notes) else "slow")
                rows.append(dict(mkt=mkt, tf=tf, kind=z.kind, pattern=z.pattern, path=path, fx_spot=mkt in FX_SPOT,
                                 formed=z.legout_end.strftime("%Y-%m-%d"), known=z.known_at.strftime("%Y-%m-%d") if z.known_at is not None else None,
                                 touch=when.strftime("%Y-%m-%d"), pre=r["pre_visited"], base_bars=z.base_bars,
                                 height_atr=z.height_atr, run_atr=z.run_atr, swing_h=z.swing_h, body_ratio=z.body_ratio,
                                 original=z.original, vol_ok=vol_ok, vol_out=z.vol_out, vol_tier=z.vol_tier,
                                 score=_pit_score(z), covered=_covered_at(z, weekly, when) if tf != "W" else None,
                                 curve=curve, trend=trend, allowed=act["allowed"], tier=act["tier"],
                                 hit1=r["hit"][1], hit2=r["hit"][2], hit3=r["hit"][3], stopped=r["stopped"],
                                 held=r["held"], failed=r["failed"], mfe=r["mfe_r"], mae=r["mae_r"],
                                 f1=r["full"][1], f2=r["full"][2], f3=r["full"][3], s2=r["scale"][2], s3=r["scale"][3],
                                 added=r["added"],
                                 c2=c["full"][2] if c else None, c_hit2=bool(c and c["hit"][2]), c_stopped=bool(c and c["stopped"])))
    return pd.DataFrame(rows)


def pf(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return None
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    return round(float(g / l), 2) if l > 0 else None


def cell(g):
    if len(g) == 0:
        return {"n": 0}
    out = {"n": int(len(g)), "markets": int(g.mkt.nunique())}
    for k in (1, 2, 3):
        hk = g[f"hit{k}"]
        res = g[hk | g.stopped]
        out[f"win{k}"] = round(float(res[f"hit{k}"].mean()), 3) if len(res) else None
        out[f"hit{k}_all"] = round(float(hk.mean()), 3)
    out["pf_full_2"], out["pf_full_3"] = pf(g.f2), pf(g.f3)
    out["pf_scale_2"], out["pf_scale_3"] = pf(g.s2), pf(g.s3)
    out["exp_full_2"] = round(float(g.f2.mean()), 3)
    out["exp_scale_2"] = round(float(g.s2.mean()), 3)
    out["exp_full_3"] = round(float(g.f3.mean()), 3)
    out["held"], out["failed"] = round(float(g.held.mean()), 3), round(float(g.failed.mean()), 3)
    out["mfe_med"], out["mae_med"] = round(float(g.mfe.median()), 2), round(float(g.mae.median()), 2)
    cg = g[g.c2.notna()]
    out["control_n"] = int(len(cg))
    out["control_pf_2"] = pf(cg.c2) if len(cg) else None
    cres = cg[cg.c_hit2 | cg.c_stopped]
    out["control_win2"] = round(float(cres.c_hit2.mean()), 3) if len(cres) else None
    return out


def summarise(df):
    res = {"meta": {"horizon": HORIZON, "markets": int(df.mkt.nunique()), "visits": int(len(df)),
                    "built": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
                    "definition": ("First visit after the zone became knowable; resting order at the proximal edge, stop a quarter "
                                   "height beyond the distal edge. full: one unit, -1R stop, +T R target. scale: half at the edge, "
                                   "other half added at the first close at or beyond +1R with the stop moved to entry. Time-outs "
                                   "marked at the final close. PF = gross R won / gross R lost. win_k = share of resolved visits "
                                   "(target or stop) that reached k R first. Control: the same band 1.5 heights away from the base."),
                    "caveat": ("Replay history, not a track record. Parameters tuned on nine zones Ben drew on 30 Oct 2026 "
                               "(in-sample); the rest shares the parameters, not the charts. No costs, no slippage, fills assumed "
                               "at the edge.")},
           "cells": {}}
    fresh = df[~df.pre]
    res["cells"]["all"] = cell(fresh)
    res["cells"]["all|pre_visited"] = cell(df[df.pre])
    for tf in ("W", "2D", "D"):
        t = fresh[fresh.tf == tf]
        res["cells"][f"{tf}"] = cell(t)
        for kind in ("demand", "supply"):
            res["cells"][f"{tf}|{kind}"] = cell(t[t.kind == kind])
        for pat in ("DBR", "RBR", "RBD", "DBD"):
            res["cells"][f"{tf}|pattern|{pat}"] = cell(t[t.pattern == pat])
        for path in ("explosive", "sustained", "slow"):
            res["cells"][f"{tf}|path|{path}"] = cell(t[t.path == path])
        v = t[t.vol_ok & t.vol_tier.notna()]
        res["cells"][f"{tf}|vol|coherent"] = cell(v)
        for tier in ("high", "normal", "low"):
            res["cells"][f"{tf}|vol|{tier}"] = cell(v[v.vol_tier == tier])
        res["cells"][f"{tf}|vol|>=2x"] = cell(v[v.vol_out >= 2.0])
        res["cells"][f"{tf}|base|<=3"] = cell(t[t.base_bars <= 3])
        res["cells"][f"{tf}|base|4-6"] = cell(t[(t.base_bars >= 4) & (t.base_bars <= 6)])
        res["cells"][f"{tf}|base|>6"] = cell(t[t.base_bars > 6])
        res["cells"][f"{tf}|original"] = cell(t[t.original])
        res["cells"][f"{tf}|not_original"] = cell(t[~t.original])
        res["cells"][f"{tf}|swing|>=3h"] = cell(t[t.swing_h >= 3])
        res["cells"][f"{tf}|swing|<3h"] = cell(t[t.swing_h < 3])
        if tf != "W":
            res["cells"][f"{tf}|covered"] = cell(t[t.covered == True])  # noqa: E712
            res["cells"][f"{tf}|not_covered"] = cell(t[t.covered == False])  # noqa: E712
        for loc in ("very low", "low", "equilibrium", "high", "very high", "open"):
            res["cells"][f"{tf}|curve|{loc}"] = cell(t[t.curve == loc])
        res["cells"][f"{tf}|matrix|allowed"] = cell(t[t.allowed == True])  # noqa: E712
        res["cells"][f"{tf}|matrix|not_allowed"] = cell(t[t.allowed == False])  # noqa: E712
        for tier in ("best", "medium", "weak"):
            res["cells"][f"{tf}|matrix|{tier}"] = cell(t[t.tier == tier])
        if len(t):
            q = t.score.quantile([0.25, 0.5, 0.75]).to_list()
            res["cells"][f"{tf}|score|q1"] = cell(t[t.score <= q[0]])
            res["cells"][f"{tf}|score|q4"] = cell(t[t.score > q[2]])
            mid = t.touch.sort_values().iloc[len(t) // 2]
            res["cells"][f"{tf}|half|first"] = cell(t[t.touch < mid])
            res["cells"][f"{tf}|half|second"] = cell(t[t.touch >= mid])
            res["meta"][f"{tf}_split"] = str(mid)
        res["cells"][f"{tf}|futures"] = cell(t[~t.fx_spot])
        res["cells"][f"{tf}|fx_spot"] = cell(t[t.fx_spot])
        # the method's own grade, pre-specified by its rules (not fitted here): the action matrix allows the zone,
        # the base is six bars or fewer, the departure ran at least three zone heights, and on the two-day and
        # daily the zone sits inside a weekly zone of the same kind (covered)
        g = grade_mask(t)
        gr = t[g]
        res["cells"][f"{tf}|grade"] = cell(gr)
        res["cells"][f"{tf}|rest"] = cell(t[~g])
        if len(gr):
            gm = gr.touch.sort_values().iloc[len(gr) // 2]
            res["cells"][f"{tf}|grade|half|first"] = cell(gr[gr.touch < gm])
            res["cells"][f"{tf}|grade|half|second"] = cell(gr[gr.touch >= gm])
            res["meta"][f"{tf}_grade_split"] = str(gm)
            gv = gr[gr.vol_ok & gr.vol_tier.notna()]
            res["cells"][f"{tf}|grade|vol|high"] = cell(gv[gv.vol_tier == "high"])
            res["cells"][f"{tf}|grade|vol|not_high"] = cell(gv[gv.vol_tier != "high"])
            res["cells"][f"{tf}|grade|demand"] = cell(gr[gr.kind == "demand"])
            res["cells"][f"{tf}|grade|supply"] = cell(gr[gr.kind == "supply"])
    return res


def grade_mask(t):
    m = (t.allowed == True) & (t.base_bars <= 6) & (t.swing_h >= 3)  # noqa: E712
    if len(t) and (t.tf != "W").any():
        m = m & ((t.tf == "W") | (t.covered == True))  # noqa: E712
    return m


def table(res):
    print(f"{'cell':28s} {'n':>6s} {'w1':>5s} {'w2':>5s} {'w3':>5s} {'PF2':>5s} {'PF3':>5s} {'sc2':>5s} {'sc3':>5s} {'exp2':>6s} {'held':>5s} | {'c.PF2':>5s} {'c.w2':>5s}")
    f = lambda x: "    ·" if x is None else (f"{x:5.0%}" if isinstance(x, float) and abs(x) <= 1 and x >= 0 else f"{x:5}")
    for k, c in res["cells"].items():
        if c.get("n", 0) == 0:
            continue
        g = c.get
        print(f"{k:28s} {c['n']:6d} {f(g('win1')):>5s} {f(g('win2')):>5s} {f(g('win3')):>5s} {str(g('pf_full_2')):>5s} {str(g('pf_full_3')):>5s} "
              f"{str(g('pf_scale_2')):>5s} {str(g('pf_scale_3')):>5s} {g('exp_full_2'):6.2f} {f(g('held')):>5s} | {str(g('control_pf_2')):>5s} {f(g('control_win2')):>5s}")


def run(path="/tmp/zones/ohlc.pkl", out="research/zones/zone_replay_v2.json"):
    data = pickle.load(open(path, "rb"))
    df = collect(data)
    df.to_pickle("/tmp/zones/replay_v2_rows.pkl")
    res = summarise(df)
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"{len(df)} visited zones on {df.mkt.nunique()} markets; pre-visited {int(df.pre.sum())}")
    table(res)
    return res


if __name__ == "__main__":
    run(*sys.argv[1:])
