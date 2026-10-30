"""What happened at detected zones: first-visit reactions, point in time, with a control.

For every detected zone the first visit after formation is scored with ``zones.reaction``:
held = the close travelled one zone height away from the proximal edge in the zone's
direction before any close through the distal edge, inside the horizon (weekly 8 bars,
daily 20 bars). Failed = a close through the distal edge came first. Neither = still
inside or drifting at the horizon.

Control: the same band shifted away from the base by 1.5 heights (a level with no base
behind it), scored the same way from the same formation date. If zones carry information,
the real band should hold more often than the shifted band.

Output: research/zones/zone_reactions.json and a printed table. Replay history, not a
track record; the detector's parameters were calibrated on 32 zones Ben drew in 2026, so
the hit rate against his zones is in-sample and the reaction statistics are on different
data (all 58 markets, 2012-2026) but share the parameters.
"""
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bhws_core.zones import detect, reaction, Zone  # noqa: E402

HORIZON = {"W": 8, "D": 20}
FX_SPOT = {"EURJPY", "EURGBP", "EURAUD", "EURCAD", "EURNZD", "EURCHF", "GBPJPY", "GBPAUD", "GBPCAD", "GBPNZD",
           "GBPCHF", "AUDJPY", "AUDNZD", "AUDCAD", "NZDJPY", "NZDCAD", "CADJPY", "CHFJPY", "AUDCHF"}


def shifted(z: Zone) -> Zone:
    h = z.hi - z.lo
    if z.kind == "demand":
        return replace(z, proximal=z.proximal + 1.5 * h, distal=z.distal + 1.5 * h, proximal_wick=z.proximal_wick + 1.5 * h)
    return replace(z, proximal=z.proximal - 1.5 * h, distal=z.distal - 1.5 * h, proximal_wick=z.proximal_wick - 1.5 * h)


def run(path="/tmp/zones/ohlc.pkl", out="research/zones/zone_reactions.json"):
    data = pickle.load(open(path, "rb"))
    rows = []
    for mkt, d in sorted(data.items()):
        for tf in ("W", "D"):
            df = d[tf]
            if tf == "D":
                df = df.loc[df.index >= df.index[-1] - pd.Timedelta(days=365 * 6)]
            zs = detect(df, tf)
            for z in zs:
                r = reaction(df, z, HORIZON[tf])
                c = reaction(df, shifted(z), HORIZON[tf])
                rows.append({"mkt": mkt, "tf": tf, "scale": z.scale, "kind": z.kind, "pattern": z.pattern,
                             "fx_spot": mkt in FX_SPOT,
                             "formed": z.legout_end.strftime("%Y-%m-%d"), "height_atr": z.height_atr,
                             "departure_base": z.departure_base, "departure_atr": z.departure_atr,
                             "base_bars": z.base_bars, "score": z.score,
                             "visited": r is not None, "held": bool(r and r["held"]), "failed": bool(r and r["failed"]),
                             "mfe_h": r["mfe_h"] if r else None, "mae_h": r["mae_h"] if r else None,
                             "touch": r["touch"].strftime("%Y-%m-%d") if r else None,
                             "hit_1r": bool(r and r["hit_1r"]), "hit_2r": bool(r and r["hit_2r"]), "hit_3r": bool(r and r["hit_3r"]),
                             "stopped": bool(r and r["stopped"]),
                             "c_hit_2r": bool(c and c["hit_2r"]), "c_hit_3r": bool(c and c["hit_3r"]), "c_stopped": bool(c and c["stopped"]),
                             "c_visited": c is not None, "c_held": bool(c and c["held"]), "c_failed": bool(c and c["failed"])})
    df = pd.DataFrame(rows)
    df.to_pickle("/tmp/zones/reactions.pkl")

    def pf(win_rate, k):
        """Profit factor for a fixed k:1 target with a 1R stop, ignoring unresolved trades."""
        if win_rate is None or win_rate <= 0:
            return None
        return round(k * win_rate / (1 - win_rate), 2) if win_rate < 1 else None

    def cell(g):
        v = g[g.visited]
        cv = g[g.c_visited]
        out = {"zones": int(len(g)), "visited": int(len(v)),
               "held": round(float(v.held.mean()), 3) if len(v) else None,
               "failed": round(float(v.failed.mean()), 3) if len(v) else None,
               "mfe_h_med": round(float(v.mfe_h.median()), 2) if len(v) else None,
               "control_visited": int(len(cv)),
               "control_held": round(float(cv.c_held.mean()), 3) if len(cv) else None,
               "control_failed": round(float(cv.c_failed.mean()), 3) if len(cv) else None}
        if len(v):
            # resolved = reached the target or hit the stop inside the horizon
            for k in (2, 3):
                hk = v[f"hit_{k}r"]
                res = v[hk | v.stopped]
                wr = float(res[f"hit_{k}r"].mean()) if len(res) else None
                out[f"win_{k}r"] = None if wr is None else round(wr, 3)
                out[f"pf_{k}r"] = pf(wr, k)
                out[f"resolved_{k}r"] = int(len(res))
                cres = cv[cv[f"c_hit_{k}r"] | cv.c_stopped]
                cwr = float(cres[f"c_hit_{k}r"].mean()) if len(cres) else None
                out[f"control_win_{k}r"] = None if cwr is None else round(cwr, 3)
                out[f"control_pf_{k}r"] = pf(cwr, k)
        return out

    res = {"meta": {"horizon_bars": HORIZON, "markets": int(df.mkt.nunique()), "zones": int(len(df)),
                    "built": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
                    "definition": "held: close travelled one zone height from the proximal edge in the zone's direction "
                                  "before any close through the distal edge, within the horizon after the first visit. "
                                  "Control: the same band shifted 1.5 heights away from the base."},
           "cells": {}}
    keys = [("all", lambda g: g), ]
    for tf in ("W", "D"):
        for scale in ("tight", "wide"):
            for kind in ("demand", "supply"):
                sub = df[(df.tf == tf) & (df.scale == scale) & (df.kind == kind)]
                res["cells"][f"{tf}|{scale}|{kind}"] = cell(sub)
        sub = df[df.tf == tf]
        res["cells"][f"{tf}|all|all"] = cell(sub)
        # by pattern
        for pat in ("DBR", "RBR", "RBD", "DBD"):
            res["cells"][f"{tf}|pattern|{pat}"] = cell(sub[sub.pattern == pat])
        # by departure strength
        res["cells"][f"{tf}|departure<1.5x"] = cell(sub[sub.departure_base < 1.5])
        res["cells"][f"{tf}|departure>=1.5x"] = cell(sub[sub.departure_base >= 1.5])
        res["cells"][f"{tf}|departure>=2.5x"] = cell(sub[sub.departure_base >= 2.5])
        # by base length
        res["cells"][f"{tf}|base<=2"] = cell(sub[sub.base_bars <= 2])
        res["cells"][f"{tf}|base>=5"] = cell(sub[sub.base_bars >= 5])
        # halves
        mid = "2019-07-01"
        res["cells"][f"{tf}|formed<{mid}"] = cell(sub[sub.formed < mid])
        res["cells"][f"{tf}|formed>={mid}"] = cell(sub[sub.formed >= mid])
        res["cells"][f"{tf}|futures"] = cell(sub[~sub.fx_spot])
        res["cells"][f"{tf}|fx_spot"] = cell(sub[sub.fx_spot])
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"{len(df)} zones on {df.mkt.nunique()} markets")
    print(f"{'cell':26s} {'zones':>6s} {'visit':>6s} {'held':>5s} {'fail':>5s} | {'w2R':>5s} {'PF2':>5s} {'w3R':>5s} {'PF3':>5s} | {'c.held':>6s} {'c.w2R':>5s} {'c.PF2':>5s} {'c.w3R':>5s} {'c.PF3':>5s}")
    for k, c in res["cells"].items():
        f = lambda x: "    ·" if x is None else (f"{x:5.0%}" if isinstance(x, float) and x <= 1 else f"{x:5}")
        g = c.get
        print(f"{k:26s} {c['zones']:6d} {c['visited']:6d} {f(c['held']):>5s} {f(c['failed']):>5s} | {f(g('win_2r')):>5s} {str(g('pf_2r')):>5s} {f(g('win_3r')):>5s} {str(g('pf_3r')):>5s} | {f(c['control_held']):>6s} {f(g('control_win_2r')):>5s} {str(g('control_pf_2r')):>5s} {f(g('control_win_3r')):>5s} {str(g('control_pf_3r')):>5s}")
    return res


if __name__ == "__main__":
    run(*sys.argv[1:])
