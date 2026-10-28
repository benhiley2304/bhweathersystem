"""Point-in-time check of the Issue 11 playbook on CFTC futures-only data.

Design fixed before results were looked at:
  * Positioning: legacy non-commercial net, futures only, house percentile
    (trailing 156 reports, mid-rank, >= 20 reports), computed point-in-time.
  * Event: first report managers cross >= 90 or <= 10; one per side per 13
    reports. Fade direction = against the crowd.
  * Timing: a report is usable only after its CFTC release (Friday 15:30 ET,
    holiday-shifted). Entry = first daily close strictly after release day.
    The naive variant enters at the observation-day (Tuesday) close, to
    measure how much look-ahead flatters results.
  * A  "at the extreme": enter on the event report.
    B  "add on the turn": first report within 6 after the event where managers
       moved against the extreme 2 reports running AND the last weekly close
       known at entry is through its 10-week average. No turn -> no trade.
  * Outcome: 12-week (60 trading day) log return in the fade direction, in
    units of 12-week volatility estimated from the prior 52 weeks only.
    Round-trip cost by group charged in the same units.
  * Split: 2006-2015 vs 2016-2026 as a stability check.
Prices are Yahoo continuous front-month, not back-adjusted: roll gaps add
noise and bias in contango markets (energy, meats). Results are reported by
group so that can be seen.
"""
import json, math, os, sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bhws_core.percentile import midrank_series
from bhws_core.playbook import find_events
from bhws_core.cot_calendar import release_for
from research.fetch_data import MARKETS

DATA = os.environ.get("BHWS_RESEARCH_DATA", "/home/user/workspace/bhws_upgrade/research_data")
BASIS = os.environ.get("BHWS_COT_BASIS", "futures_only")   # or "combined"
PREFIX = "cot_" if BASIS == "futures_only" else "cotcomb_"
SUFFIX = "" if BASIS == "futures_only" else "_combined"
COST_BP = {"fx": 2, "equity": 2, "rates": 2, "metals": 4, "energy": 6, "ags": 8, "softs": 10, "meats": 10}
H = 60


def load(mid):
    c = pd.read_csv(f"{DATA}/{PREFIX}{mid}.csv", parse_dates=["date"])
    p = pd.read_csv(f"{DATA}/px_{mid}.csv", index_col=0, parse_dates=True)["Close"].astype(float)
    p = p[p > 0]
    return c, p


def run(hi=90, lo=10, turn_n=2, horizon=H, naive=False, start="2006-01-01"):
    rows = []
    for mid, (_, _, grp) in MARKETS.items():
        try:
            c, p = load(mid)
        except FileNotFoundError:
            continue
        c = c.sort_values("date").reset_index(drop=True)
        pct = midrank_series(c.mgr_net.tolist())
        # re-label extremes at custom thresholds
        ev = find_events(pct, hi=hi, lo=lo)
        lp = np.log(p)
        wk = p.resample("W-FRI").last().dropna()
        sma10 = wk.rolling(10).mean()
        wret = np.log(wk).diff()
        vol12 = wret.rolling(52).std() * math.sqrt(12)
        idx = p.index

        def entry_after(d_obs):
            d = pd.Timestamp(d_obs if naive else release_for(d_obs.date()).released)
            pos = idx.searchsorted(d, side="left" if naive else "right")
            return pos if pos < len(idx) else None

        def outcome(pos, fade):
            if pos is None or pos + horizon >= len(idx):
                return None
            t0 = idx[pos]
            v = vol12[:t0].dropna()
            if v.empty or not v.iloc[-1] > 0:
                return None
            r = (lp.iloc[pos + horizon] - lp.iloc[pos]) * fade
            cost = COST_BP[grp] / 1e4
            return (r - cost) / v.iloc[-1], t0

        def span(p0, p1, fade, v0):
            """log return p0->p1 in sigma units of the vol known at p0, net of one cost."""
            if p0 is None or p1 is None or p1 >= len(idx) or p1 <= p0:
                return None
            return ((lp.iloc[p1] - lp.iloc[p0]) * fade - COST_BP[grp] / 1e4) / v0

        for i, fade in ev:
            d0 = c.date.iloc[i]
            if d0 < pd.Timestamp(start):
                continue
            pos_e = entry_after(d0)
            add_pos = None
            a = outcome(pos_e, fade)
            if a:
                rows.append(dict(mkt=mid, grp=grp, fade=fade, variant="A_extreme", t=a[1], r=a[0]))
            # B: add on the turn within 6 reports
            took = False
            for k in range(i + 1, min(i + 7, len(c))):
                ok = all((c.mgr_net.iloc[j] - c.mgr_net.iloc[j - 1]) * fade > 0 for j in range(k - turn_n + 1, k + 1)) \
                    and k - turn_n + 1 > i
                if not ok:
                    continue
                pos = entry_after(c.date.iloc[k])
                if pos is None:
                    break
                t_ent = idx[pos]
                wk_known = wk[:t_ent - pd.Timedelta(days=0 if naive else 1)]
                s_known = sma10.reindex(wk_known.index)
                if wk_known.empty or pd.isna(s_known.iloc[-1]):
                    continue
                if (wk_known.iloc[-1] - s_known.iloc[-1]) * fade > 0:
                    add_pos = pos
                    b = outcome(pos, fade)
                    if b:
                        rows.append(dict(mkt=mid, grp=grp, fade=fade, variant="B_turn_add", t=b[1], r=b[0]))
                        took = True
                    break
            rows.append(dict(mkt=mid, grp=grp, fade=fade, variant="B_taken" if took else "B_no_turn",
                             t=pd.Timestamp(d0), r=np.nan))
            # C: scaled playbook WITHOUT zones (zones cannot be tested mechanically).
            # Half at the first close after the extreme's release. If the add fires
            # within six reports, add the other half; each half is held 12 weeks
            # from its own entry. If no add, the starter is closed at the first
            # close after the sixth report's release (Issue 11 time stop).
            if a and pos_e is not None:
                v0 = vol12[:idx[pos_e]].dropna().iloc[-1]
                if add_pos is not None and took:
                    r_s = span(pos_e, pos_e + horizon, fade, v0)
                    r_a = span(add_pos, add_pos + horizon, fade, v0)
                    if r_s is not None and r_a is not None:
                        rows.append(dict(mkt=mid, grp=grp, fade=fade, variant="C_scaled", t=idx[pos_e],
                                         r=0.5 * r_s + 0.5 * r_a, added=True))
                elif add_pos is None and i + 6 < len(c):
                    pos_x = entry_after(c.date.iloc[i + 6])
                    r_s = span(pos_e, pos_x, fade, v0)
                    if r_s is not None:
                        rows.append(dict(mkt=mid, grp=grp, fade=fade, variant="C_scaled", t=idx[pos_e],
                                         r=0.5 * r_s, added=False))
    return pd.DataFrame(rows)


def summarise(df, label):
    out = {}
    for v in ("A_extreme", "B_turn_add", "C_scaled"):
        x = df[df.variant == v].r.dropna().values
        if len(x) == 0:
            continue
        rng = np.random.default_rng(11)
        boots = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        losses = x[x < 0]
        out[v] = {
            "n": int(len(x)), "hit": round(float((x > 0).mean()), 3),
            "mean_sigma": round(float(x.mean()), 3),
            "mean_ci90": [round(float(np.percentile(boots, 5)), 3), round(float(np.percentile(boots, 95)), 3)],
            "median_sigma": round(float(np.median(x)), 3),
            "avg_loss_sigma": round(float(losses.mean()), 3) if len(losses) else None,
            "worst5_sigma": round(float(np.percentile(x, 5)), 3),
        }
    nt = (df.variant == "B_no_turn").sum(); tk = (df.variant == "B_taken").sum()
    out["events"] = int(nt + tk); out["turn_rate"] = round(tk / (nt + tk), 3) if nt + tk else None
    out["label"] = label
    return out


if __name__ == "__main__":
    res = {}
    base = run()
    res["base"] = summarise(base, "release-aligned, 90/10, 2 reports, 12w")
    res["naive_obs_date"] = summarise(run(naive=True), "NAIVE: entry at Tuesday observation close")
    res["thr_95_5"] = summarise(run(hi=95, lo=5), "95/5")
    res["thr_85_15"] = summarise(run(hi=85, lo=15), "85/15")
    res["turn_1"] = summarise(run(turn_n=1), "1 report turning")
    res["h_8w"] = summarise(run(horizon=40), "8-week horizon")
    res["early_2006_2015"] = summarise(base[base.t < "2016-01-01"], "2006-2015")
    res["late_2016_2026"] = summarise(base[base.t >= "2016-01-01"], "2016-2026")
    for g in sorted(base.grp.unique()):
        res[f"grp_{g}"] = summarise(base[base.grp == g], f"group {g}")
    # per market (A and B), small n: descriptive only
    pm = {}
    for m in sorted(base.mkt.unique()):
        s = summarise(base[base.mkt == m], m)
        pm[m] = {k: (v["n"], v["hit"], v["mean_sigma"]) for k, v in s.items() if isinstance(v, dict)}
    res["per_market_(n,hit,mean)"] = pm
    out = os.path.join(DATA, f"playbook_pit_results{SUFFIX}.json")
    json.dump(res, open(out, "w"), indent=1, default=str)
    base.to_csv(os.path.join(DATA, f"playbook_pit_trades{SUFFIX}.csv"), index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "per_market_(n,hit,mean)"}, indent=1, default=str))
