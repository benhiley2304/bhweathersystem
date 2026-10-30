"""Crowd haircuts and within-complex relative positioning.

Question 1 (haircut): when the managers are crowded on the side of a trade (>= 90th percentile for a
long, <= 10th for a short, commercials on the other side), how did forward returns compare with all weeks
in the same market? Reported per market and pooled per complex, long side and short side separately.

Question 2 (relative): inside a complex (equity indices, FX, metals, energy, grains, softs, livestock,
bonds), when market A's managers sat 50 or more percentile points above market B's, how did B do against
A over the following 4, 8 and 13 weeks, compared with the pair's unconditional spread?

Data: CFTC legacy futures-only (research/replay data snapshot), weekly closes aligned to the Friday after the
Tuesday observation. Percentiles are the house method: trailing 156 reports, mid-rank, point in time.
Forward windows overlap, so n counts weeks, not independent episodes; the JSON also carries a thinned count
(every 8th qualifying week) as a rough guide to independence. Nothing here is a probability for a trade.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bhws_core.percentile import midrank_series  # noqa: E402

DATA = Path("/tmp/bt/data.pkl")
OUT = Path(__file__).resolve().parent / "complex_evidence.json"
START = "2012-01-01"
HORIZONS = (4, 8, 13)
GAP = 50.0

COMPLEXES = {
    "equity": {"label": "US equity indices", "ids": ["ES", "NQ", "YM", "RTY"]},
    "fx": {"label": "G10 currencies vs USD", "ids": ["6E", "6B", "6J", "6A", "6C", "6S", "6N", "6M"]},
    "metals": {"label": "Metals", "ids": ["GC", "SI", "PL", "PA", "HG"]},
    "energy": {"label": "Energy", "ids": ["CL", "NG", "RB", "HO"]},
    "grains": {"label": "Grains", "ids": ["ZW", "ZC", "ZS"]},
    "softs": {"label": "Softs", "ids": ["KC", "SB", "CC", "CT"]},
    "livestock": {"label": "Livestock", "ids": ["LE", "HE", "GF"]},
    "bonds": {"label": "US Treasuries", "ids": ["ZB", "ZN", "ZF", "ZT"]},
    "crypto": {"label": "Crypto", "ids": ["BTC", "ETH"]},
}


def _stats(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean_pct": round(100 * float(x.mean()), 2),
            "median_pct": round(100 * float(x.median()), 2), "pos": round(float((x > 0).mean()), 3)}


def build_frame(mk, cot, px) -> pd.DataFrame:
    rows = []
    for m in mk:
        code = m.get("cftc_code")
        if not code or code not in cot or m["yf"] not in px:
            continue
        c = cot[code].copy()
        c["date"] = pd.to_datetime(c["date"])
        c = c.sort_values("date").drop_duplicates("date")
        c["mgr_pct"] = midrank_series(c["mgr_net"].tolist())
        c["comm_pct"] = midrank_series(c["comm_net"].tolist())
        p = px[m["yf"]].dropna()
        p.index = pd.to_datetime(p.index)
        for _, r in c.iterrows():
            rel = r["date"] + pd.Timedelta(days=3)
            i = p.index.searchsorted(rel)
            if i >= len(p):
                continue
            t0, c0 = p.index[i], float(p.iloc[i])
            if not c0:
                continue
            fw = {}
            for h in HORIZONS:
                j = p.index.searchsorted(t0 + pd.Timedelta(weeks=h))
                fw[f"r{h}"] = float(p.iloc[j]) / c0 - 1 if j < len(p) else np.nan
            rows.append(dict(id=m["id"], obs=r["date"], t0=t0, close=c0, mgr=r["mgr_pct"], comm=r["comm_pct"], **fw))
    df = pd.DataFrame(rows).dropna(subset=["mgr", "comm"])
    return df[df.obs >= START].reset_index(drop=True)


def haircuts(df: pd.DataFrame) -> dict:
    out = {}
    for mid, d in df.groupby("id"):
        long_crowd = d[(d.mgr >= 90) & (d.comm <= 10)]
        short_crowd = d[(d.mgr <= 10) & (d.comm >= 90)]
        rec = {"n_weeks": int(len(d)), "from": str(d.obs.min().date()), "to": str(d.obs.max().date())}
        for h in HORIZONS:
            rec[f"all_{h}"] = _stats(d[f"r{h}"])
            # long side: crowd long = managers crowded on the long side; the long trade's return is r
            rec[f"long_crowded_{h}"] = _stats(long_crowd[f"r{h}"])
            # short side: crowd short; the short trade's return is -r
            rec[f"short_crowded_{h}"] = _stats(-short_crowd[f"r{h}"])
            rec[f"all_short_{h}"] = _stats(-d[f"r{h}"])
        rec["n_long_crowded"] = int(len(long_crowd))
        rec["n_short_crowded"] = int(len(short_crowd))
        rec["n_long_crowded_thin"] = int(len(long_crowd.iloc[::8]))
        rec["n_short_crowded_thin"] = int(len(short_crowd.iloc[::8]))
        out[mid] = rec
    return out


def relative(df: pd.DataFrame) -> dict:
    w = df.pivot_table(index="obs", columns="id", values=["mgr", "comm"] + [f"r{h}" for h in HORIZONS])
    out = {}
    for ckey, c in COMPLEXES.items():
        ids = [i for i in c["ids"] if i in w["mgr"].columns]
        pairs = {}
        pooled = {f"d{h}": [] for h in HORIZONS}
        pooled_base = {f"d{h}": [] for h in HORIZONS}
        n_thin = 0
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                d = pd.DataFrame({"gap": w["mgr"][a] - w["mgr"][b]})
                for h in HORIZONS:
                    d[f"d{h}"] = w[f"r{h}"][b] - w[f"r{h}"][a]
                d = d.dropna()
                q = d[d.gap >= GAP]
                rec = {"n_all": int(len(d)), "n": int(len(q)), "n_thin": int(len(q.iloc[::8]))}
                for h in HORIZONS:
                    rec[f"base_{h}"] = _stats(d[f"d{h}"])
                    rec[f"cond_{h}"] = _stats(q[f"d{h}"])
                    pooled[f"d{h}"].append(q[f"d{h}"])
                    pooled_base[f"d{h}"].append(d[f"d{h}"])
                n_thin += rec["n_thin"]
                pairs[f"{a}>{b}"] = rec
        agg = {"label": c["label"], "ids": ids, "gap_points": GAP, "pairs": pairs, "n_thin": n_thin}
        for h in HORIZONS:
            allq = pd.concat(pooled[f"d{h}"]) if pooled[f"d{h}"] else pd.Series(dtype=float)
            allb = pd.concat(pooled_base[f"d{h}"]) if pooled_base[f"d{h}"] else pd.Series(dtype=float)
            agg[f"cond_{h}"] = _stats(allq)
            agg[f"base_{h}"] = _stats(allb)
        out[ckey] = agg
    return out


def main() -> None:
    mk, cot, px = pd.read_pickle(DATA)
    df = build_frame(mk, cot, px)
    df.to_pickle("/tmp/bt/complex_frame.pkl")
    hc = haircuts(df)
    rel = relative(df)
    # Pooled haircut per complex (long side), equal weight by week
    pooled = {}
    for ckey, c in COMPLEXES.items():
        d = df[df.id.isin(c["ids"])]
        lc = d[(d.mgr >= 90) & (d.comm <= 10)]
        sc = d[(d.mgr <= 10) & (d.comm >= 90)]
        rec = {"label": c["label"], "n_weeks": int(len(d)), "n_long_crowded": int(len(lc)), "n_short_crowded": int(len(sc))}
        for h in HORIZONS:
            rec[f"all_{h}"] = _stats(d[f"r{h}"])
            rec[f"long_crowded_{h}"] = _stats(lc[f"r{h}"])
            rec[f"all_short_{h}"] = _stats(-d[f"r{h}"])
            rec[f"short_crowded_{h}"] = _stats(-sc[f"r{h}"])
        pooled[ckey] = rec
    out = {
        "built": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
        "method": {
            "percentile": "CFTC legacy futures-only, trailing 156 reports, mid-rank, point in time (house method)",
            "entry": "close on the Friday after the Tuesday observation; forward 4, 8 and 13 week simple returns",
            "crowded_long": "managers >= 90th percentile and commercials <= 10th on the observation",
            "crowded_short": "managers <= 10th and commercials >= 90th",
            "relative": f"within a complex, weeks where A's managers percentile exceeded B's by {GAP:.0f} points or more; "
                        "the statistic is B's forward return minus A's, against the pair's unconditional spread",
            "caveats": "forward windows overlap so n counts weeks, not episodes (n_thin keeps every 8th week); "
                       "no costs; indices carry a long drift; Russell data from 2017; history, not probability",
        },
        "complexes": {k: {"label": v["label"], "ids": v["ids"]} for k, v in COMPLEXES.items()},
        "market_complex": {i: k for k, v in COMPLEXES.items() for i in v["ids"]},
        "haircut": hc,
        "haircut_pooled": pooled,
        "relative": rel,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT, OUT.stat().st_size, "bytes")
    # console summary
    for ckey, rec in pooled.items():
        a, l, s = rec["all_13"], rec["long_crowded_13"], rec["short_crowded_13"]
        print(f"{ckey:10s} 13w long: all {a.get('mean_pct')}% pos {a.get('pos')} | crowded long n={l.get('n')} {l.get('mean_pct')}% pos {l.get('pos')} | "
              f"short side all {rec['all_short_13'].get('mean_pct')}% | crowded short n={s.get('n')} {s.get('mean_pct')}% pos {s.get('pos')}")
    for ckey, agg in rel.items():
        print(f"REL {ckey:10s} base13 {agg['base_13'].get('mean_pct')}% pos {agg['base_13'].get('pos')} | cond13 n={agg['cond_13'].get('n')} "
              f"{agg['cond_13'].get('mean_pct')}% pos {agg['cond_13'].get('pos')} | cond8 {agg['cond_8'].get('mean_pct')}% pos {agg['cond_8'].get('pos')} thin {agg['n_thin']}")


if __name__ == "__main__":
    main()
