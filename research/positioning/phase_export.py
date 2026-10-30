"""Crowd phase artifact: one compact JSON per market for the Positioning tab's phase plot.

Each weekly report is placed by the commercials' and fund managers' house percentiles (trailing 156
reports, mid-rank, point in time, CFTC legacy futures-only), with the retail percentile for the tooltip,
the entry close (Friday after the Tuesday observation) and the simple forward return over 4, 8 and 13
weeks. The frontend draws the cloud from this file and takes the current point and the recent path from
the live series, so the plot never pretends the artifact is fresher than it is.

Same frame as complex.py (`build_frame`), so every number the panel prints can be checked against
complex_evidence.json. Output: research/positioning/phase/{ID}.json plus index.json.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from bhws_core.percentile import midrank_series  # noqa: E402
from research.positioning.complex import DATA, START, HORIZONS, build_frame  # noqa: E402

OUT_DIR = HERE / "phase"


def _r(v, nd):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return round(float(v), nd)


def retail_pct_by_market(mk, cot) -> dict:
    out = {}
    for m in mk:
        code = m.get("cftc_code")
        if not code or code not in cot:
            continue
        c = cot[code].copy()
        c["date"] = pd.to_datetime(c["date"])
        c = c.sort_values("date").drop_duplicates("date")
        pct = midrank_series(c["retail_net"].tolist())
        out[m["id"]] = dict(zip(c["date"].dt.strftime("%Y-%m-%d"), pct))
    return out


def export(df: pd.DataFrame, mk, cot) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    retail = retail_pct_by_market(mk, cot)
    names = {m["id"]: m.get("name") for m in mk}
    index = {"built": built, "method": "midrank_156w", "basis": "futures_only_legacy",
             "entry": "Friday close after the Tuesday observation", "horizons": list(HORIZONS),
             "markets": {}}
    for mid, d in df.groupby("id"):
        d = d.sort_values("obs")
        dates = d.obs.dt.strftime("%Y-%m-%d").tolist()
        rp = retail.get(mid, {})
        rec = {
            "market": mid, "name": names.get(mid), "built": built,
            "from": dates[0], "to": dates[-1], "n": int(len(d)),
            "method": index["method"], "basis": index["basis"], "entry": index["entry"],
            "dates": dates,
            "mgr": [_r(v, 1) for v in d.mgr], "comm": [_r(v, 1) for v in d.comm],
            "retail": [_r(rp.get(x), 1) for x in dates],
            "close": [_r(v, 6) for v in d.close],
        }
        for h in HORIZONS:
            rec[f"r{h}"] = [_r(v, 4) for v in d[f"r{h}"]]
        (OUT_DIR / f"{mid}.json").write_text(json.dumps(rec, separators=(",", ":")))
        index["markets"][mid] = {"from": rec["from"], "to": rec["to"], "n": rec["n"]}
    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=1))
    return index


def main() -> None:
    mk, cot, px = pd.read_pickle(DATA)
    frame = Path("/tmp/bt/complex_frame.pkl")
    df = pd.read_pickle(frame) if frame.exists() else build_frame(mk, cot, px)
    idx = export(df, mk, cot)
    sizes = sorted((p.stat().st_size // 1024, p.name) for p in OUT_DIR.glob("*.json"))
    print(f"wrote {len(idx['markets'])} markets, largest {sizes[-1][1]} {sizes[-1][0]} KB, "
          f"total {sum(s for s, _ in sizes)} KB")


if __name__ == "__main__":
    main()
