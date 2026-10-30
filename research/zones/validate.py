"""Check the detector against zones Ben marked by hand in the BH Macro reports.

Each row: market, timeframe the zone was drawn on, kind, low, high, report date.
Levels are quoted from the report text (Issues 00a to 11, July to October 2026).
A match is a detected zone of the same kind, formed before the report date, whose
price band overlaps at least half of Ben's band. The script reports the best
match for each row and a hit rate. It is a calibration check, not a backtest.
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import detect, Zone  # noqa: E402

BEN = [
    # market, tf, kind, lo, hi, asof, note
    ("NQ", "W", "demand", 28800, 29563, "2026-10-26", "DBR, four weeks of base"),
    ("NQ", "D", "demand", 28800, 29700, "2026-10-11", "HTF demand on the daily"),
    ("NQ", "D", "demand", 29220, 29800, "2026-07-13", "daily demand, re-entry"),
    ("NQ", "D", "supply", 30200, 31000, "2026-07-13", "supply block"),
    ("NQ", "D", "supply", 30600, 31150, "2026-10-26", "first supply"),
    ("NQ", "D", "demand", 30370, 30600, "2026-10-04", "second demand"),
    ("NQ", "D", "demand", 27000, 28000, "2026-07-13", "demand shelf"),
    ("SI", "D", "demand", 55.0, 60.0, "2026-10-05", "held since July"),
    ("SI", "D", "demand", 45.6, 51.8, "2026-10-05", "larger demand"),
    ("SI", "W", "demand", 45.5, 50.8, "2026-10-26", "fresh demand"),
    ("SB", "W", "supply", 19.2, 21.1, "2026-10-05", "weekly supply after straight-line rally"),
    ("SB", "W", "supply", 21.1, 23.6, "2026-10-05", "upper weekly supply"),
    ("SB", "W", "demand", 13.2, 15.4, "2026-10-05", "the base"),
    ("DX", "W", "supply", 101.8, 104.3, "2026-10-05", "stacked weekly supply"),
    ("DX", "D", "demand", 97.4, 99.7, "2026-09-20", "demand"),
    ("6B", "W", "demand", 1.300, 1.325, "2026-10-05", "weekly demand, large zone"),
    ("6E", "W", "demand", 1.106, 1.124, "2026-10-05", "lower weekly demand"),
    ("6E", "D", "demand", 1.1520, 1.1580, "2026-09-13", "entry demand"),
    ("6E", "D", "demand", 1.1375, 1.1500, "2026-09-13", "larger base"),
    ("6E", "W", "supply", 1.1750, 1.1875, "2026-09-13", "weekly supply"),
    ("USDJPY", "D", "supply", 158.0, 160.6, "2026-10-05", "capped August"),
    ("ZB", "W", "demand", 107.25, 110.0, "2026-09-06", "multi-year weekly demand base"),
    ("LE", "D", "demand", 204.0, 217.0, "2026-09-13", "Q4 2025 base"),
    ("PL", "D", "demand", 1705, 1800, "2026-09-27", "daily demand"),
    ("PL", "D", "demand", 1540, 1675, "2026-09-27", "deeper band"),
    ("CT", "D", "supply", 83.30, 89.00, "2026-07-13", "built Apr/May 2026"),
    ("CT", "D", "supply", 92.5, 97.5, "2026-09-13", "supply capping"),
    ("CT", "D", "supply", 86.0, 88.6, "2026-09-27", "overhead supply"),
    ("CT", "D", "demand", 71.0, 77.5, "2026-09-27", "demand"),
    ("HG", "D", "demand", 6.32, 6.44, "2026-09-21", "demand objective (4H chart)"),
    ("NG", "D", "demand", 2.62, 2.74, "2026-09-21", "demand beneath (4H chart)"),
    ("GC", "D", "supply", 4280, 4330, "2026-10-11", "overhead"),
]


def overlap(z: Zone, lo: float, hi: float) -> float:
    a, b = max(z.lo, lo), min(z.hi, hi)
    return 0.0 if b <= a else (b - a) / (hi - lo)


def main(path="/tmp/zones/ohlc.pkl"):
    data = pickle.load(open(path, "rb"))
    cache = {}
    hits = 0
    rows = []
    for mkt, tf, kind, lo, hi, asof, note in BEN:
        key = (mkt, tf)
        if key not in cache:
            cache[key] = detect(data[mkt][tf], tf)
        asof_ts = pd.Timestamp(asof)
        cands = [z for z in cache[key] if z.kind == kind and z.legout_end <= asof_ts
                 and (z.broken is None or z.broken > asof_ts)]
        best = max(cands, key=lambda z: overlap(z, lo, hi), default=None)
        ov = overlap(best, lo, hi) if best else 0.0
        # also look on the other timeframe
        other = "W" if tf == "D" else "D"
        okey = (mkt, other)
        if okey not in cache:
            cache[okey] = detect(data[mkt][other], other)
        oc = [z for z in cache[okey] if z.kind == kind and z.legout_end <= asof_ts
              and (z.broken is None or z.broken > asof_ts)]
        obest = max(oc, key=lambda z: overlap(z, lo, hi), default=None)
        oov = overlap(obest, lo, hi) if obest else 0.0
        hit = ov >= 0.5 or oov >= 0.5
        hits += hit
        rows.append((mkt, tf, kind, lo, hi, asof, note, ov, best, oov, obest, hit))
    for mkt, tf, kind, lo, hi, asof, note, ov, best, oov, obest, hit in rows:
        tag = "HIT " if hit else "miss"
        b = f"{best.pattern} {best.lo:g}-{best.hi:g} ({best.base_start.date()} x{best.base_bars}, dep {best.departure_base:.1f}x, tests {best.tests}, score {best.score})" if best else "none"
        o = f"{obest.pattern} {obest.lo:g}-{obest.hi:g} ({obest.tf})" if obest else "none"
        print(f"{tag} {mkt:6s} {tf} {kind:6s} {lo:g}-{hi:g} [{asof}] {note}")
        print(f"       same tf {ov:4.0%}: {b}")
        print(f"       other tf {oov:4.0%}: {o}")
    print(f"\n{hits}/{len(rows)} of Ben's zones overlap >= 50% with a detected zone of the same kind formed before the report date")
    return rows


if __name__ == "__main__":
    main(*sys.argv[1:])
