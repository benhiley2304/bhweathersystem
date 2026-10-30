"""Calibration of the second detector against the nine zones Ben drew on 30 Oct 2026 (six TradingView charts).

A hit is an unbroken detected zone of the same kind whose band overlaps his by at least half of the narrower
band, with a base inside his base window. The parameters in bhws_core/zones_v2.py were tuned on these nine
zones, so a 9/9 result is in-sample agreement with hand-drawn boxes, not validation and not payoff evidence.

Data: a pickle of Yahoo daily and weekly frames (58 markets, to 2026-10-30), built by research/zones/replay.py's
loader; set ZONES_OHLC to its path. Run from the repo root: python3 research/zones/validate_v2.py
"""
import os, pickle, sys, pandas as pd, numpy as np
sys.path.insert(0, '.')
from bhws_core import zones_v2 as Z2
D = pickle.load(open(os.environ.get('ZONES_OHLC', '/tmp/zones/ohlc.pkl'), 'rb'))
# Ben's zones (drawn 30 Oct 2026): market, timeframe, kind, lo, hi, base window
BEN=[
 ('SI','W','demand',45.5,50.5,'2025-10-01','2025-12-01'),
 ('SB','W','supply',21.0,23.6,'2024-06-01','2024-08-15'),
 ('BTC','2D','demand',75500,78800,'2026-09-01','2026-10-01'),
 ('BTC','2D','demand',58000,65000,'2026-07-01','2026-09-01'),
 ('HG','D','demand',5.81,6.00,'2026-04-25','2026-05-20'),
 ('USDJPY','D','supply',159.4,160.4,'2026-08-25','2026-09-15'),
 ('USDJPY','D','demand',153.0,154.3,'2026-09-08','2026-09-30'),
 ('6E','2D','demand',1.1085,1.1195,'2025-04-20','2025-06-10'),
 ('6E','2D','demand',1.0755,1.0845,'2025-03-20','2025-05-01'),
]
def frame(mid,tf):
    d=D[mid]
    if tf=='2D': return Z2.resample_2d(d['D'])
    return d[tf]
def ov(a_lo,a_hi,b_lo,b_hi):
    lo,hi=max(a_lo,b_lo),min(a_hi,b_hi)
    if hi<=lo: return 0.0
    return (hi-lo)/min(a_hi-a_lo,b_hi-b_lo)
for mid,tf,kind,lo,hi,d0,d1 in BEN:
    df=frame(mid,tf)
    if tf!="W" and Z2.looks_like_spot_fx_defect(D[mid]["D"]): df=Z2.resample_2d(Z2.repair_spot_fx(D[mid]["D"])) if tf=="2D" else Z2.repair_spot_fx(D[mid]["D"])
    zs=Z2.detect(df,tf)
    price=float(df['Close'].iloc[-1])
    print(f"\n=== {mid} {tf} Ben {kind} {lo}-{hi} base {d0}..{d1}  price {price:.4g}  zones total {len(zs)}")
    hits=[z for z in zs if z.kind==kind and ov(lo,hi,z.lo,z.hi)>=0.5]
    for z in hits:
        print(f"  HIT {z.pattern} {z.lo:.4g}-{z.hi:.4g} base {z.base_start:%Y-%m-%d}..{z.base_end:%Y-%m-%d} bars {z.base_bars} run {z.run_atr:.1f}atr swing {z.swing_h:.1f}h tests {z.tests} pen {z.pen_max:.2f} {z.fresh_state} broken {z.broken} score {z.score} notes {z.notes}")
    if not hits:
        near=[z for z in zs if z.kind==kind and z.base_end>=pd.Timestamp(d0)-pd.Timedelta(days=60) and z.base_start<=pd.Timestamp(d1)+pd.Timedelta(days=60)]
        for z in near: print(f"  near {z.pattern} {z.lo:.4g}-{z.hi:.4g} base {z.base_start:%Y-%m-%d}..{z.base_end:%Y-%m-%d} bars {z.base_bars} run {z.run_atr:.1f} swing {z.swing_h:.1f} {z.fresh_state} sc {z.score}")
        if not near: print("  no zone of that kind near the base window")
    # what would be displayed
    a=Z2.atr(df).iloc[-1]
    lv=Z2.live(zs,price,a)
    shown,rest=Z2.select(Z2.stack(lv,price),price,2,typ=float(Z2.typical_range(df).iloc[-1]))
    print(f"  shown ({len(shown)}): "+'; '.join(f"{z.kind[0].upper()} {z.lo:.4g}-{z.hi:.4g} {z.fresh_state} {z.base_start:%b%y} sc{z.score}" for z in sorted(shown,key=lambda z:-z.hi)))
