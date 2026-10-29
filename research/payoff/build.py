"""Payoff profile engine ("does it pay"): first-passage and excursion statistics for every setup class
in the point-in-time Desk replay (research/replay). Inputs: /tmp/bt/fwd.pkl (signals + forward returns) and
/tmp/bt/data.pkl (daily closes). Output: research/payoff/payoff_profiles.json, served by the backend.

Units: one sigma = the market's trailing 52-week weekly standard deviation measured before the signal.
Paths are side-signed and measured on daily closes from the entry close (first trading day after the signal Friday)
for 13 weeks (91 calendar days). Stops and targets are evaluated on closes, so intraday hits are not counted.
Everything here is replay history for a class of setups. None of it is a probability for a live trade."""
import pickle, json, sys, math, numpy as np, pandas as pd, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/user/workspace/weather-station-railway/research/payoff/payoff_profiles.json'
M, FO, PX = pickle.load(open('/tmp/bt/data.pkl', 'rb')); YF = {m['id']: m.get('yf') for m in M}
D = pd.read_pickle('/tmp/bt/fwd.pkl')
V = D[(D.variant == 'fixed') & (D.side != 0) & D.cls.isin(['prime', 'setup', 'developing'])].dropna(subset=['R13raw']).copy()
V = V[V.sd > 0].reset_index(drop=True)
KS = [0.5, 0.75, 1.0, 1.5]
HW = {1: 7, 2: 14, 4: 28, 8: 56, 13: 91}
MIN_N = 40

# ── per-signal paths ──
recs = []
for mid, g in V.groupby('id'):
    s = PX.get(YF.get(mid))
    if s is None: continue
    s = s.dropna(); s = s[~s.index.duplicated()]; idx = s.index.values; v = s.values.astype(float)
    mu_d = float(np.nanmean(np.diff(v) / v[:-1]))          # in-sample mean daily return, for the drift check
    for k_, row in g.iterrows():
        i = np.searchsorted(idx, np.datetime64(row.t + pd.Timedelta(days=1)))
        if i >= len(v): continue
        e_dt = s.index[i]
        j = np.searchsorted(idx, np.datetime64(e_dt + pd.Timedelta(days=91)), side='right') - 1
        if j <= i or s.index[-1] < e_dt + pd.Timedelta(days=91): continue
        seg = v[i:j + 1]; days = (s.index[i:j + 1] - e_dt).days.values
        p = row.side * (seg / seg[0] - 1) / row.sd                         # sigma units, side-signed
        tdays = np.arange(len(seg))
        p_adj = row.side * ((seg / seg[0] - 1) - mu_d * tdays) / row.sd    # drift removed
        rec = dict(idx=k_, id=mid, cat=row['cat'], t=row.t, arch=row.arch, side=int(row.side), confirm=str(row.confirm),
                   cls=row.cls, period=row.period, s_rel=str(row.s), tr_rel=str(row.tr), p_rel=str(row.p))
        for h, cd in HW.items():
            m_ = days <= cd
            rec[f'mfe{h}'] = float(p[m_].max()); rec[f'mae{h}'] = float(-p[m_].min())
        rec['r13'] = float(p[-1])
        m2 = days <= 14
        rec['min2w'] = float(-p[m2].min())
        for k in KS:
            for lab, arr in (('', p), ('d', p_adj)):
                hit_s = np.where(arr[1:] <= -k)[0]; hit_2 = np.where(arr[1:] >= 2 * k)[0]; hit_3 = np.where(arr[1:] >= 3 * k)[0]
                fs = hit_s[0] if len(hit_s) else 10 ** 6; f2 = hit_2[0] if len(hit_2) else 10 ** 6; f3 = hit_3[0] if len(hit_3) else 10 ** 6
                o2 = 'target' if f2 < fs else 'stop' if fs < 10 ** 6 else 'neither'
                o3 = 'target' if f3 < fs else 'stop' if fs < 10 ** 6 else 'neither'
                rec[f'o2{lab}_{k}'] = o2; rec[f'o3{lab}_{k}'] = o3
                rec[f't2{lab}_{k}'] = (days[1:][f2] / 7.0) if o2 == 'target' else np.nan
                if lab == '':
                    # did the path offer a better entry (half the stop distance against) within two weeks, before the target?
                    first_half = np.where(arr[1:] <= -0.5 * k)[0]
                    rec[f'be_{k}'] = bool(len(first_half) and days[1:][first_half[0]] <= 14 and first_half[0] < f2)
                    # R at 13 weeks if neither target nor stop closed the trade, in R units (stop = 1R)
                    rec[f'rn_{k}'] = float(p[-1] / k) if o2 == 'neither' else np.nan
        recs.append(rec)
P = pd.DataFrame(recs)
print('paths', len(P), 'markets', P.id.nunique(), flush=True)

# ── unconditional baseline: every Friday from 2012, both sides, same first-passage rules ──
brec = []
for mid in sorted(P.id.unique()):
    s = PX.get(YF.get(mid))
    if s is None: continue
    s = s.dropna(); s = s[~s.index.duplicated()]; idx = s.index.values; v = s.values.astype(float)
    w = s.resample('W-FRI').last().pct_change(); sdw = w.rolling(52).std().shift(1)
    for t, sd in sdw.items():
        if not (sd > 0) or t < pd.Timestamp('2012-01-01'): continue
        i = np.searchsorted(idx, np.datetime64(t + pd.Timedelta(days=1)))
        if i >= len(v): continue
        e_dt = s.index[i]
        if s.index[-1] < e_dt + pd.Timedelta(days=91): continue
        j = np.searchsorted(idx, np.datetime64(e_dt + pd.Timedelta(days=91)), side='right') - 1
        if j <= i: continue
        seg = v[i:j + 1]; days = (s.index[i:j + 1] - e_dt).days.values
        for side in (1, -1):
            p = side * (seg / seg[0] - 1) / sd
            r = dict(id=mid, side=side, r13=float(p[-1]))
            for h, cd in HW.items():
                m_ = days <= cd; r[f'mfe{h}'] = float(p[m_].max()); r[f'mae{h}'] = float(-p[m_].min())
            for k in KS:
                hit_s = np.where(p[1:] <= -k)[0]; hit_2 = np.where(p[1:] >= 2 * k)[0]; hit_3 = np.where(p[1:] >= 3 * k)[0]
                fs = hit_s[0] if len(hit_s) else 10 ** 6; f2 = hit_2[0] if len(hit_2) else 10 ** 6; f3 = hit_3[0] if len(hit_3) else 10 ** 6
                r[f'o2_{k}'] = 'target' if f2 < fs else 'stop' if fs < 10 ** 6 else 'neither'
                r[f'o3_{k}'] = 'target' if f3 < fs else 'stop' if fs < 10 ** 6 else 'neither'
                r[f't2_{k}'] = (days[1:][f2] / 7.0) if r[f'o2_{k}'] == 'target' else np.nan
                fh = np.where(p[1:] <= -0.5 * k)[0]
                r[f'be_{k}'] = bool(len(fh) and days[1:][fh[0]] <= 14 and fh[0] < f2)
            brec.append(r)
B = pd.DataFrame(brec)
print('baseline paths', len(B), flush=True)
def base_cell(g):
    n = int(len(g)); out = dict(label='Every week, both sides', level=-1, n=n, markets=int(g.id.nunique()), hit13=round(float((g.r13 > 0).mean()), 3),
        p_up1=round(float((g.r13 > 1).mean()), 3), p_dn1=round(float((g.r13 < -1).mean()), 3))
    out['excursion'] = {str(h): dict(mfe=[round(pct(g[f'mfe{h}'], q), 2) for q in (25, 50, 75)], mae=[round(pct(g[f'mae{h}'], q), 2) for q in (25, 50, 75)]) for h in HW}
    for h in HW: out['excursion'][str(h)]['skew'] = round(out['excursion'][str(h)]['mfe'][1] / max(out['excursion'][str(h)]['mae'][1], 1e-6), 2)
    rows = []
    for k in KS:
        o2 = g[f'o2_{k}'].value_counts(normalize=True); o3 = g[f'o3_{k}'].value_counts(normalize=True); tt = g[f't2_{k}'].dropna()
        rows.append(dict(k=k, target2=round(float(o2.get('target', 0)), 3), stop=round(float(o2.get('stop', 0)), 3), neither=round(float(o2.get('neither', 0)), 3),
                         target3=round(float(o3.get('target', 0)), 3), stop3=round(float(o3.get('stop', 0)), 3), neither3=round(float(o3.get('neither', 0)), 3), weeks_to_target=[round(pct(tt, q), 1) for q in (25, 50, 75)], within4=round(float((tt <= 4).sum() / n), 3), within8=round(float((tt <= 8).sum() / n), 3),
                         better_entry_2w=round(float(g[f'be_{k}'].mean()), 3), theory_target2=round(1 / 3, 3), theory_target3=0.25))
    out['first_passage'] = rows
    return out
def pct(x, q): x = x.dropna(); return float(np.percentile(x, q)) if len(x) else None
BASE = {'all': base_cell(B), 'long': base_cell(B[B.side == 1]), 'short': base_cell(B[B.side == -1])}
for ct, g in B.merge(P[['id', 'cat']].drop_duplicates('id'), on='id').groupby('cat'):
    BASE[ct] = base_cell(g)


def pct(x, q): x = x.dropna(); return float(np.percentile(x, q)) if len(x) else None
def cell(g, label, level):
    n = int(len(g))
    if n < MIN_N: return None
    out = dict(label=label, level=level, n=n, markets=int(g.id.nunique()), hit13=round(float((g.r13 > 0).mean()), 3),
               p_up1=round(float((g.r13 > 1).mean()), 3), p_dn1=round(float((g.r13 < -1).mean()), 3),
               r13_med=round(float(g.r13.median()), 2), first=str(g.t.min().date()), last=str(g.t.max().date()))
    exc = {}
    for h in HW:
        exc[str(h)] = dict(mfe=[round(pct(g[f'mfe{h}'], q), 2) for q in (25, 50, 75)], mae=[round(pct(g[f'mae{h}'], q), 2) for q in (25, 50, 75)])
        exc[str(h)]['skew'] = round(exc[str(h)]['mfe'][1] / max(exc[str(h)]['mae'][1], 1e-6), 2)
    out['excursion'] = exc
    rows = []
    for k in KS:
        o2 = g[f'o2_{k}'].value_counts(normalize=True); o3 = g[f'o3_{k}'].value_counts(normalize=True)
        o2d = g[f'o2d_{k}'].value_counts(normalize=True)
        a = g[g.period == '2012-2018']; b = g[g.period == '2019-2026']
        tt = g[f't2_{k}'].dropna()
        rows.append(dict(k=k, target2=round(float(o2.get('target', 0)), 3), stop=round(float(o2.get('stop', 0)), 3), neither=round(float(o2.get('neither', 0)), 3),
                         target3=round(float(o3.get('target', 0)), 3), stop3=round(float(o3.get('stop', 0)), 3), neither3=round(float(o3.get('neither', 0)), 3), target2_drift=round(float(o2d.get('target', 0)), 3), stop_drift=round(float(o2d.get('stop', 0)), 3),
                         target2_h1=(round(float((a[f'o2_{k}'] == 'target').mean()), 3) if len(a) >= 20 else None),
                         target2_h2=(round(float((b[f'o2_{k}'] == 'target').mean()), 3) if len(b) >= 20 else None),
                         stop_h1=(round(float((a[f'o2_{k}'] == 'stop').mean()), 3) if len(a) >= 20 else None),
                         stop_h2=(round(float((b[f'o2_{k}'] == 'stop').mean()), 3) if len(b) >= 20 else None),
                         weeks_to_target=[round(pct(tt, q), 1) for q in (25, 50, 75)] if len(tt) >= 10 else None,
                         within4=round(float((tt <= 4).sum() / n), 3), within8=round(float((tt <= 8).sum() / n), 3),
                         better_entry_2w=round(float(g[f'be_{k}'].mean()), 3),
                         neither_r_med=(round(float(g[f'rn_{k}'].median()), 2) if g[f'rn_{k}'].notna().sum() >= 10 else None)))
    out['first_passage'] = rows
    return out

SIDE = {1: 'long', -1: 'short'}
cells = {}
def put(key, g, label, level):
    c = cell(g, label, level)
    if c: cells[key] = c
put('all', P, 'All graded setups', 0)
for a, g in P.groupby('arch'):
    put(f'{a}', g, a, 1)
    for sd, g2 in g.groupby('side'):
        put(f'{a}|{SIDE[sd]}', g2, f'{a}, {SIDE[sd]}', 2)
        for cf, g3 in g2.groupby('confirm'):
            put(f'{a}|{SIDE[sd]}|{cf}', g3, f'{a}, {SIDE[sd]}, {cf.lower()}', 3)
            for ct, g4 in g3.groupby('cat'):
                put(f'{a}|{SIDE[sd]}|{cf}|{ct}', g4, f'{a}, {SIDE[sd]}, {cf.lower()}, {ct}', 4)
    for cf, g3 in g.groupby('confirm'):
        put(f'{a}|*|{cf}', g3, f'{a}, {cf.lower()}, both sides', 2)
# the market's own history (lower threshold, labelled)
own = {}
for (mid, a), g in P.groupby(['id', 'arch']):
    if len(g) >= 15:
        c = cell(g.assign(), f'{mid} {a}', 9)
        if c is None:
            n = int(len(g)); c = dict(label=f'{mid} {a}', level=9, n=n, markets=1, hit13=round(float((g.r13 > 0).mean()), 3), excursion={str(h): dict(mfe=[round(pct(g[f'mfe{h}'], q), 2) for q in (25, 50, 75)], mae=[round(pct(g[f'mae{h}'], q), 2) for q in (25, 50, 75)]) for h in HW},
                     first_passage=[dict(k=k, target2=round(float((g[f'o2_{k}'] == 'target').mean()), 3), stop=round(float((g[f'o2_{k}'] == 'stop').mean()), 3), neither=round(float((g[f'o2_{k}'] == 'neither').mean()), 3), target3=round(float((g[f'o3_{k}'] == 'target').mean()), 3), stop3=round(float((g[f'o3_{k}'] == 'stop').mean()), 3)) for k in KS])
        own[f'{mid}|{a}'] = c
        for sd, g2 in g.groupby('side'):
            if len(g2) >= 15:
                c2 = cell(g2, f'{mid} {a}, {SIDE[sd]}', 9) or dict(label=f'{mid} {a}, {SIDE[sd]}', level=9, n=int(len(g2)), markets=1, hit13=round(float((g2.r13 > 0).mean()), 3),
                     excursion={str(h): dict(mfe=[round(pct(g2[f'mfe{h}'], q), 2) for q in (25, 50, 75)], mae=[round(pct(g2[f'mae{h}'], q), 2) for q in (25, 50, 75)]) for h in HW},
                     first_passage=[dict(k=k, target2=round(float((g2[f'o2_{k}'] == 'target').mean()), 3), stop=round(float((g2[f'o2_{k}'] == 'stop').mean()), 3), neither=round(float((g2[f'o2_{k}'] == 'neither').mean()), 3), target3=round(float((g2[f'o3_{k}'] == 'target').mean()), 3), stop3=round(float((g2[f'o3_{k}'] == 'stop').mean()), 3)) for k in KS])
                own[f'{mid}|{a}|{SIDE[sd]}'] = c2
# context modifiers on the whole graded set, at k=1.0 (reported, not applied)
mods = {}
for name, mask in {'seasonal against': P.s_rel == 'against', 'seasonal with': P.s_rel == 'with', 'trend exhausted against': P.tr_rel == 'exhausted',
                   'positioning against': P.p_rel == 'against', 'prime': P.cls == 'prime', 'setup': P.cls == 'setup', 'developing': P.cls == 'developing'}.items():
    c = cell(P[mask], name, 5)
    if c: mods[name] = c
meta = dict(built=pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%MZ'), signals=int(len(P)), markets=int(P.id.nunique()), first=str(P.t.min().date()), last=str(P.t.max().date()),
            ks=KS, horizons=list(HW.keys()), min_n=MIN_N,
            unit='one sigma = trailing 52-week weekly standard deviation before the signal; paths side-signed on daily closes from the entry close',
            method='Point-in-time Desk replay (research/replay), fixed variant, graded prime/setup/developing rows with a side. Entry: close of the first trading day after the signal Friday. '
                   'Stops at k sigma and targets at 2k and 3k are tested on closes over 91 calendar days. Drift rows remove each market\'s own in-sample mean daily return. No costs, no slippage, no intraday fills.',
            baseline_note='Baseline cells run the same first-passage rules on every Friday since 2012 for both sides, so they show what a stop at k sigma and a target at 2k would have done with no setup at all. For a driftless walk the target-first share is 1/3 at 2R and 1/4 at 3R before the time cap.',
            caveat='Replay history for a class of setups. Not a probability for any live trade. Cells with n under 40 are not reported; market-own cells (level 9) use n >= 15 and are labelled.')
json.dump(dict(meta=meta, cells=cells, own=own, modifiers=mods, baseline=BASE), open(OUT, 'w'), separators=(',', ':'))
P.to_pickle('/tmp/bt/payoff_paths.pkl')
print('cells', len(cells), 'own', len(own), 'mods', len(mods), '->', OUT)
# quick print of the headline classes at k=1
b = BASE['all']['first_passage'][2]; print(f"baseline k=1.0 2R-first={b['target2']:.3f} stop-first={b['stop']:.3f} neither={b['neither']:.3f} 3R={b['target3']:.3f} be={b['better_entry_2w']:.2f}")
for key in ['all', 'Crowd fade|long|Not turned', 'Crowd fade|long|Turned', 'Counter-trend|long', 'Counter-trend|long|Waiting', 'Pullback in trend|long|Confirmed', 'Divergence fade|short', 'Range|long', 'Exhaustion fade|short']:
    c = cells.get(key)
    if not c: print(key, 'n/a'); continue
    r = [x for x in c['first_passage'] if x['k'] == 1.0][0]
    print(f"{key:40s} n={c['n']:>5} 2R-first={r['target2']:.2f} stop-first={r['stop']:.2f} neither={r['neither']:.2f} 3R={r['target3']:.2f} drift2R={r['target2_drift']:.2f} halves={r['target2_h1']}/{r['target2_h2']} wks={r['weeks_to_target']} be={r['better_entry_2w']:.2f} skew13={c['excursion']['13']['skew']}")
