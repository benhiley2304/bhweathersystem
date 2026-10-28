"""Forward-return analysis of the BHWS Desk replay (sig.parquet).
Entry: close of the first trading day after signal Friday t. Exit: last close on or before entry + 7h days.
Returns are side-signed. R = return / (trailing 52-week weekly std * sqrt(h)), std measured before t.
CIs: moving-block bootstrap over signal dates (13-week blocks), so cross-sectional and overlap correlation is kept."""
import pickle, json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
M, FO, PX = pickle.load(open('/tmp/bt/data.pkl','rb'))
YF = {m['id']: m.get('yf') for m in M}
SRC = sys.argv[1] if len(sys.argv) > 1 else '/tmp/bt/partial.pkl'
sig = pd.DataFrame([x for y in pickle.load(open(SRC, 'rb')) for x in y])
sig['t'] = pd.to_datetime(sig['t'])
H = [4, 8, 13]
rng = np.random.default_rng(7)

# ── forward returns per (id, t) ──
fw = {}
for mid, g in sig.groupby('id'):
    s = PX.get(YF.get(mid))
    if s is None: continue
    s = s.dropna(); s = s[~s.index.duplicated()]
    idx = s.index.values; v = s.values.astype(float)
    wk = s.resample('W-FRI').last().dropna()
    wr = wk.pct_change()
    vol = wr.rolling(52, min_periods=40).std()
    for t in g['t'].unique():
        t = pd.Timestamp(t)
        i = np.searchsorted(idx, np.datetime64(t + pd.Timedelta(days=1)))
        if i >= len(v): continue
        e_dt = s.index[i]; e = v[i]
        sd = vol[:t].iloc[-1] if len(vol[:t]) else np.nan
        rec = {'sd': sd}
        for h in H:
            j = np.searchsorted(idx, np.datetime64(e_dt + pd.Timedelta(days=7*h)), side='right') - 1
            if j <= i or (s.index[-1] < e_dt + pd.Timedelta(days=7*h)): rec[h] = np.nan
            else: rec[h] = v[j] / e - 1
        fw[(mid, t)] = rec
F = pd.DataFrame.from_dict(fw, orient='index'); F.index = pd.MultiIndex.from_tuples(F.index, names=['id', 't'])
F = F.reset_index()
D = sig.merge(F, on=['id', 't'], how='left')
for h in H:
    D[f'r{h}'] = D[h]                                   # raw (unsigned) market return
    D[f'R{h}raw'] = D[h] / (D['sd'] * np.sqrt(h))       # unsigned, vol-normalised
D = D[D['sd'] > 0]
D['period'] = np.where(D['t'] < '2019-01-01', '2012-2018', '2019-2026')

# market drift (in-sample, per id) for the demeaned robustness check
drift = {}
for h in H:
    drift[h] = D[D.variant == 'fixed'].groupby('id')[f'R{h}raw'].mean()

def signed(df, sidecol, h, demean=False):
    x = df[f'R{h}raw'] * df[sidecol]
    if demean: x = x - df['id'].map(drift[h]) * df[sidecol]
    return x

dates_all = np.array(sorted(D['t'].unique()))
def boot_ci(df, vals, nb=400, block=13):
    """moving-block bootstrap over dates: mean of vals, 90% CI"""
    tmp = pd.DataFrame({'t': df['t'].values, 'x': vals.values}).dropna()
    if len(tmp) < 20: return (np.nan, np.nan)
    by = tmp.groupby('t')['x'].agg(['sum', 'count'])
    by = by.reindex(dates_all).fillna(0)
    S = by['sum'].values; C = by['count'].values; n = len(S)
    nbk = int(np.ceil(n / block)); out = []
    for _ in range(nb):
        st = rng.integers(0, n - block, nbk)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        c = C[ix].sum()
        if c: out.append(S[ix].sum() / c)
    return (float(np.percentile(out, 5)), float(np.percentile(out, 95)))

def stats(df, sidecol='side', h=13, demean=False, ci=True):
    df = df[df[sidecol] != 0].dropna(subset=[f'R{h}raw'])
    if not len(df): return None
    x = signed(df, sidecol, h, demean)
    raw = df[f'r{h}'] * df[sidecol]
    lo, hi = boot_ci(df, x) if ci else (np.nan, np.nan)
    return {'n': int(len(df)), 'mkts': int(df['id'].nunique()), 'meanR': float(x.mean()), 'ci90': [lo, hi],
            'hit': float((raw > 0).mean()), 'mean_ret_pct': float(raw.mean() * 100), 'med_ret_pct': float(raw.median() * 100)}

def episodes(df):
    """first week of a run in the same grade and side for a market"""
    df = df.sort_values(['id', 't']).copy()
    key = df['cls'] + '|' + df['side'].astype(str)
    prev = key.groupby(df['id']).shift(1)
    gap = df.groupby('id')['t'].diff().dt.days.fillna(99)
    return df[(key != prev) | (gap > 8)]

OUT = {'meta': {'start': str(D.t.min().date()), 'end': str(D.t.max().date()), 'markets': int(D.id.nunique()),
                'rows': int(len(D)), 'horizons_weeks': H}}
res = {}
for var in ['fixed', 'asbuilt']:
    V = D[D.variant == var]
    E = episodes(V)
    r = {}
    for h in H:
        g = {}
        for k in ['prime', 'setup', 'developing', 'backdrop', 'none']:
            g[k] = {'weeks': stats(V[V.cls == k], h=h), 'episodes': stats(E[E.cls == k], h=h)}
        g['prime+setup'] = {'weeks': stats(V[V.cls.isin(['prime', 'setup'])], h=h), 'episodes': stats(E[E.cls.isin(['prime', 'setup'])], h=h)}
        # baselines over all market-weeks
        A = V.copy(); A['one'] = 1; A['trend'] = A['tsign'].fillna(0)
        g['base_long_all'] = stats(A, 'one', h)
        g['base_trend_all'] = stats(A, 'trend', h)
        P = A[A.pb.notna()].copy(); P['pbd'] = P['pb']
        g['base_playbook_all'] = stats(P, 'pbd', h)
        # same rows as prime+setup: does the desk side beat trend sign / always-long on identical market-weeks?
        L = A[A.cls.isin(['prime', 'setup'])]
        g['same_rows_trend'] = stats(L, 'trend', h)
        g['same_rows_long'] = stats(L, 'one', h)
        g['prime+setup_demeaned'] = stats(V[V.cls.isin(['prime', 'setup'])], h=h, demean=True)
        g['prime_demeaned'] = stats(V[V.cls == 'prime'], h=h, demean=True)
        r[h] = g
    # breakdowns at 13w (and 8w for archetype)
    LV = V[V.cls.isin(['prime', 'setup'])]
    r['by_arch'] = {a: {h: stats(LV[LV.arch == a], h=h, ci=(h == 13)) for h in [8, 13]} for a in sorted(LV.arch.dropna().unique())}
    r['by_cat'] = {c: stats(LV[LV.cat == c], h=13) for c in sorted(LV.cat.dropna().unique())}
    r['by_confirm'] = {str(c): stats(LV[LV.confirm.astype(str) == str(c)], h=13) for c in LV.confirm.astype(str).unique()}
    r['by_period'] = {p: {k: stats(V[(V.period == p) & (V.cls == k)], h=13) for k in ['prime', 'setup', 'developing']} for p in ['2012-2018', '2019-2026']}
    r['by_side'] = {k: {'long': stats(V[(V.cls == k) & (V.side > 0)], h=13), 'short': stats(V[(V.cls == k) & (V.side < 0)], h=13)} for k in ['prime', 'setup']}
    r['by_tstate'] = {str(s): stats(LV[LV.tstate.astype(str) == str(s)], h=13) for s in LV.tstate.astype(str).unique()}
    r['by_year'] = {}
    for y in range(2012, 2027):
        Y = LV[LV.t.dt.year == y]
        st_ = stats(Y, h=13, ci=False)
        if st_: r['by_year'][y] = st_
    r['counts'] = V.groupby('cls').size().to_dict()
    r['weeks_per_cls_per_date'] = (V.groupby(['t', 'cls']).size().unstack(fill_value=0).mean()).to_dict()
    res[var] = r
OUT['results'] = res
json.dump(OUT, open('/tmp/bt/results.json', 'w'), indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o))
D.to_pickle('/tmp/bt/fwd.pkl')
# compact print
def f(s):
    if not s: return '—'
    return f"n={s['n']:>5} R={s['meanR']:+.3f} [{s['ci90'][0]:+.3f},{s['ci90'][1]:+.3f}] hit={s['hit']*100:4.1f}% ret={s['mean_ret_pct']:+.2f}%"
for var in ['fixed', 'asbuilt']:
    print('=====', var)
    for h in H:
        g = res[var][h]; print('-- h', h)
        for k in ['prime', 'setup', 'developing', 'backdrop', 'prime+setup']:
            print(f"  {k:12s} wk {f(g[k]['weeks'])}\n  {'':12s} ep {f(g[k]['episodes'])}")
        for k in ['base_long_all', 'base_trend_all', 'base_playbook_all', 'same_rows_trend', 'same_rows_long', 'prime+setup_demeaned', 'prime_demeaned']:
            print(f"  {k:22s} {f(g[k])}")
    print(' by arch'); [print('  ', a, '8w', f(v[8]), '| 13w', f(v[13])) for a, v in res[var]['by_arch'].items()]
    print(' by cat'); [print('  ', a, f(v)) for a, v in res[var]['by_cat'].items()]
    print(' by confirm'); [print('  ', a, f(v)) for a, v in res[var]['by_confirm'].items()]
    print(' by period'); [print('  ', p, k, f(v)) for p, d in res[var]['by_period'].items() for k, v in d.items()]
    print(' by side'); [print('  ', k, s, f(v)) for k, d in res[var]['by_side'].items() for s, v in d.items()]
    print(' by tstate'); [print('  ', k, f(v)) for k, v in res[var]['by_tstate'].items()]
    print(' by year'); [print('  ', k, f(v)) for k, v in res[var]['by_year'].items()]
