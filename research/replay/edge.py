"""Edge matrix from the replay (fixed variant, prime+setup+developing with a side). 13w and 8w R, block-bootstrap CI,
split-period sign check. Writes /tmp/bt/edge.json."""
import pickle, json, numpy as np, pandas as pd, sys
sys.argv=['x']; exec(open('/tmp/bt/analyze.py').read().split("OUT = {'meta'")[0].split("# ── forward returns")[0])
D=pd.read_pickle('/tmp/bt/fwd.pkl'); dates_all=np.array(sorted(D['t'].unique()))
rng=np.random.default_rng(11)
def boot(df,x,nb=300,block=13):
    tmp=pd.DataFrame({'t':df['t'].values,'x':x.values}).dropna()
    if len(tmp)<25: return (None,None)
    by=tmp.groupby('t')['x'].agg(['sum','count']).reindex(dates_all).fillna(0); S=by['sum'].values; C=by['count'].values; n=len(S)
    out=[]
    for _ in range(nb):
        ix=(rng.integers(0,n-block,int(np.ceil(n/block)))[:,None]+np.arange(block)).ravel()[:n]; c=C[ix].sum()
        if c: out.append(S[ix].sum()/c)
    return (float(np.percentile(out,5)),float(np.percentile(out,95)))
def st(df,h=13,ci=True):
    df=df[(df.side!=0)].dropna(subset=[f'R{h}raw'])
    if len(df)<15: return None
    x=df[f'R{h}raw']*df.side; raw=df[f'r{h}']*df.side
    lo,hi=boot(df,x) if ci else (None,None)
    a=df[df.period=='2012-2018']; b=df[df.period=='2019-2026']
    pa=float((a[f'R{h}raw']*a.side).mean()) if len(a)>=10 else None; pb=float((b[f'R{h}raw']*b.side).mean()) if len(b)>=10 else None
    return dict(n=int(len(df)),mk=int(df.id.nunique()),R=round(float(x.mean()),3),lo=None if lo is None else round(lo,3),hi=None if hi is None else round(hi,3),
                hit=round(float((raw>0).mean()),3),ret=round(float(raw.mean()*100),2),p1=None if pa is None else round(pa,3),p2=None if pb is None else round(pb,3))
V=D[D.variant=='fixed']; L=V[V.cls.isin(['prime','setup'])]; LD=V[V.cls.isin(['prime','setup','developing'])]
E={}
E['arch_conf']={f'{a}|{c}':st(g) for (a,c),g in LD.groupby(['arch',LD.confirm.astype(str)])}
E['arch_cat']={f'{a}|{c}':st(g) for (a,c),g in LD.groupby(['arch','cat'])}
E['arch_cls']={f'{a}|{c}':st(g) for (a,c),g in LD.groupby(['arch','cls'])}
E['arch_side']={f'{a}|{"long" if s>0 else "short"}':st(g) for (a,s),g in LD[LD.side!=0].groupby(['arch','side'])}
E['arch']={a:st(g) for a,g in LD.groupby('arch')}
E['arch8']={a:st(g,8) for a,g in LD.groupby('arch')}
E['pillars']={}
for col in ['p','v','tr','s']:
    for val,g in L.groupby(L[col].astype(str)): E['pillars'][f'{col}|{val}']=st(g)
E['nagainst']={}
L2=L.copy(); L2['nag']=(L2[['p','v','tr','s']].astype(str)=='against').sum(1); L2['nwith']=(L2[['p','v','tr','s']].astype(str)=='with').sum(1)
for k,g in L2.groupby('nwith'): E['nagainst'][f'with{k}']=st(g)
for k,g in L2.groupby('nag'): E['nagainst'][f'against{k}']=st(g)
E['tstate']={f'{a}|{t}':st(g) for (a,t),g in LD.groupby(['arch',LD.tstate.astype(str)])}
json.dump({k:{kk:vv for kk,vv in v.items() if vv} for k,v in E.items()},open('/tmp/bt/edge.json','w'),indent=1)
def pr(sec):
    print('==',sec)
    for k,v in sorted(E[sec].items(),key=lambda kv:-(kv[1] or {}).get('n',0)):
        if v: print(f"  {k:45s} n={v['n']:>5} mk={v['mk']:>2} R={v['R']:+.3f} [{v['lo']},{v['hi']}] hit={v['hit']*100:.0f}% p1={v['p1']} p2={v['p2']}")
for s in ['arch','arch8','arch_cls','arch_conf','arch_side','arch_cat','pillars','nagainst','tstate']: pr(s)
