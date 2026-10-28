import pickle, json, numpy as np, pandas as pd
M,FO,PX=pickle.load(open('/tmp/bt/data.pkl','rb')); YF={m['id']:m.get('yf') for m in M}
D=pd.read_pickle('/tmp/bt/fwd.pkl'); V=D[(D.variant=='fixed')&(D.side!=0)&D.cls.isin(['prime','setup','developing'])].dropna(subset=['R13raw']).copy()
# MFE / MAE over 13 weeks, in units of sd*sqrt(13)
mfe=np.full(len(V),np.nan); mae=np.full(len(V),np.nan)
for mid,g in V.groupby('id'):
    s=PX[YF[mid]].dropna(); s=s[~s.index.duplicated()]; idx=s.index.values; v=s.values.astype(float)
    for k,(i_,row) in enumerate(g.iterrows()):
        i=np.searchsorted(idx,np.datetime64(row.t+pd.Timedelta(days=1)))
        j=np.searchsorted(idx,np.datetime64(s.index[i]+pd.Timedelta(days=91)),side='right')-1
        p=v[i:j+1]/v[i]-1; p=p*row.side; u=row.sd*np.sqrt(13)
        pos=V.index.get_loc(i_); mfe[pos]=p.max()/u; mae[pos]=-p.min()/u
V['mfe']=mfe; V['mae']=mae
drift=D[D.variant=='fixed'].groupby('id')['R13raw'].mean(); V['Rd']=(V.R13raw-V.id.map(drift))*V.side; V['R']=V.R13raw*V.side
V.to_pickle('/tmp/bt/asym.pkl')
def s(g):
    if len(g)<30: return None
    a=g[g.period=='2012-2018']; b=g[g.period=='2019-2026']
    return dict(n=len(g),R=round(g.R.mean(),3),Rd=round(g.Rd.mean(),3),p1=round(a.R.mean(),3) if len(a)>9 else None,p2=round(b.R.mean(),3) if len(b)>9 else None,
      mfe=round(g.mfe.median(),2),mae=round(g.mae.median(),2),skew=round(g.mfe.median()/max(g.mae.median(),1e-6),2),
      up=round((g.R>1).mean(),3),dn=round((g.R<-1).mean(),3))
rows=[]
for name,mask in {'ALL graded':V.index==V.index,'prime':V.cls=='prime','setup':V.cls=='setup','developing':V.cls=='developing'}.items(): rows.append((name,s(V[mask])))
for a,g in V.groupby('arch'): rows.append((a,s(g)))
for (a,c),g in V.groupby(['arch',V.confirm.astype(str)]): rows.append((a+'|'+c,s(g)))
for (a,sd),g in V.groupby(['arch','side']): rows.append((a+'|'+('long' if sd>0 else 'short'),s(g)))
rows.append(('seasonal against',s(V[V.s.astype(str)=='against']))); rows.append(('seasonal with',s(V[V.s.astype(str)=='with'])))
rows.append(('trend exhausted',s(V[V.tr.astype(str)=='exhausted'])))
rows.append(('positioning against',s(V[V.p.astype(str)=='against'])))
for n,r in rows:
    if r: print(f"{n:38s} n={r['n']:>5} R={r['R']:+.3f} Rd={r['Rd']:+.3f} p1={r['p1']} p2={r['p2']} MFE={r['mfe']} MAE={r['mae']} skew={r['skew']} P(R>1)={r['up']} P(R<-1)={r['dn']}")
json.dump({n:r for n,r in rows if r},open('/tmp/bt/asym.json','w'),indent=1,default=float)
