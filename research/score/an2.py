import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
M,FO,PX=pickle.load(open('/tmp/bt/data.pkl','rb')); YF={m['id']:m.get('yf') for m in M}
D=pd.read_pickle('feat.pkl'); D=D[D.side!=0].dropna(subset=['e_side']).copy()
# delayed-entry returns
cache={}
def ser(mid):
    if mid not in cache:
        s=PX.get(YF.get(mid)); s=s.dropna(); s=s[~s.index.duplicated()]; cache[mid]=(s.index.values,s.values.astype(float))
    return cache[mid]
def ret(mid,t,lag,h):
    idx,v=ser(mid); i=np.searchsorted(idx,np.datetime64(pd.Timestamp(t)+pd.Timedelta(days=1)))+int(lag)
    if i>=len(v): return np.nan
    j=np.searchsorted(idx,idx[i]+np.timedelta64(7*h,'D'),side='right')-1
    if idx[-1]<idx[i]+np.timedelta64(7*h,'D') or j<=i: return np.nan
    return v[j]/v[i]-1
for h in [4,8]:
    D[f'D{h}']=[ret(m,t,e,h)*s/(sd*np.sqrt(h)) for m,t,e,s,sd in zip(D.id,D.t,D.e_side,D.side,D.sd)]
    D[f'S{h}']=D[f'R{h}raw']*D.side
D.to_pickle('feat2.pkl')
def st(r):
    if r.g40_side<=-1 or (r.g20_side<=-1): base='against'
    else: base=None
    appr = r.e_side>=4 and r.leg_side>=1.5
    if r.g20_side<=-0.4 and appr: return 'against_now_window_ahead'
    if base: return 'against'
    if r.e_side<=3 and r.nowleg_side>=1.0: return 'in_window'
    if appr: return 'approaching'
    if r.g20_side>=0.4: return 'with_mild'
    return 'flat'
D['state']=D.apply(st,axis=1)
g=D.groupby(['state','period']).agg(n=('S4','size'),e=('e_side','median'),S4=('S4','mean'),S8=('S8','mean'),D4=('D4','mean'),D8=('D8','mean'),hit8=('S8',lambda x:(x>0).mean())).round(3)
print(g.unstack('period').to_string())
