import pickle, numpy as np, pandas as pd
D=pickle.load(open('/tmp/bt/fwd.pkl','rb')); D=D[D.variant=='fixed'].copy()
S=pickle.load(open('/tmp/sx/seas_pit.pkl','rb'))
def feats(row_id,t,s,sd):
    v=S.get((row_id,pd.Timestamp(t)))
    if v is None or s==0 or not sd>0: return [np.nan]*9
    F,H,n=v; g=s*F/100.0/sd   # weekly-sd units
    if np.isnan(g[:66]).any(): return [np.nan]*9
    e=int(np.nanargmin(g[:26])); dip=g[e]
    hi=g[e+10:e+41]; leg=np.nanmax(hi)-dip; kpk=e+10+int(np.nanargmax(hi))
    nowleg=np.nanmax(g[5:41])  # from now, best 1-8wk
    hit=s*(H[20]-0.5)*2 if not np.isnan(H[20]) else np.nan
    return [e,dip,leg,kpk,nowleg,g[10],g[20],g[40],hit]
cols=['e','dip','leg','kpk','nowleg','g10','g20','g40','hit20']
out=[]
for side_col in ['side','lean']:
    X=np.array([feats(i,t,s,sd) for i,t,s,sd in zip(D.id,D.t,D[side_col].fillna(0),D.sd)],float)
    for j,c in enumerate(cols): D[f'{c}_{side_col}']=X[:,j]
D.to_pickle('/tmp/sx/feat.pkl'); print(D.shape, D['e_side'].notna().sum(), D['e_lean'].notna().sum())
print(D[[c+'_side' for c in cols]].describe().T)
