"""Point-in-time seasonal path for every replay signal (id, Friday t).
Years strictly before year(t); weights = 0.5**(age/15) * cycle weight (same as research/replay/run.py).
Output: per (id,t) the weighted-median forward path F(k), k=0..80 TD, in %, plus weighted hit-up per k."""
import pickle, json, numpy as np, pandas as pd
D = pickle.load(open('/tmp/bt/fwd.pkl','rb')); D = D[D.variant=='fixed'].copy()
M, FO, PX = pickle.load(open('/tmp/bt/data.pkl','rb'))
SEAS = json.load(open('/home/user/workspace/weather-station-railway/seasonality_dynamic.json'))
HEAVY={"ES","NQ","YM","RTY","6E","6B","6J","6A","6C","6S","6N","6M","DX","ZB","ZN","ZF","ZT"}
CYC=lambda y:['election','post_election','midterm','pre_election'][y%4]
CAT={m['id']:m.get('category') for m in M}
K=80
def wmed(V, w):
    # V: (n_years, K+1) with nan; w: (n_years,)
    out=np.full(V.shape[1],np.nan); hit=np.full(V.shape[1],np.nan); nn=np.zeros(V.shape[1],int)
    for k in range(V.shape[1]):
        v=V[:,k]; ok=~np.isnan(v)
        if ok.sum()<8: continue
        vv=v[ok]; ww=w[ok]; o=np.argsort(vv); vv=vv[o]; ww=ww[o]; c=np.cumsum(ww)
        out[k]=vv[np.searchsorted(c,0.5*c[-1])]; hit[k]=ww[vv>0].sum()/c[-1]; nn[k]=ok.sum()
    return out,hit,nn
res={}
for mid,g in D.groupby('id'):
    e=SEAS.get(mid)
    if not e or not e.get('years'): continue
    yrs={int(y):np.array([np.nan if x is None else x for x in p],float) for y,p in e['years'].items()}
    cw=2.5 if mid in HEAVY else 1.5 if CAT.get(mid)=='fx_cross' else 1.0
    for t in sorted(g['t'].unique()):
        t=pd.Timestamp(t); Y=t.year
        a=max(1,round(t.dayofyear/365*252))-1
        rows=[];ws=[]
        for y,p in yrs.items():
            if y>=Y or len(p)<=a or np.isnan(p[a]): continue
            ext=1+p/100.0
            if y+1<Y and (y+1) in yrs:
                q=yrs[y+1]; ext=np.concatenate([ext, ext[-1]*(1+q/100.0)])
            seg=ext[a:a+K+1]
            r=np.full(K+1,np.nan); r[:len(seg)]=(seg/ext[a]-1)*100
            rows.append(r); ws.append(0.5**((Y-y)/15)*(cw if CYC(y)==CYC(Y) else 1))
        if len(rows)<8: continue
        F,H,N=wmed(np.array(rows),np.array(ws))
        res[(mid,t)]=(F.astype(np.float32),H.astype(np.float32),int(len(rows)))
pickle.dump(res,open('/tmp/sx/seas_pit.pkl','wb'))
print(len(res))
