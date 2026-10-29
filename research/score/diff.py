import pandas as pd, numpy as np
D=pd.read_pickle('scored2.pkl'); dates=np.array(sorted(D.t.unique())); rng=np.random.default_rng(11)
def bdiff(a,b,col,nb=500,block=13):
    A=a.dropna(subset=[col]).groupby('t')[col].agg(['sum','count']).reindex(dates).fillna(0)
    B=b.dropna(subset=[col]).groupby('t')[col].agg(['sum','count']).reindex(dates).fillna(0); n=len(dates); out=[]
    for _ in range(nb):
        st=rng.integers(0,n-block,int(np.ceil(n/block))); ix=(st[:,None]+np.arange(block)).ravel()[:n]
        out.append(A['sum'].values[ix].sum()/max(1,A['count'].values[ix].sum())-B['sum'].values[ix].sum()/max(1,B['count'].values[ix].sum()))
    return round(float(np.mean(out)),3),np.percentile(out,[5,95]).round(3).tolist()
for per in ['2012-2018','2019-2026']:
    g=D[D.period==per]
    print(per,'score>=6 minus <=3.5  YX',bdiff(g[g.score>=6],g[g.score<=3.5],'YX'),' Y',bdiff(g[g.score>=6],g[g.score<=3.5],'Y'))
    w=g[g.seas=='wait']
    w=w.assign(Dm=(w.D4+w.D8)/2)
    print('  wait: now Y %.3f  delayed %.3f  n %d'%(w.Y.mean(),w.Dm.mean(),len(w)), 'diff',bdiff(w.assign(z=w.Dm-w.Y),w.assign(z=0*w.Y),'z'))
    a=g[g.seas=='approaching'].assign(Dm=lambda x:(x.D4+x.D8)/2)
    print('  approaching: now %.3f delayed %.3f'%(a.Y.mean(),a.Dm.mean()))
    ag=g[g.seas=='against'].assign(Dm=lambda x:(x.D4+x.D8)/2); print('  against: now %.3f delayed %.3f n %d'%(ag.Y.mean(),ag.Dm.mean(),len(ag)))
