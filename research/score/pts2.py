import pandas as pd, numpy as np, json, warnings, sys; warnings.filterwarnings('ignore')
D=pd.read_pickle('feat3.pkl')
D['ac']=D['ac'].replace({'Pullback in trend|In zone':'Pullback in trend|Confirmed'})
D['seas']=D.state.map({'against':'against','against_now_window_ahead':'wait','approaching':'approaching','flat':'ok','with_mild':'ok','in_window':'ok'})
STEP=float(sys.argv[1]) if len(sys.argv)>1 else 0.03
ins=D.period=='2012-2018'
F=['ac','tr','seas','p','v']; mu=D.loc[ins,'YX'].mean(); PTS={}
for f in F:
    g=D[ins].groupby(f)['YX'].agg(['mean','size']); sh=(g['mean']-mu)*g['size']/(g['size']+400)
    PTS[f]={k:float(v) for k,v in np.clip(np.round(sh/STEP*2)/2,-3,3).items()}
print(json.dumps(PTS))
D['score']=np.clip(5+sum(D[f].map(PTS[f]).fillna(0) for f in F),0,10)
print(D.groupby('period').score.describe().round(2))
D['sb']=pd.cut(D.score,[-1,3.5,5.5,7,10.1],labels=['0-3.5','4-5.5','6-7','7.5-10'])
dates=np.array(sorted(D.t.unique())); rng=np.random.default_rng(7)
def ci(df,col,nb=400,block=13):
    df=df.dropna(subset=[col]); by=df.groupby('t')[col].agg(['sum','count']).reindex(dates).fillna(0); S=by['sum'].values;C=by['count'].values;n=len(S); out=[]
    for _ in range(nb):
        st=rng.integers(0,n-block,int(np.ceil(n/block))); ix=(st[:,None]+np.arange(block)).ravel()[:n]; c=C[ix].sum()
        if c: out.append(S[ix].sum()/c)
    return np.percentile(out,[5,95]).round(3).tolist()
D['S13']=D.R13raw*D.side
res={}
for per in ['2012-2018','2019-2026']:
    print('\n',per); res[per]={}
    for b,g in D[D.period==per].groupby('sb'):
        r=dict(n=len(g),R4=g.S4.mean(),R8=g.S8.mean(),R13=g.S13.mean(),YX=g.YX.mean(),ci=ci(g,'YX'),hit8=(g.S8>0).mean(),mkts=g.id.nunique())
        res[per][b]={k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()}; print(b,res[per][b])
    g=D[D.period==per]; m=g.YX.notna(); print('IC', round(np.corrcoef(g.score[m],g.YX[m])[0,1],4))
    # top vs bottom diff CI
D.to_pickle('scored2.pkl'); json.dump({'pts':PTS,'step':STEP,'res':res},open(f'pts2_{STEP}.json','w'),indent=1)
