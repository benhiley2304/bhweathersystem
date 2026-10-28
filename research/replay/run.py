"""Point-in-time weekly replay of the BHWS Desk rules (bhws_core.strategy via desk.build_row).
Each Friday t: COT rows observed <= t-3d, prices <= t, seasonal years < year(t).
Macro: not reproducible point-in-time -> abstains (None). Entry: close of first trading day after t."""
import sys, pickle, json, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/user/workspace/weather-station-railway')
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
import bhws_desk_api as dapi, bhws_evidence_api as evapi
from bhws_core import desk, strategy
from multiprocessing import Pool
import functools
from bhws_core import cot_calendar as _cc
_cc.release_for=functools.lru_cache(maxsize=None)(_cc.release_for)
evapi.cot_calendar.release_for=_cc.release_for
MARKETS, FO, PX = pickle.load(open('/tmp/bt/data.pkl','rb'))
SEAS = json.load(open('/home/user/workspace/weather-station-railway/seasonality_dynamic.json'))
HEAVY={"ES","NQ","YM","RTY","6E","6B","6J","6A","6C","6S","6N","6M","DX","ZB","ZN","ZF","ZT"}
CYC=lambda y:['election','post_election','midterm','pre_election'][y%4]
VARIANT = sys.argv[1] if len(sys.argv)>1 else 'fixed'
START = pd.Timestamp(sys.argv[2] if len(sys.argv)>2 else '2012-01-06')

def seas_fwd(mid, t):
    e=SEAS.get(mid)
    if not e or not e.get('years'): return None
    cat=next((m.get('category') for m in MARKETS if m['id']==mid),None)
    cw=2.5 if mid in HEAVY else 1.5 if cat=='fx_cross' else 1.0
    Y=t.year
    W={y:0.5**((Y-int(y))/15)*(cw if CYC(int(y))==CYC(Y) else 1) for y in e['years'] if int(y)<Y}
    td=max(1,round(t.dayofyear/365*252))
    return strategy.seasonal_forward(e['years'],W,td,Y)

def engine_for(d, variant):
    c=d.values.astype(float)
    if len(c)<260: return {}
    cur=c[-1]; hi=c[-252:].max(); lo=c[-252:].min()
    pr=(cur-lo)/(hi-lo)*100 if hi>lo else 50.0
    sma=c[-200:].mean(); spd=(cur-sma)/sma*100
    wk=d.resample('W-FRI').last().dropna().values.astype(float)
    r=np.diff(wk)/wk[:-1]; cap=np.nanpercentile(np.abs(r),98); r=np.clip(r,-cap,cap)
    wc=wk[0]*np.concatenate([[1.0],np.cumprod(1+r)])
    lt=(wc[-1]/wc[-27]-1)*100; st=(wc[-1]/wc[-5]-1)*100; mt=(wc[-1]/wc[-14]-1)*100
    seg=wc[-27:]; er=abs(seg[-1]-seg[0])/np.sum(np.abs(np.diff(seg)))
    tl=1 if lt>2 else -1 if lt<-2 else 0
    if tl==0: tl=1 if spd>3 else -1 if spd<-3 else 0
    ts_=1 if st>1 else -1 if st<-1 else 0
    if tl>0: ts='Uptrend · short-term pullback' if ts_<0 else 'Uptrend · momentum aligned' if ts_>0 else 'Uptrend · short-term flat'
    elif tl<0: ts='Downtrend · short-term bounce' if ts_>0 else 'Downtrend · momentum aligned' if ts_<0 else 'Downtrend · short-term flat'
    else: ts='Rangebound · no clear trend'
    if variant=='asbuilt':   # daily-bar misnomers as the live engine labels them
        r4=(c[-1]/c[-5]-1)*100; r13=(c[-1]/c[-14]-1)*100; r26=(c[-1]/c[-27]-1)*100
    else:
        r4,r13,r26=st,mt,lt
    return {"trend_state":ts,"scores":{"momentum":{"detail":{"pct_range":pr,"roc4w_pct":r4,"roc13w_pct":r13,"roc26w_pct":r26,
            "sma200_pct_diff":spd,"efficiency_ratio":er}}}}

def run_market(m):
    mid=m['id']; out=[]
    daily_all=PX.get(m.get('yf'))
    if daily_all is None: return out
    fo_all=FO.get(m.get('cftc_code')) if m.get('cftc_code') else None
    fridays=pd.date_range(START,'2026-10-23',freq='W-FRI')
    refs={k:PX.get(t) for k,t in dapi.REFS.items()}
    for t in fridays:
        d=daily_all[:t]
        if len(d)<300 or (t-d.index[-1]).days>6: continue
        d=d.iloc[-1300:]
        cut=t-pd.Timedelta(days=900)
        px={m['yf']:d[cut:]}
        for k,s in refs.items():
            if s is not None: px[dapi.REFS[k]]=s[cut:t]
        fo=None
        if fo_all is not None:
            fo=fo_all[pd.to_datetime(fo_all.date)<=t-pd.Timedelta(days=3)]
            fo=fo.iloc[-600:].reset_index(drop=True)
            if len(fo)<60: fo=None
        now=(t+pd.Timedelta(hours=21)).to_pydatetime().replace(tzinfo=timezone.utc)
        engs={v:engine_for(d,v) for v in ('fixed','asbuilt')}
        ev=None; ch=None
        if fo is not None:
            try: ev=evapi.build_evidence(m,fo,now.isoformat(),d,now.isoformat(),engs['fixed'],now.isoformat(),None,now)
            except Exception as e: ev=None
            ch=desk.positioning_changes(fo.mgr_net.tolist(),fo.comm_net.tolist(),fo.retail_net.tolist())
        try: val=dapi._valuation_for(m,px)
        except Exception: val=None
        try: div=dapi._divergence_for(fo,d)
        except Exception: div=None
        try: sf=seas_fwd(mid,t)
        except Exception: sf=None
        wst=dapi._weekly_state(d)
        for vname,eng in engs.items():
            try:
                row=desk.build_row(market=m,engine=eng,evidence=ev,macro=None,changes=ch,valuation=val,events=[],now=now,
                                   divergence=div,seas_fwd=sf,weekly=wst)
            except Exception as e:
                continue
            pl=row.get('pillars') or {}
            cf=row.get('confirm')
            out.append(dict(variant=vname,id=mid,cat=m.get('category'),t=t.date().isoformat(),cls=row['class']['key'],side=row['side'],
                 lean=row.get('lean'),arch=row.get('archetype'),
                 confirm=(cf.get('state') or cf.get('status') or cf.get('label')) if isinstance(cf,dict) else cf,
                 p=pl.get('positioning',{}).get('rel'),v=pl.get('valuation',{}).get('rel'),tr=pl.get('trend',{}).get('rel'),
                 tstate=pl.get('trend',{}).get('state'),s=pl.get('seasonal',{}).get('rel'),
                 tsign=pl.get('trend',{}).get('sign'),pb=((ev or {}).get('playbook') or {}).get('direction'),
                 pbst=((ev or {}).get('playbook') or {}).get('status'),
                 has_pos=ev is not None and bool((ev or {}).get('positioning')),has_val=val is not None,has_seas=bool(sf)))
    return out

if __name__=='__main__':
    ms=[m for m in MARKETS if m.get('yf')]
    with Pool(2) as p:
        res=[]
        for i,x in enumerate(p.imap_unordered(run_market,ms)):
            res.append(x); print(i,len(x),flush=True)
            pickle.dump(res,open('/tmp/bt/partial.pkl','wb'))
    rows=[r for x in res for r in x]
    pd.DataFrame(rows).to_parquet('/tmp/bt/sig.parquet')
    print(VARIANT,len(rows))
