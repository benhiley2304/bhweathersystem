import json
R=json.load(open('/tmp/bt/results.json')); E=json.load(open('/tmp/bt/edge.json')); A=json.load(open('/tmp/bt/asym.json'))
V=R['results']['fixed']; m=R['meta']
def g(s,k=None):
    if not s: return None
    return dict(k=k,n=s['n'],R=round(s['meanR'],3),lo=round(s['ci90'][0],3) if s['ci90'][0] is not None else None,hi=round(s['ci90'][1],3) if s['ci90'][1] is not None else None,hit=round(s['hit'],3))
h=V['13']
grades=[g(h[k]['weeks'],k.capitalize()) for k in ['prime','setup','developing']]
base=[dict(g(h['base_long_all'],'Always long'),base=1),dict(g(h['base_trend_all'],'Trend sign'),base=1)]
cells={}
def put(key,e,a):
    if not e or e['n']<40: return
    c=dict(n=e['n'],mk=e['mk'],R=e['R'],lo=e['lo'],hi=e['hi'],hit=e['hit'],p1=e['p1'],p2=e['p2'])
    if a: c.update(Rd=a['Rd'],mfe=a['mfe'],mae=a['mae'],skew=a['skew'],up=a['up'],dn=a['dn'])
    cells[key]=c
for k,e in E['arch_conf'].items(): put('c:'+k,e,A.get(k))
for k,e in E['arch_side'].items(): put('s:'+k,e,A.get(k))
for k,e in E['arch'].items(): put('a:'+k,e,A.get(k))
for k,e in E['arch_cat'].items(): put('k:'+k,e,None)
flags={}
for key,lab,ek in [('seasonal_against','seasonal against','s|against'),('trend_exhausted','trend exhausted','tr|exhausted'),('positioning_against','positioning against','p|against'),('seasonal_with','seasonal with','s|with')]:
    e=E['pillars'].get(ek); a=A.get(lab)
    if e: flags[key]=dict(n=e['n'],R=e['R'],lo=e['lo'],hi=e['hi'],p1=e['p1'],p2=e['p2'],Rd=(a or {}).get('Rd'),skew=(a or {}).get('skew'))
gr={k:A.get(k) for k in ['prime','setup','developing','ALL graded']}
OUT=dict(span=m['start'][:4]+'–'+m['end'][:4],start=m['start'],end=m['end'],markets=m['markets'],h=13,grades=grades,base=base,
  grade_asym=gr,cells=cells,flags=flags,
  arch=[dict(g(V['by_arch'][a]['13'],a)) for a in sorted(V['by_arch'],key=lambda a:-V['by_arch'][a]['13']['n'])],
  note='Point-in-time weekly replay of the Desk rules on '+str(m['markets'])+' markets, '+m['start']+' to '+m['end']+'. R is the side-signed 13-week return divided by trailing 52-week volatility; bars are 90% block-bootstrap ranges. Macro pillar excluded, continuous futures with roll gaps, no costs. Evidence about the rules, not a forecast for any trade.')
s=json.dumps(OUT,separators=(',',':'))
open('/tmp/bt/edge_front.json','w').write(s); print(len(s)); print(json.dumps(OUT['grades'])); print(json.dumps(OUT['flags']))
