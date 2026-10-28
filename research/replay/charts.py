import json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
R=json.load(open('/tmp/bt/results.json')); V=R['results']['fixed']
OUT='/home/user/workspace/bhws_backtest/'
BG='#0b0d13'; FG='#e6ebf5'; DIM='#8a93a6'; G='#22C97A'; RD='#E05252'; GOLD='#C8A96E'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':FG,'axes.labelcolor':DIM,'xtick.color':DIM,'ytick.color':FG,
  'axes.edgecolor':'#222736','figure.facecolor':BG,'axes.facecolor':BG,'font.size':10})
def bar(ax,labels,st,title,base_idx=()):
    y=np.arange(len(labels))[::-1]
    for yi,l,s,i in zip(y,labels,st,range(len(labels))):
        if not s: ax.text(0,yi,'no rows',color=DIM,va='center',ha='center',fontsize=8); continue
        m=s['meanR']; lo,hi=s['ci90']; b=i in base_idx
        c=GOLD if b else (G if m>=0 else RD)
        ax.barh(yi,m,height=.42 if b else .6,color=c,alpha=.55 if b else .9)
        if lo is not None and hi is not None: ax.plot([lo,hi],[yi,yi],color='white',lw=1,alpha=.7); ax.plot([lo,lo],[yi-.15,yi+.15],color='white',lw=1,alpha=.7); ax.plot([hi,hi],[yi-.15,yi+.15],color='white',lw=1,alpha=.7)
        ax.text(ax.get_xlim()[1] if False else 0, yi, '', va='center')
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.axvline(0,color='white',lw=.8,alpha=.4)
    ax.set_title(title,loc='left',color=FG,fontsize=11,fontweight='bold'); ax.grid(axis='x',color='#1b2030',lw=.6)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    xl=ax.get_xlim()
    for yi,s in zip(y,st):
        if s: ax.text(xl[1],yi,f"  {s['meanR']:+.2f}  hit {s['hit']*100:.0f}%  n {s['n']}",va='center',fontsize=8,color=DIM,clip_on=False)
# 1: grades at 4/8/13
K=['prime','setup','developing','backdrop','none']
fig,axs=plt.subplots(1,3,figsize=(15,4.6),sharey=True)
for ax,h in zip(axs,['4','8','13']):
    g=V[h]; labels=[k.capitalize() for k in K]+['Always long','Trend sign']
    st=[g[k]['weeks'] for k in K]+[g['base_long_all'],g['base_trend_all']]
    bar(ax,labels,st,f'{h}-week hold',base_idx=(5,6))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4)); ax.set_xlabel('mean side-signed return / (52w weekly vol × √h)')
fig.suptitle('BHWS Desk replay 2012–2026: result by grade, 90% block-bootstrap CI',x=.01,ha='left',fontsize=13,fontweight='bold',color=FG)
fig.tight_layout(rect=(0,0,.94,.93)); fig.savefig(OUT+'bt_grades.png',dpi=150); plt.close(fig)
# 2: archetype 13w
A=V['by_arch']; ks=sorted(A,key=lambda a:-(A[a]['13'] or {}).get('n',0))
fig,ax=plt.subplots(figsize=(10,4.4)); bar(ax,ks,[A[a]['13'] for a in ks],'Prime + setup by archetype, 13-week hold')
ax.set_xlabel('vol-normalised R'); fig.tight_layout(rect=(0,0,.86,1)); fig.savefig(OUT+'bt_archetype.png',dpi=150); plt.close(fig)
# 3: by year
Y=V['by_year']; ys=sorted(Y,key=int)
fig,ax=plt.subplots(figsize=(11,3.6)); vals=[Y[y]['meanR'] for y in ys]
ax.bar(ys,vals,color=[G if v>=0 else RD for v in vals],alpha=.9); ax.axhline(0,color='white',lw=.8,alpha=.4)
for i,y in enumerate(ys): ax.text(i,vals[i]+(0.02 if vals[i]>=0 else -0.02),f"n{Y[y]['n']}",ha='center',va='bottom' if vals[i]>=0 else 'top',fontsize=7,color=DIM)
ax.set_title('Prime + setup by signal year, 13-week hold (mean R, no CI)',loc='left',fontweight='bold'); ax.grid(axis='y',color='#1b2030',lw=.6)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT+'bt_years.png',dpi=150); plt.close(fig)
# 4: confirm + category + side
fig,axs=plt.subplots(1,2,figsize=(14,4.2))
C=V['by_confirm']; ck=[k for k in sorted(C,key=lambda k:-(C[k] or {}).get('n',0)) if C[k] and k!='nan' and C[k]['n']>=20]
bar(axs[0],ck,[C[k] for k in ck],'Prime + setup by confirmation state, 13w')
CA=V['by_cat']; cats=sorted(CA); bar(axs[1],cats,[CA[c] for c in cats],'Prime + setup by asset class, 13w')
fig.tight_layout(rect=(0,0,.93,1)); fig.savefig(OUT+'bt_confirm_cat.png',dpi=150); plt.close(fig)
print('charts ok')
