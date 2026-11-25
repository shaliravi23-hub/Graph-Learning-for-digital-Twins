import sqlite3, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
# Usage, from this folder: python search_progress.py [path to the stage9_bo output folder]
B=sys.argv[1] if len(sys.argv)>1 else '../stage9_bo'
MODELS=[('dyngem','DynGEM'),('aernn','AERNN'),('vgrnn','VGRNN'),('tgae','TGAE'),('vrdag','VRDAG'),('static_prev','Static VGAE (prev.)'),('tgae_bilinear','Asym. TGAE')]
COL=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6']
SIMPLE={'unsw':(0.8848,'EWMA (L=4)'),'deakin':(0.9510,'History-LR')}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':'#8a8984','axes.linewidth':0.6,
                     'xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b'})
fig,axes=plt.subplots(1,2,figsize=(7.2,3.3),dpi=600)
for ax,(ds,title) in zip(axes,[('unsw','UNSW'),('deakin','Deakin')]):
    ends=[]
    for (m,label),c in zip(MODELS,COL):
        con=sqlite3.connect(f'file:{B}/{ds}/studies/{m}.db?mode=ro',uri=True)
        rows=con.execute("select t.number,t.state,v.value from trials t left join trial_values v on v.trial_id=t.trial_id order by t.number").fetchall()
        best=[];cur=np.nan
        for n,s,v in rows:
            if s=='COMPLETE' and (np.isnan(cur) or v>cur): cur=v
            best.append(cur)
        x=np.arange(len(best))
        ax.step(x,best,where='post',color=c,lw=1.6,solid_capstyle='round')
        ax.plot([0],[best[0]],'o',ms=3.5,color=c,mec='#fcfcfb',mew=0.8,zorder=3)
        ends.append((best[-1],label,c))
    v,lab=SIMPLE[ds]
    ax.hlines(v,0,39.3,color='#52514e',lw=1,ls=(0,(4,3)),zorder=0)
    ends.append((v,f'{lab} (best simple)','#52514e'))
    if ds=='deakin':
        ax.hlines(0.669,0,39.3,color='#8a8984',lw=0.9,ls=(0,(1,2)),zorder=0)
        ends.append((0.669,'Symmetric-decoder ceiling','#52514e'))
    # direct labels at line ends, spread to avoid collisions
    ends.sort(); ys=[e[0] for e in ends]; lo,hi=ax.get_ylim() if False else (None,None)
    span=0.022; pos=[]
    for y in ys:
        p=y if not pos else max(y,pos[-1]+span); pos.append(p)
    for (y,label,c),p in zip(ends,pos):
        ax.plot([39.3,40.3],[y,p],color='#b5b4ad',lw=0.5)
        ax.text(40.6,p,label,fontsize=7,va='center',color='#52514e' if ('best simple' in label or 'ceiling' in label) else '#0b0b0b')
    ax.set_xlim(-1,62); ax.set_xticks([0,10,20,30,39]); ax.set_xticklabels(['0\n(published)','10','20','30','39'])
    ax.set_title(title,fontsize=9,loc='left',color='#0b0b0b',pad=4)
    ax.set_xlabel('Trial'); ax.grid(axis='y',color='#e6e5e0',lw=0.6); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
axes[0].set_ylabel('Best validation F1 so far (seed 42)')
axes[0].set_ylim(0.66,0.93); axes[1].set_ylim(0.55,0.98)
fig.tight_layout(w_pad=1.5)
fig.savefig('search_progress.png',dpi=600,facecolor='#ffffff')
print('ok')
