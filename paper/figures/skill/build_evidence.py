"""Academic-figure-workflow: data-backed plots, never image-generated results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RESULTS=ROOT/'QuantumQUBO Agent/results'
EVIDENCE=RESULTS/'paper_evidence_20261011'
backend=json.loads((EVIDENCE/'backend_statistics.json').read_text())
hybrid=json.loads((EVIDENCE/'dwave_hybrid_statistics.json').read_text())
lookup={(g['backend'],g['case'],g['policy']):g for g in backend['groups']}
lookup.update({('hybrid',g['case'],g['policy']):g for g in hybrid['groups']})
raw={}
for folder in ['paper_backends_20261011','dwave_hybrid_10s_20261011']:
    for p in (RESULTS/'experiments'/folder/'runs').glob('*/result.json'):
        d=json.loads(p.read_text());be='hybrid' if d['backend']=='dwave_hybrid' else d['backend']
        raw.setdefault((be,d['case'],d['policy']),[]).append(d['gap'])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,(a,b)=plt.subplots(1,2,figsize=(7.16,3.2),gridspec_kw={'width_ratios':[1,1.6]})
colors=['#17334a','#087f8c','#b36b12']
succ=np.array([69,72,70]);unknown=np.array([4,6,6]);fail=100-succ-unknown
base=np.zeros(3)
for vals,label,color in [(succ,'Success',colors[1]),(unknown,'Unverifiable','#b8c4cb'),(fail,'Failed',colors[2])]:
    a.barh(np.arange(3),vals,left=base,color=color,height=.42,label=label)
    for i,(v,z) in enumerate(zip(vals,base)):
        if v>8:a.text(z+v/2,i,str(v),ha='center',va='center',color='white',fontsize=9)
    base+=vals
a.set(yticks=range(3),yticklabels=['Run 1','Run 2','Run 3'],xlim=(0,100),ylim=(2.8,-.8),xlabel='Benchmarks (100 per run)')
for i,v in enumerate(unknown):a.text(103,i,f'{v} unverifiable',fontsize=7.5,va='center')
a.set_title('A  Verification outcomes',loc='left',fontsize=10,fontweight='bold',pad=12)
a.legend(loc='upper left',bbox_to_anchor=(-.1,-.29),ncol=3,fontsize=7,frameon=False,columnspacing=.7,handletextpad=.4)

rows=[('cga','wine'),('hybrid','wine'),('hybrid','knapsack60'),('hybrid','select80')]
policies=[('default','Default'),('penalty_tpe','TPE search'),('agent','LLM tuning')]
for j,(key,label) in enumerate(policies):
    for i,(be,case) in enumerate(rows):
        vals=raw[(be,case,key)];assert len(vals)==3 and np.isfinite(vals).all()
        mean=lookup[(be,case,key)]['mean_gap'];assert np.isclose(np.mean(vals),mean,atol=.00006)
        y=i+(j-1)*.23
        b.scatter(vals,y+np.array([-.035,0,.035]),s=13,marker=['o','s','^'][j],facecolors='white',edgecolors=colors[j],linewidths=.7,zorder=3,label=label if i==0 else None)
        b.plot([mean,mean],[y-.09,y+.09],color=colors[j],lw=1.7,zorder=4)
b.set(yticks=range(4),yticklabels=['CGA · wine','Hybrid · wine','Hybrid · knapsack','Hybrid · selection'],ylim=(3.55,-.55),xlim=(-.015,.64),xlabel='Relative gap to exact reference')
b.xaxis.set_major_formatter(PercentFormatter(1));b.grid(axis='x',color='#dde4e8',lw=.5)
b.set_title('B  Fixed-model solving outcomes',loc='left',fontsize=10,fontweight='bold',pad=12)
b.legend(loc='upper left',bbox_to_anchor=(0,-.27),ncol=3,fontsize=7.5,columnspacing=.6,handletextpad=.3,frameon=False)
for ax in [a,b]:ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8)
fig.subplots_adjust(left=.08,right=.98,top=.84,bottom=.31,wspace=.95)
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'evidence.{ext}',dpi=250)
print('Evidence figure uses original benchmark counts and all three actual results per displayed condition.')
