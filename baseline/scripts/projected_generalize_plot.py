"""Plot independent-panel accuracy including unresolved reference gates."""
import os
os.environ['MPLCONFIGDIR']='/tmp/cnn-projected-mpl'
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/projected_generalization'
a=json.loads((OUT/'analysis.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
for ax,law in zip(axes,('gamma1','laplace')):
    for i,d in enumerate((16,64,128)):
        ss=[s for s in a['cells'] if s['law']==law and s['d']==d]
        for p,s in enumerate(ss):
            xx=i+(p-1)*.13
            ax.scatter(xx,s['gaussian']['error'],color='#777777',marker='x',label='Gaussian error' if i==0 and p==0 else None)
            ax.scatter(xx,s['candidate']['bound'],facecolors='none' if s['status']=='REFERENCE_LIMITED' else '#1675aa',edgecolors='#1675aa',label='Correction allowance' if i==0 and p==0 else None)
    ax.axhline(.003,color='#b44230',ls='--',label='Target 0.003')
    ax.set(yscale='log',xticks=[0,1,2],xticklabels=['16','64','128'],xlabel='Input dimension d',title=law)
    ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Relative Frobenius error / allowance')
axes[1].legend(fontsize=8)
fig.suptitle('Three independent panels of 32 inputs per dimension')
fig.text(.5,.01,'Open circles: reference precision gate not met; no certified target decision.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.95])
for ext in ('png','pdf'):fig.savefig(OUT/f'generalization.{ext}',dpi=170)
