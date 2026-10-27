"""Standalone scientific figures from resolved evidence."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cnn-projected-mpl')
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/projected_resolution'

def main():
    a=json.loads((OUT/'analysis.json').read_text());s=a['summary']
    styles={'gaussian':('#777777','x','Gaussian'), 'second_order':('#d62728','*','Cumulant correction'),
            'sobol':('#1f77b4','o','Sobol'),'conditional':('#2ca02c','s','Conditional Sobol'),
            'quadratic_cv':('#9467bd','^','Quadratic-CV Sobol')}
    fig,axs=plt.subplots(2,3,figsize=(12,7),sharey=True)
    for row,law in enumerate(('gamma1','laplace')):
        for col,d in enumerate((16,32,64)):
            ax=axs[row,col]
            for method,(color,marker,label) in styles.items():
                ss=sorted((r for r in s if (r['law'],r['d'],r['geometry'],r['method'])==(law,d,'dense',method)),key=lambda r:r['n'])
                ax.loglog([r['seconds']*1000 for r in ss],[r['allowance'] for r in ss],color=color,marker=marker,
                          linestyle='-' if len(ss)>1 else 'None',markersize=9 if method=='second_order' else 4,label=label)
            ax.axhline(.003,color='black',ls=':',lw=1)
            ax.set_title(f'{law}, d={d}');ax.grid(alpha=.2,which='both');ax.set_xlabel('Total time (ms)')
            if col==0:ax.set_ylabel('Relative error + allowance')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Dense-input ReLU kernels: time to accuracy\nDotted line: target 0.003; empirical allowances, not rigorous bounds',fontsize=12)
    fig.tight_layout(rect=(0,.1,1,.92));fig.savefig(OUT/'time_accuracy.png',dpi=180);fig.savefig(OUT/'time_accuracy.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(11,6),sharex=True,sharey=True)
    for row,law in enumerate(('gamma1','laplace')):
        for col,geo in enumerate(('dense','sparse','near')):
            ax=axs[row,col]
            for method in ('gaussian','second_order'):
                color,marker,label=styles[method];ss=sorted((r for r in s if (r['law'],r['geometry'],r['method'])==(law,geo,method)),key=lambda r:r['d'])
                ax.loglog([r['d'] for r in ss],[r['rmse'] for r in ss],color=color,marker=marker,label=label)
            ax.set_title(f'{law}: {geo}');ax.grid(alpha=.2,which='both');ax.set_xlabel('Dimension d')
            ax.set_xticks([4,16,32,64],labels=['4','16','32','64'])
            if col==0:ax.set_ylabel('Relative Frobenius error')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.suptitle('Geometry dependence of the observed approximation error',fontsize=12)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(OUT/'geometry_accuracy.png',dpi=180);fig.savefig(OUT/'geometry_accuracy.pdf');plt.close(fig)
if __name__=='__main__':main()
