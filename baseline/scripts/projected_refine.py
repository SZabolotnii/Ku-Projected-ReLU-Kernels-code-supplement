"""Post-primary independent reference refinement with exact quadratic controls."""
from projected_core import *
from projected_study import reference_stats,dump
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]; OLD=ROOT/'evidence/projected_cumulant'; OUT=ROOT/'evidence/projected_refined'

def replicate(x,law,n,seed):
    engine=stats.qmc.Sobol(x.shape[-1],scramble=True,seed=seed); flat=x.reshape(-1,x.shape[-1])
    cov=np.einsum('gik,gjk->gij',x,x); rho=np.clip(cov,-1,1)
    a=(1+rho)/(2*np.sqrt(2*np.pi)); b=np.sqrt(np.maximum(0,1-rho*rho))/(4*np.pi)
    c=.25+np.arcsin(rho)/(2*np.pi)
    k=np.zeros_like(cov); means=np.zeros(x.shape[:-1]); second=np.zeros_like(cov); guards=0
    for start in range(0,n,8192):
        u=engine.random(min(8192,n-start)); guards+=int(np.sum((u==0)|(u==1)))
        z=(transform(u,law)@flat.T).reshape(len(u),*x.shape[:-1]); f=np.maximum(z,0)
        k+=np.einsum('ngi,ngj->gij',f,f,optimize=True)/n
        means+=z.sum(axis=0)/n
        second+=np.einsum('ngi,ngj->gij',z,z,optimize=True)/n
    variance=np.diagonal(second,axis1=-2,axis2=-1)
    correction=a*(means[:,:,None]+means[:,None,:])+b*(variance[:,:,None]+variance[:,None,:]-2)+c*(second-cov)
    return k-correction,guards,k

def main():
    if OUT.exists(): raise RuntimeError('Fresh output required')
    OUT.mkdir(parents=True); diagnostics={}; guards={}; parity={}
    for li,law in enumerate(('gamma1','laplace')):
        for d in (4,16,32,64):
            key=f'{law}_d{d}'; x=make_inputs(d); arrays={}
            for label,power,base in (('coarse',17,510000000),('fine',19,610000000),('upgrade',21,710000000)):
                if label=='upgrade' and all(v['three_se']<=.0001 and v['convergence'] for v in diag): break
                values=[]; raw=[]
                for rep in range(8):
                    seed=base+li*100000+d*100+rep
                    v,count,uncorrected=replicate(x,law,2**power,seed); values.append(v);raw.append(uncorrected)
                    if count: guards[str(seed)]=count
                arrays[label]=np.array(values); arrays[label+'_direct']=np.array(raw)
                if label!='coarse': diag=reference_stats(arrays[label],arrays['coarse' if label=='fine' else 'fine'])
                print(key,label,flush=True)
            use='upgrade' if 'upgrade' in arrays else 'fine'; arrays.update(mean=arrays[use].mean(axis=0),inputs=x)
            original=np.load(OLD/f'reference_{key}.npz'); old=original['upgrade' if 'upgrade' in original else 'fine']
            parity[key]=reference_stats(arrays[use],old)
            diagnostics[key]=diag; np.savez_compressed(OUT/f'reference_{key}.npz',**arrays)
            dump(OUT/'reference_diagnostics.json',diagnostics); dump(OUT/'direct_parity.json',parity)
    dump(OUT/'endpoint_guards.json',guards)
    files=list(OUT.glob('*'))+[Path(__file__),ROOT/'PROJECTED_REFERENCE_REFINEMENT_SPEC.md',ROOT/'scripts/projected_core.py',ROOT/'scripts/projected_study.py']
    dump(OUT/'manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print('REFINEMENT DONE',flush=True)
if __name__=='__main__': main()
