"""Independent fixed-panel replication; no tuning of the frozen candidate."""
from projected_core import *
from projected_refine import replicate
from projected_study import dump,timed
from pathlib import Path
import hashlib,json,platform,csv
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/projected_generalization'

def inputs(d,p):
    x=np.random.default_rng(2026102500+100*d+p).normal(size=(32,d))
    return x/np.linalg.norm(x,axis=1)[:,None]

def diagnostics(fine,coarse,direct):
    mean=fine.mean(0); norm=np.linalg.norm(mean)
    v=fine.var(0,ddof=1)/len(fine); cv=coarse.var(0,ddof=1)/len(coarse)
    u=3*np.sqrt(v.sum())/norm; diff=np.linalg.norm(mean-coarse.mean(0))/norm
    combined=3*np.sqrt((v+cv).sum())/norm
    delta=fine-direct
    return dict(u=float(u),difference=float(diff),combined=float(combined),valid=bool(u<=.0003 and diff<=combined),
        paired_direct_difference=float(np.linalg.norm(delta.mean(0))/norm),
        paired_direct_three_se=float(3*np.sqrt(delta.var(0,ddof=1).sum()/len(delta))/norm))

def main():
    if OUT.exists():raise RuntimeError('Fresh output required')
    OUT.mkdir(parents=True); rows=[]; values={}; times={}; ds={}; guards={}
    for li,law in enumerate(('gamma1','laplace')):
        for d in (16,64,128):
            for panel in range(3):
                cell=f'{law}_d{d}_p{panel}';x=inputs(d,panel); refs={}
                for level,power in enumerate((18,20,22)):
                    if level==2 and diag['valid']:break
                    vv=[];raw=[]
                    for rep in range(8):
                        seed=10000000+li*1000000+d*1000+panel*100+level*10+rep
                        k,g,r=replicate(x[None],law,2**power,seed)
                        vv.append(k[0]);raw.append(r[0]);guards[str(seed)]=g
                    refs[f'level{level}']=np.array(vv);refs[f'direct{level}']=np.array(raw)
                    if level:diag=diagnostics(refs[f'level{level}'],refs[f'level{level-1}'],refs[f'direct{level}'])
                    print(cell,'reference',power,diag if level else '',flush=True)
                selected=max(int(k[5:]) for k in refs if k.startswith('level'))
                ref=refs[f'level{selected}'].mean(0);norm=np.linalg.norm(ref)
                refs.update(inputs=x,mean=ref);np.savez_compressed(OUT/f'reference_{cell}.npz',**refs)
                ds[cell]=diag;dump(OUT/'diagnostics.json',ds)
                def add(method,n,rep,k,tt):
                    key=f'{cell}_{method}_{n}_{rep}';values[key]=k;times[key]=tt
                    rows.append(dict(cell=cell,law=law,d=d,panel=panel,method=method,n=n,rep=rep,key=key,
                        error=float(np.linalg.norm(k-ref)/norm),seconds=float(np.median(tt)),mineig=float(np.linalg.eigvalsh(k)[0]/norm)))
                for method,order in (('gaussian',0),('second_order',3)):
                    k,tt=timed(lambda:matrices(x,law,order)[order]);add(method,0,0,k,tt)
                for n in (4096,16384,65536,262144):
                    for rep in range(8):
                        seed=30000000+li*1000000+d*1000+panel*100+rep
                        for method in ('sobol','quadratic_cv'):
                            fn=(lambda:sobol_kernel(x[None],law,n,seed)[0][0]) if method=='sobol' else (lambda:replicate(x[None],law,n,seed)[0][0])
                            k,tt=timed(fn);add(method,n,rep,k,tt)
                print(cell,'methods done',flush=True)
                # Checkpoints retain completed cells if execution is interrupted.
                with (OUT/'raw.csv').open('w') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                np.savez_compressed(OUT/'kernels.npz',**values);dump(OUT/'timings.json',times)
    dump(OUT/'endpoint_guards.json',guards)
    dump(OUT/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=__import__('scipy').__version__,threads=__import__('threadpoolctl').threadpool_info()))
    files=list(OUT.glob('*'))+[Path(__file__),ROOT/'PROJECTED_GENERALIZATION_SPEC.md']+[ROOT/'scripts'/s for s in ('projected_core.py','projected_refine.py','projected_study.py')]
    dump(OUT/'manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print('COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
