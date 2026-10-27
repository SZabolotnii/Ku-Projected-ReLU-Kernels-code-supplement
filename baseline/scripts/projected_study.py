"""Fresh-output primary experiment; all costs include geometry/node generation."""
from projected_core import *
from pathlib import Path
from time import perf_counter
import json, csv, hashlib, platform
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/projected_cumulant'
GEOMS=('dense','sparse','near')
DIMS=(4,16,32,64)

def dump(path,obj): path.write_text(json.dumps(obj,indent=2)+'\n')

def reference_stats(fine,coarse):
    mean=fine.mean(axis=0); norm=np.linalg.norm(mean,axis=(-2,-1))
    var=fine.var(axis=0,ddof=1)/len(fine); cv=coarse.var(axis=0,ddof=1)/len(coarse)
    u=3*np.sqrt(var.sum(axis=(-2,-1)))/norm
    difference=np.linalg.norm(mean-coarse.mean(axis=0),axis=(-2,-1))/norm
    combined=3*np.sqrt((var+cv).sum(axis=(-2,-1)))/norm
    return [dict(three_se=float(u[i]),difference=float(difference[i]),combined=float(combined[i]),
                 convergence=bool(difference[i]<=combined[i])) for i in range(3)]

def timed(fn):
    fn(); times=[]; value=None
    for _ in range(5):
        start=perf_counter(); value=fn(); times.append(perf_counter()-start)
    return value,times

def main():
    if OUT.exists(): raise RuntimeError('Fresh output required')
    from projected_preflight import run
    pre=run()
    if not all(pre['checks'].values()): raise RuntimeError('Preflight failed')
    OUT.mkdir(parents=True); dump(OUT/'preflight.json',pre)
    rows=[]; arrays={}; times={}; diagnostics={}; guards={}
    for li,law in enumerate(('gamma1','laplace')):
        for d in DIMS:
            x=make_inputs(d); key=f'{law}_d{d}'
            ref={}
            for label,power,offset in (('coarse',15,110000000),('fine',17,210000000),('upgrade',19,310000000)):
                if label=='upgrade' and all(z['three_se']<=.0003 and z['convergence'] for z in diag): break
                values=[]
                for rep in range(8):
                    seed=offset+li*100000+d*100+rep
                    k,count=sobol_kernel(x,law,2**power,seed); values.append(k)
                    if count: guards[str(seed)]=count
                ref[label]=np.array(values)
                if label!='coarse': diag=reference_stats(ref[label],ref['coarse' if label=='fine' else 'fine'])
                print(key,label,flush=True)
            selected=ref['upgrade'] if 'upgrade' in ref else ref['fine']
            mean=selected.mean(axis=0); ref.update(inputs=x,mean=mean)
            np.savez_compressed(OUT/f'reference_{key}.npz',**ref)
            diagnostics[key]=diag; dump(OUT/'reference_diagnostics.json',diagnostics)
            for gi,geometry in enumerate(GEOMS):
                cell=f'{key}_{geometry}'; xx=x[gi]; norm=np.linalg.norm(mean[gi])
                def add(method,n,rep,k,tt):
                    name=f'{cell}_{method}_{n}_{rep}'; arrays[name]=k; times[name]=tt
                    rows.append(dict(law=law,d=d,geometry=geometry,method=method,n=n,rep=rep,
                        error=float(np.linalg.norm(k-mean[gi])/norm),
                        seconds=float(np.median(tt)),mineig=float(np.linalg.eigvalsh(k)[0]/norm),
                        minentry=float(k.min()),key=name))
                for order,method in enumerate(METHODS):
                    k,tt=timed(lambda:matrices(xx,law,order)[order]); add(method,0,0,k,tt)
                for method,ns in (('sobol',(256,1024,4096,16384,65536)),('conditional',(256,1024,4096))):
                    for n in ns:
                        for rep in range(8):
                            seed=410000000+li*100000+d*100+rep+(5000000 if method=='conditional' else 0)
                            fn=(lambda:sobol_kernel(xx,law,n,seed)[0]) if method=='sobol' else (lambda:conditional_kernel(xx,law,n,seed))
                            k,tt=timed(fn); add(method,n,rep,k,tt)
                print(cell,'methods complete',flush=True)
    with (OUT/'raw.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    np.savez_compressed(OUT/'kernels.npz',**arrays); dump(OUT/'timings.json',times); dump(OUT/'endpoint_guards.json',guards)
    dump(OUT/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
                                   scipy=__import__('scipy').__version__,blas=__import__('threadpoolctl').threadpool_info()))
    files=list(OUT.glob('*'))+[Path(__file__),ROOT/'PROJECTED_CUMULANT_SPEC.md']
    files+=[ROOT/'scripts'/n for n in ('projected_core.py','projected_preflight.py','reuse_fast.py','reuse_core.py','positive_core.py','kernel_common.py')]
    dump(OUT/'manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print('DONE',len(rows),flush=True)

if __name__=='__main__': main()
