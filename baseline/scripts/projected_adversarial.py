"""Additional budgets and conditional controls, paired session timings."""
from projected_generalize import *
from projected_preflight import run
OLD=OUT; REF=ROOT/'evidence/projected_gamma64_refinement'; OUT=ROOT/'evidence/projected_adversarial'

def main():
    if OUT.exists():raise RuntimeError('Fresh output required')
    OUT.mkdir(parents=True);pre=run();dump(OUT/'preflight.json',pre)
    if not all(pre['checks'].values()):raise RuntimeError('Preflight failed')
    rows=[];arrays={};times={}
    for li,(law,d) in enumerate((('gamma1',64),('laplace',16))):
        for panel in range(3):
            cell=f'{law}_d{d}_p{panel}';x=inputs(d,panel)
            for method,n in [('gaussian',0),('second_order',0)]+[(m,n) for m in ('sobol','quadratic_cv') for n in (32768,131072)]+[('sobol',262144 if law=='gamma1' else 65536)]+[('conditional',n) for n in (1024,4096)]:
                for rep in range(1 if n==0 else 8):
                    seed=(70000000 if method=='conditional' else 30000000)+li*1000000+d*1000+panel*100+rep
                    if method in ('gaussian','second_order'):
                        order=0 if method=='gaussian' else 3;fn=lambda:matrices(x,law,order)[order]
                    elif method=='sobol':fn=lambda:sobol_kernel(x[None],law,n,seed)[0][0]
                    elif method=='quadratic_cv':fn=lambda:replicate(x[None],law,n,seed)[0][0]
                    else:fn=lambda:conditional_kernel(x,law,n,seed)
                    k,tt=timed(fn);key=f'{cell}_{method}_{n}_{rep}';arrays[key]=k;times[key]=tt
                    rows.append(dict(cell=cell,law=law,d=d,panel=panel,method=method,n=n,rep=rep,seed=seed,key=key,seconds=float(np.median(tt))))
                print(cell,method,n,flush=True)
    with (OUT/'raw.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'kernels.npz',**arrays);dump(OUT/'timings.json',times)
    files=list(OUT.glob('*'))+[Path(__file__),ROOT/'PROJECTED_ADVERSARIAL_SPEC.md']+[ROOT/'scripts'/s for s in ('projected_preflight.py','projected_core.py','projected_generalize.py','projected_refine.py','projected_study.py','reuse_fast.py','kernel_common.py')]
    files += [p/f for p in (OLD,REF) for f in ('manifest.json','analysis_manifest.json')]
    dump(OUT/'manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
if __name__=='__main__':main()
