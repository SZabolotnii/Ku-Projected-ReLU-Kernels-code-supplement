"""Recompute primary metrics and independently check sparse integration and provenance."""
from projected_core import *
from pathlib import Path
import hashlib,json,csv
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'evidence/projected_cumulant'; REF=ROOT/'evidence/projected_refined'

def sparse_reference(x,law):
    from reuse_core import incomplete
    from kernel_common import Law
    ll=Law(law); ii,jj=np.triu_indices(len(x)); a=x[ii,1]; b=x[jj,1]
    def fn(w):
        zi=x[ii,0]*w; zj=x[jj,0]*w
        lo=np.full(len(ii),ll.lo); hi=np.full(len(ii),ll.hi)
        for c,z in ((a,zi),(b,zj)):
            cuts=np.divide(-z,c,out=np.zeros_like(z),where=c!=0)
            lo=np.where(c>0,np.maximum(lo,cuts),lo); hi=np.where(c<0,np.minimum(hi,cuts),hi)
        valid=(hi>lo)&~(((a==0)&(zi<=0))|((b==0)&(zj<=0)))
        lo=np.clip(lo,ll.lo,ll.hi); hi=np.clip(hi,ll.lo,ll.hi)
        hi=np.maximum(lo,hi)
        m=[incomplete(ll,k,lo,hi) for k in range(3)]
        return ll.pdf(w)*np.where(valid,a*b*m[2]+(a*zj+b*zi)*m[1]+zi*zj*m[0],0)
    # The density and support break at zero; integrate both sides separately.
    v1,e1=integrate.quad_vec(fn,ll.lo,0,epsabs=1e-11,epsrel=1e-10)
    v2,e2=integrate.quad_vec(fn,0,ll.hi,epsabs=1e-11,epsrel=1e-10)
    k=np.zeros((len(x),len(x)));k[ii,jj]=k[jj,ii]=v1+v2
    return k,float(e1+e2)

def main():
    from projected_report import analyze
    checks={}; sparse={}; replay_error=0.; rows=list(csv.DictReader((OUT/'raw.csv').open()))
    for folder in (OUT,REF):
        for path,h in json.loads((folder/'manifest.json').read_text()).items():
            checks[f'hash:{folder.name}:{path}']=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
    checks['169_preflight']=all(json.loads((OUT/'preflight.json').read_text())['checks'].values())
    checks['rows_1632']=len(rows)==1632
    arrays=np.load(OUT/'kernels.npz');timings=json.loads((OUT/'timings.json').read_text())
    oldrefs={}; newrefs={}; diag=json.loads((REF/'reference_diagnostics.json').read_text())
    for law in ('gamma1','laplace'):
        for d in (4,16,32,64):
            cell=f'{law}_d{d}'; oldrefs[cell]=np.load(OUT/f'reference_{cell}.npz');newrefs[cell]=np.load(REF/f'reference_{cell}.npz')
            x=make_inputs(d)
            checks[f'inputs:{cell}']=bool(np.array_equal(oldrefs[cell]['inputs'],x) and np.array_equal(newrefs[cell]['inputs'],x))
            exact,err=sparse_reference(x[1],law);ref=newrefs[cell]['mean'][1]
            difference=float(np.linalg.norm(exact-ref)/np.linalg.norm(exact))
            sparse[cell]=dict(relative_difference=difference,quadrature_error=err,three_se=diag[cell][1]['three_se'])
            checks[f'sparse_independent:{cell}']=difference<=diag[cell][1]['three_se']+err/np.linalg.norm(exact)+1e-9
            # Direct source calculation replay of a reference scramble, with controls.
            from projected_refine import replicate
            rr,_,_=replicate(x,law,2**17,510000000+('gamma1','laplace').index(law)*100000+d*100)
            e=float(np.max(abs(rr-newrefs[cell]['coarse'][0])));replay_error=max(replay_error,e)
            checks[f'reference_replay:{cell}']=e<1e-13
    maxmetric=0.; maxkernel=0.
    for row in rows:
        name=row['key'];k=arrays[name];cell=f"{row['law']}_d{row['d']}";gi=('dense','sparse','near').index(row['geometry'])
        ref=oldrefs[cell]['mean'][gi];norm=np.linalg.norm(ref)
        error=np.linalg.norm(k-ref)/norm;mineig=np.linalg.eigvalsh(k)[0]/norm
        e=max(abs(error-float(row['error'])),abs(mineig-float(row['mineig'])),abs(np.median(timings[name])-float(row['seconds'])))
        maxmetric=max(maxmetric,float(e)); checks[f'metrics:{name}']=bool(e<1e-13)
        if row['method'] in METHODS:
            order=METHODS.index(row['method']);replay=matrices(oldrefs[cell]['inputs'][gi],row['law'],order)[order]
            maxkernel=max(maxkernel,float(np.max(abs(replay-k))))
    checks['all_deterministic_replay']=maxkernel<1e-13
    # Replay one seed of every stochastic budget/method in a representative dense cell.
    for row in rows:
        if (row['law'],row['d'],row['geometry'],row['rep'])!=('gamma1','16','dense','0') or row['method'] in METHODS: continue
        n=int(row['n']);seed=410000000+1600+(5000000 if row['method']=='conditional' else 0);xx=make_inputs(16)[0]
        replay=sobol_kernel(xx,'gamma1',n,seed)[0] if row['method']=='sobol' else conditional_kernel(xx,'gamma1',n,seed)
        checks[f'stochastic_replay:{row["key"]}']=bool(np.max(abs(replay-arrays[row['key']]))<1e-13)
    for refined,folder in ((False,OUT),(True,REF)):
        expected=analyze(refined); actual=json.loads((folder/'analysis.json').read_text())
        checks[f'analysis_recompute:{folder.name}']=json.loads(json.dumps(expected))==actual
    for cell,records in json.loads((REF/'direct_parity.json').read_text()).items():
        for gi,r in enumerate(records):checks[f'direct_refined:{cell}:{gi}']=r['convergence']
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(checks=checks,passed=sum(checks.values()),total=len(checks),sparse_quadrature=sparse,
                maxmetric=maxmetric,maxkernel=maxkernel,reference_replay_error=replay_error,
                note='Technical checks are separate from scientific/practical gates; empirical reference uncertainty is not a rigorous bound.')
    (REF/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"Verification {result['passed']}/{result['total']}; metric {maxmetric}; kernel {maxkernel}")
    for name,ok in checks.items():
        if not ok: print('FAIL',name)
    if not all(checks.values()): raise SystemExit(1)

if __name__=='__main__': main()
