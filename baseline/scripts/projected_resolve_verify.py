"""Audit final references, stronger controls, and the resolved gate."""
from projected_core import *
from projected_verify import sparse_reference
from projected_study import reference_stats
from projected_report import analyze
from projected_refine import replicate
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/projected_resolution';OLD=ROOT/'evidence/projected_refined'

def main():
    checks={};sparse={};rowmetrics=0.
    for path,h in json.loads((OUT/'manifest.json').read_text()).items(): checks[f'hash:{path}']=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
    prior=json.loads((OLD/'verification.json').read_text());checks['previous_technical_checks']=all(prior['checks'].values())
    diag=json.loads((OUT/'reference_diagnostics.json').read_text());refs={}
    for law in ('gamma1','laplace'):
        for d in (16,32,64):
            cell=f'{law}_d{d}';path=OUT/f'reference_{cell}.npz'
            refs[cell]=np.load(path if path.exists() else OLD/path.name)
            if d==16:continue
            r=refs[cell];previous=np.load(OLD/path.name)
            recomputed=reference_stats(r['final'],previous['upgrade'])
            checks[f'reference_diagnostics:{cell}']=recomputed==diag[cell]
            exact,err=sparse_reference(r['inputs'][1],law)
            difference=float(np.linalg.norm(exact-r['mean'][1])/np.linalg.norm(exact));u=diag[cell][1]['three_se']
            sparse[cell]=dict(relative_difference=difference,three_se=u,quadrature_error=err)
            checks[f'sparse_exact:{cell}']=bool(difference<=u+err/np.linalg.norm(exact)+1e-9)
            checks[f'mean:{cell}']=bool(np.array_equal(r['mean'],r['final'].mean(axis=0)))
            checks[f'inputs:{cell}']=bool(np.array_equal(r['inputs'],make_inputs(d)))
    rows=list(csv.DictReader((OUT/'raw.csv').open()));arrays=np.load(OUT/'kernels.npz');timings=json.loads((OUT/'timings.json').read_text())
    checks['288_stronger_rows']=len(rows)==288
    for row in rows:
        key=row['key'];k=arrays[key];ref=refs[f"{row['law']}_d{row['d']}"]['mean'][0];norm=np.linalg.norm(ref)
        e=max(abs(np.linalg.norm(k-ref)/norm-float(row['error'])),abs(np.linalg.eigvalsh(k)[0]/norm-float(row['mineig'])),abs(np.median(timings[key])-float(row['seconds'])))
        checks[f'metrics:{key}']=bool(e<1e-13);rowmetrics=max(rowmetrics,float(e))
        if row['rep']=='0' and row['n']=='1024':
            li=('gamma1','laplace').index(row['law']);d=int(row['d']);seed=910000000+li*100000+d*100
            replay=replicate(make_inputs(d)[0][None],row['law'],1024,seed)[0][0]
            checks[f'replay:{key}']=bool(np.array_equal(replay,k))
    for cell,items in json.loads((OUT/'direct_parity.json').read_text()).items():
        for i,item in enumerate(items):checks[f'direct_parity:{cell}:{i}']=item['convergence']
    checks['analysis_recompute']=json.loads(json.dumps(analyze(resolved=True)))==json.loads((OUT/'analysis.json').read_text())
    result=dict(checks=checks,passed=sum(checks.values()),total=len(checks),sparse_quadrature=sparse,maxmetric=rowmetrics,
                note='Conditional technical verification; scientific gate and runtime portability are separate.')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Resolved verification',result['passed'],result['total'])
    for k,v in checks.items():
        if not v: print('FAIL',k)
    if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':main()
