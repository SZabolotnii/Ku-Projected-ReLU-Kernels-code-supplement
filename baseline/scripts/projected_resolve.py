"""Final bounded reference extension and stronger polynomial-CV comparator."""
from projected_core import *
from projected_refine import replicate
from projected_study import reference_stats,dump,timed
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[1]; PREV=ROOT/'evidence/projected_refined'; OUT=ROOT/'evidence/projected_resolution'

def main():
    if OUT.exists(): raise RuntimeError('Fresh output required')
    OUT.mkdir(parents=True); diagnostics=json.loads((PREV/'reference_diagnostics.json').read_text()); parity={};guards={}
    for li,law in enumerate(('gamma1','laplace')):
        for d in (32,64):
            cell=f'{law}_d{d}';x=make_inputs(d); values=[];direct=[]
            for rep in range(8):
                seed=810000000+li*100000+d*100+rep
                k,count,raw=replicate(x,law,2**23,seed);values.append(k);direct.append(raw)
                if count:guards[str(seed)]=count
                print(cell,'reference replicate',rep+1,flush=True)
            values=np.array(values); previous=np.load(PREV/f'reference_{cell}.npz');coarse=previous['upgrade']
            diagnostics[cell]=reference_stats(values,coarse)
            original=np.load(ROOT/f'evidence/projected_cumulant/reference_{cell}.npz')
            parity[cell]=reference_stats(values,original['upgrade'])
            np.savez_compressed(OUT/f'reference_{cell}.npz',inputs=x,final=values,mean=values.mean(axis=0),direct=np.array(direct))
            dump(OUT/'reference_diagnostics.json',diagnostics);dump(OUT/'direct_parity.json',parity)
    rows=[];arrays={};times={}
    for li,law in enumerate(('gamma1','laplace')):
        for d in (16,32,64):
            cell=f'{law}_d{d}';x=make_inputs(d)[0][None]
            rp=OUT/f'reference_{cell}.npz'
            if not rp.exists():rp=PREV/rp.name
            ref=np.load(rp)['mean'][0];norm=np.linalg.norm(ref)
            for n in (256,1024,4096,16384,65536,262144):
                for rep in range(8):
                    seed=910000000+li*100000+d*100+rep
                    value,tt=timed(lambda:replicate(x,law,n,seed)[0][0])
                    key=f'{cell}_dense_quadratic_cv_{n}_{rep}';arrays[key]=value;times[key]=tt
                    rows.append(dict(law=law,d=d,geometry='dense',method='quadratic_cv',n=n,rep=rep,
                        error=float(np.linalg.norm(value-ref)/norm),seconds=float(np.median(tt)),
                        mineig=float(np.linalg.eigvalsh(value)[0]/norm),minentry=float(value.min()),key=key))
            print(cell,'strong control complete',flush=True)
    with (OUT/'raw.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'kernels.npz',**arrays);dump(OUT/'timings.json',times);dump(OUT/'endpoint_guards.json',guards)
    files=list(OUT.glob('*'))+[Path(__file__),ROOT/'PROJECTED_RESOLUTION_SPEC.md',ROOT/'scripts/projected_refine.py',ROOT/'scripts/projected_core.py',ROOT/'scripts/projected_study.py',PREV/'manifest.json']
    dump(OUT/'manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print('RESOLUTION DONE',len(rows),flush=True)
if __name__=='__main__':main()
