"""Bounded independent post-primary reference extension, frozen methods."""
from projected_generalize import *
OLD=OUT
OUT=ROOT/'evidence/projected_gamma64_refinement'

def main():
    if OUT.exists():raise RuntimeError('Fresh output required')
    OUT.mkdir(parents=True);ds={};guards={}
    for panel in range(3):
        cell=f'gamma1_d64_p{panel}';previous=np.load(OLD/f'reference_{cell}.npz');x=previous['inputs'];coarse=previous['level2'];arrays={}
        for level,power in enumerate((23,24)):
            values=[];raw=[]
            for rep in range(8):
                seed=50000000+panel*100+level*10+rep
                k,g,r=replicate(x[None],'gamma1',2**power,seed)
                values.append(k[0]);raw.append(r[0]);guards[str(seed)]=g
                print(cell,power,'rep',rep+1,flush=True)
            fine=np.array(values);direct=np.array(raw);diag=diagnostics(fine,coarse,direct)
            arrays[f'level{level}']=fine;arrays[f'direct{level}']=direct
            if diag['valid']:break
            coarse=fine
        arrays.update(inputs=x,mean=fine.mean(0));np.savez_compressed(OUT/f'reference_{cell}.npz',**arrays)
        ds[cell]=diag;dump(OUT/'diagnostics.json',ds);print(cell,diag,flush=True)
    dump(OUT/'endpoint_guards.json',guards)
    files=list(OUT.glob('*'))+[Path(__file__),ROOT/'PROJECTED_GAMMA64_REFINEMENT_SPEC.md',OLD/'manifest.json',OLD/'analysis_manifest.json']+[ROOT/'scripts'/s for s in ('projected_generalize.py','projected_core.py','projected_refine.py','projected_study.py')]
    dump(OUT/'manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
if __name__=='__main__':main()
