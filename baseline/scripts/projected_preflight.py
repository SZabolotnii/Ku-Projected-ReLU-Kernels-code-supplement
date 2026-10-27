"""Independent numerical and invariance checks, before the primary study."""
from projected_core import *
import json
from pathlib import Path

def run():
    checks={}; errors=[]
    for r in (-.9999995,-.9,-.2,0.,.5,.99,.9999995):
        s=np.sqrt(1-r*r); vals=hermite_integrals(np.array([r])); worst=0
        for (p,q),v in vals.items():
            def f(u):
                a=-r*u/s
                inner=(r*u*special.ndtr(-a)+s*stats.norm.pdf(a) if q==0 else
                       s*special.ndtr(-a) if q==1 else
                       s*stats.norm.pdf(a)*special.eval_hermitenorm(q-2,a))
                return u*special.eval_hermitenorm(p,u)*stats.norm.pdf(u)*inner
            # Include boundary-layer scale explicitly for almost collinear pairs.
            ref=integrate.quad(f,0,min(1.,20*s),epsabs=1e-11)[0]
            ref+=integrate.quad(f,min(1.,20*s),np.inf,epsabs=1e-11)[0]
            worst=max(worst,abs(v[0]-ref))
        checks[f'quadrature_{r}']=worst<1e-9; errors.append(worst)
    # Independently integrate the inner variable rather than using its closed formula.
    worst=0
    for r in (-.4,.7):
        s=np.sqrt(1-r*r)
        for p,q in ((0,3),(2,2),(3,3),(5,1)):
            def outer(u):
                inner=integrate.quad(lambda v:(r*u+s*v)*special.eval_hermitenorm(q,v)*stats.norm.pdf(v),
                                     -r*u/s,np.inf,epsabs=2e-10)[0]
                return u*special.eval_hermitenorm(p,u)*stats.norm.pdf(u)*inner
            ref=integrate.quad(outer,0,10,epsabs=2e-9)[0]
            worst=max(worst,abs(ref-hermite_integrals(np.array([r]))[p,q][0]))
    checks['nested_direct_quadrature']=worst<1e-8
    for d in (4,16,32,64):
        for gi,x in enumerate(make_inputs(d)):
            for law in CUMULANTS:
                k=matrices(x,law); rev=matrices(x[::-1],law)[:,::-1,::-1]
                checks[f'swap_{d}_{gi}_{law}']=bool(np.max(abs(k-rev))<2e-9)
                checks[f'scale_{d}_{gi}_{law}']=bool(np.max(abs(matrices(2*x,law)-4*k))<1e-10)
                checks[f'finite_{d}_{gi}_{law}']=bool(np.isfinite(k).all())
                checks[f'coordinates_{d}_{gi}_{law}']=bool(np.max(abs(matrices(x[:,::-1],law)-k))<2e-9)
                if gi==2:
                    checks[f'antipode_duplicate_{d}_{law}']=bool(abs(k[:,0,4]).max()<1e-12 and np.max(abs(k[:,0]-k[:,3]))<2e-9)
    x=np.eye(2)
    checks['gaussian_orthogonal']=bool(np.max(abs(matrices(x,'normal')[:,0,1]-1/(2*np.pi)))<1e-14)
    checks['laplace_orthogonal_fourth']=bool(abs(matrices(x,'laplace')[3,0,1]-3/(8*np.pi))<1e-14)
    checks['gamma_diagonal_third']=bool(abs(matrices(x,'gamma1')[3,0,0]-(.5+2*PHI0/3))<1e-14)
    # Conditional integrator exact orthogonal entries: independent positive means.
    for law,meanpos in (('laplace',1/(2*np.sqrt(2))),('gamma1',np.exp(-1))):
        k=conditional_kernel(x,law,2**16,167)
        checks[f'conditional_exact_{law}']=bool(abs(k[0,1]-meanpos**2)<1e-4)
    checks={k:bool(v) for k,v in checks.items()}
    return dict(checks=checks,passed=sum(checks.values()),total=len(checks),
                maximum_quadrature_error=max(errors),nested_error=worst)

if __name__=='__main__':
    result=run(); print(json.dumps(result,indent=2));
    if not all(result['checks'].values()): raise SystemExit(1)
