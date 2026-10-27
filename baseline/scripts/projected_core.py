"""Analytic bivariate Hermite contractions for homogeneous ReLU kernels."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from math import factorial
import numpy as np
from scipy import special, stats, integrate
from threadpoolctl import threadpool_limits
threadpool_limits(1)
H=[np.polynomial.hermite_e.herme2poly(np.eye(7)[k]) for k in range(7)]
PHI0=1/np.sqrt(2*np.pi)
METHODS=('gaussian','third','fourth','second_order')
CUMULANTS={'normal':(0.,0.),'gamma1':(2.,6.),'laplace':(0.,3.)}


def hermite_integrals(r):
    """All needed E[u+ (rho*u+s*v)+ Hp(u)Hq(v)] for iid N(0,1)."""
    r=np.asarray(r); s=np.sqrt((1-r)*(1+r)); c=r/s
    # J[k] = integral_0^inf u^k phi(u)phi(c*u)du.
    J=[special.gamma((k+1)/2)*(2*s*s)**((k+1)/2)/(4*np.pi) for k in range(9)]
    I=[.25+np.arcsin(r)/(2*np.pi), PHI0/2+c*J[0]]
    for k in range(2,9): I.append((k-1)*I[k-2]+c*J[k-1])
    ans={}
    for total in (0,3,4,6):
        for p in range(total+1):
            q=total-p
            if q==0:
                val=sum(h*(r*I[k+2]+s*J[k+1]) for k,h in enumerate(H[p]))
            elif q==1:
                val=s*sum(h*I[k+1] for k,h in enumerate(H[p]))
            else:
                val=sum(h*g*(-c)**l*s*J[k+l+1]
                        for k,h in enumerate(H[p]) for l,g in enumerate(H[q-2]))
            ans[p,q]=val
    return ans


def matrices(x,law,order=3):
    norms=np.linalg.norm(x,axis=1); xx=x/norms[:,None]
    ii,jj=np.triu_indices(len(x)); r=np.clip(np.sum(xx[ii]*xx[jj],axis=1),-1,1)
    scale=norms[ii]*norms[jj]
    g=(np.sqrt(np.maximum(0,1-r*r))+(np.pi-np.arccos(r))*r)/(2*np.pi)
    values=np.tile(g,(4,1)); k3,k4=CUMULANTS[law]
    if order==0 or (k3==0 and k4==0):
        out=np.zeros((4,len(x),len(x)))
        out[:,ii,jj]=out[:,jj,ii]=values*scale
        return out
    regular=(1-np.abs(r))>1e-12
    a=xx[ii[regular]]; rr=r[regular]; s=np.sqrt((1-rr)*(1+rr))
    b=(xx[jj[regular]]-rr[:,None]*a)/s[:,None]
    integrals=hermite_integrals(rr)
    thirds={}
    c3=np.zeros(len(rr)); c4=c3.copy(); c6=c3.copy()
    for p in range(4):
        q=3-p; v=k3*np.sum(a**p*b**q,axis=1)/(factorial(p)*factorial(q))
        thirds[p,q]=v; c3+=v*integrals[p,q]
    for p in (range(5) if order>=2 else []):
        q=4-p; v=k4*np.sum(a**p*b**q,axis=1)/(factorial(p)*factorial(q))
        c4+=v*integrals[p,q]
    for (p,q),v in (thirds.items() if order>=3 else []):
        for (pp,qq),vv in thirds.items(): c6+=.5*v*vv*integrals[p+pp,q+qq]
    values[1:,regular]+=c3
    values[2:,regular]+=c4
    values[3,regular]+=c6
    same=(~regular)&(r>0)
    values[1:,same]+=k3*np.sum(xx[ii[same]]**3,axis=1)*PHI0/3
    values[:,(~regular)&(r<0)]=0
    out=np.zeros((4,len(x),len(x)))
    out[:,ii,jj]=out[:,jj,ii]=values*scale
    return out


def make_inputs(d):
    rng=np.random.default_rng(2026102401+d)
    def unit(a): return a/np.linalg.norm(a,axis=1)[:,None]
    dense=unit(rng.normal(size=(8,d)))
    sparse=np.zeros((8,d)); sparse[:,:2]=unit(rng.normal(size=(8,2)))
    near=dense.copy(); v=dense[1]-np.dot(dense[1],dense[0])*dense[0]; v/=np.linalg.norm(v)
    near[1]=np.cos(.001)*dense[0]+np.sin(.001)*v
    near[2]=-np.cos(.001)*dense[0]+np.sin(.001)*v
    near[3]=dense[0]; near[4]=-dense[0]
    return np.array([dense,sparse,near])


def transform(u,law):
    u=np.clip(u,.5/2**30,1-.5/2**30)
    if law=='gamma1': return -np.log1p(-u)-1
    if law=='laplace': return np.where(u<.5,np.log(2*u),-np.log(2*(1-u)))/np.sqrt(2)
    return special.ndtri(u)


def sobol_kernel(x,law,n,seed,chunk=8192):
    engine=stats.qmc.Sobol(x.shape[-1],scramble=True,seed=seed)
    flat=x.reshape(-1,x.shape[-1]); out=np.zeros(x.shape[:-1]+(x.shape[-2],))
    guards=0
    for start in range(0,n,chunk):
        u=engine.random(min(chunk,n-start)); guards+=int(np.sum((u==0)|(u==1)))
        f=np.maximum(transform(u,law)@flat.T,0).reshape(len(u),*x.shape[:-1])
        out+=np.einsum('n...i,n...j->...ij',f,f,optimize=True)/n
    return out,guards


def conditional_kernel(x,law,n,seed):
    """Integrate one coordinate exactly, retaining zero coefficients."""
    from reuse_fast import moments
    from kernel_common import Law
    ll=Law(law); coordinate=int(np.flatnonzero(np.any(x!=0,axis=0))[-1])
    keep=np.arange(x.shape[1])!=coordinate
    u=stats.qmc.Sobol(x.shape[1]-1,scramble=True,seed=seed).random_base2(int(np.log2(n)))
    outer=transform(u,law); ci=x[:,coordinate]
    ii,jj=np.triu_indices(len(x)); result=np.zeros(len(ii))
    for start in range(0,n,256):
        z=outer[start:start+256]@x[:,keep].T
        zi,zj=z[:,ii],z[:,jj]; a,b=ci[ii],ci[jj]
        lo=np.full_like(zi,ll.lo); hi=np.full_like(zi,ll.hi)
        for coeff,shift in ((a,zi),(b,zj)):
            cuts=np.zeros_like(shift)
            np.divide(-shift,coeff,out=cuts,where=coeff!=0)
            lo=np.where(coeff>0,np.maximum(lo,cuts),lo)
            hi=np.where(coeff<0,np.minimum(hi,cuts),hi)
        valid=(hi>lo)&~(((a==0)&(zi<=0))|((b==0)&(zj<=0)))
        lo=np.clip(lo,ll.lo,ll.hi); hi=np.clip(hi,ll.lo,ll.hi)
        m0,m1,m2=moments(ll,lo,hi)
        val=a*b*m2+(a*zj+b*zi)*m1+zi*zj*m0
        result+=np.sum(np.where(valid,val,0),axis=0)/n
    out=np.zeros((len(x),len(x))); out[ii,jj]=out[jj,ii]=result
    return out
