"""Closed incomplete moments for the two primary laws; independent slow crosscheck."""
import numpy as np


def moments(law, lo, hi):
    valid = hi > lo
    if law.name == 'gamma1':
        l = np.maximum(lo, -1.)
        h = np.maximum(hi, -1.)
        el, eh = np.exp(-(l+1)), np.exp(-(h+1))
        hf = np.where(np.isfinite(h), h, 0.)
        return (np.where(valid, el-eh, 0),
                np.where(valid, (l+1)*el-(hf+1)*eh, 0),
                np.where(valid, (l*l+2*l+2)*el-(hf*hf+2*hf+2)*eh, 0))
    out = [np.zeros_like(lo) for _ in range(3)]
    rate = np.sqrt(2)
    for side in (-1, 1):
        l, h = (np.maximum(lo, 0), np.maximum(hi, 0)) if side == 1 else (np.maximum(-hi, 0), np.maximum(-lo, 0))
        el, eh = np.exp(-rate*l)/2, np.exp(-rate*h)/2
        lf, hf = np.where(np.isfinite(l), l, 0), np.where(np.isfinite(h), h, 0)
        out[0] += el-eh
        out[1] += side*((lf+1/rate)*el-(hf+1/rate)*eh)
        out[2] += (lf*lf+2*lf/rate+1)*el-(hf*hf+2*hf/rate+1)*eh
    return tuple(np.where(valid, v, 0) for v in out)


def conditional_batch(law, outer, omega, x):
    ii, jj = np.triu_indices(len(x))
    ci, cj = x[ii,-1], x[jj,-1]
    result = np.zeros((2, len(ii)))
    for start in range(0, len(outer), 256):
        z = outer[start:start+256]@x[:,:-1].T
        zi, zj = z[:,ii], z[:,jj]
        for ia, threshold in enumerate((0., .5)):
            cuts = np.sort(np.stack([np.full_like(zi,law.lo),np.full_like(zi,law.hi),
                                     np.clip((threshold-zi)/ci,law.lo,law.hi),
                                     np.clip((threshold-zj)/cj,law.lo,law.hi)],axis=-1),axis=-1)
            lo,hi=cuts[...,:-1],cuts[...,1:]
            mid=.5*(np.where(np.isfinite(lo),lo,0)+np.where(np.isfinite(hi),hi,0))
            mid=np.where(~np.isfinite(hi),lo+1,mid)
            mid=np.where(~np.isfinite(lo),hi-1,mid)
            ai=zi[...,None]+ci[None,:,None]*mid>threshold
            aj=zj[...,None]+cj[None,:,None]*mid>threshold
            s,t=ai*ci[None,:,None],ai*(zi[...,None]-threshold)
            u,v=aj*cj[None,:,None],aj*(zj[...,None]-threshold)
            m0,m1,m2=moments(law,lo,hi)
            values=np.sum(s*u*m2+(s*v+t*u)*m1+t*v*m0,axis=-1)
            result[ia]+=omega[start:start+256]@values
    matrix=np.empty((2,len(x),len(x)))
    matrix[:,ii,jj]=matrix[:,jj,ii]=result
    return matrix
