"""Sparse positive calibration and conditional kernels on dense inputs."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import sys
sys.dont_write_bytecode = True
from time import perf_counter
import numpy as np
from scipy import special, stats
from threadpoolctl import threadpool_limits
from kernel_common import Law, distinct_cuts
from positive_core import CalibrationBasis, calibrate, truncated_vector

threadpool_limits(1)
DIMS = (2, 4, 8, 16)
LAWS = ('gamma1', 'laplace')
NS = (256, 1024, 4096, 16384)
BS = (1, 4, 16)
CONFIGS = {'half_square': ('hl', 1., 1.), 'fractional': ('hl', .75, .2),
           'hermite': ('hermite', None, None)}
ACTS = ('relu', 'shift_relu')
NODE_GUARDS = []


def open_uniform(u):
    """Sobol uses 30-bit cells; map only endpoints to their cell midpoint."""
    return np.clip(u, .5/2**30, 1-.5/2**30)


def inputs(d):
    rng = np.random.default_rng(202610240+d)
    x = []
    while len(x) < 128:
        z = rng.normal(size=d)
        z /= np.linalg.norm(z)
        if np.min(abs(z)) >= .1/np.sqrt(d):
            x.append(z)
    return np.array(x).reshape(16, 8, d)


def nodes(law, n, d, seed):
    u = stats.qmc.Sobol(d, scramble=True, seed=seed).random_base2(int(np.log2(n)))
    count = int(np.sum((u == 0) | (u == 1)))
    if count:
        NODE_GUARDS.append(dict(n=n, d=d, seed=seed, endpoints=count))
    u = open_uniform(u)
    if law.name == 'gamma1':
        return -np.log1p(-u)-1
    if law.name == 'laplace':
        return np.where(u < .5, np.log(2*u), -np.log(2*(1-u)))/np.sqrt(2)
    return special.ndtri(u)


class SparseBasis:
    def __init__(self, law, config):
        self.one = CalibrationBasis(law, config)

    def evaluate(self, w):
        n, d = w.shape
        a = (self.one.base.raw(w.ravel())[:, 1:]-self.one.mean)@self.one.whiten
        a = a.reshape(n, d*3)
        i, j = np.triu_indices(d, 1)
        return np.column_stack([np.ones(n), a, w[:, i]*w[:, j]])


def fit_weights(design):
    n, p = design.shape
    target = np.eye(1, p)[0]
    coefficient = np.linalg.solve(design.T@design, target)
    omega = design@coefficient
    residual = float(np.max(abs(design.T@omega-target)))
    if omega.min() >= 0 and residual <= 1e-8:
        return omega, dict(status='calibrated', success=True, path='positive_unconstrained',
                           residual=residual, stationarity=0.), n*coefficient
    omega, info, dual = calibrate(design)
    info['path'] = 'constrained'
    return omega, info, dual


def point_kernels(w, omega, x):
    z = (w@x.reshape(-1, x.shape[-1]).T).reshape(len(w), len(x), 8)
    f = np.maximum(z, 0)
    g = np.maximum(z-.5, 0)
    return np.stack([np.einsum('n,nbi,nbj->bij', omega, f, f, optimize=True),
                     np.einsum('n,nbi,nbj->bij', omega, g, g, optimize=True)])


def point_batch(w, omega, x):
    z = w@x.T
    out = []
    for threshold in (0., .5):
        f = np.maximum(z-threshold, 0)
        out.append(f.T@(omega[:, None]*f))
    return np.array(out)


def incomplete(law, k, lo, hi):
    if law.name != 'normal':
        return truncated_vector(law, k, lo, hi)
    if k == 0:
        return np.where(lo > 0, special.ndtr(-lo)-special.ndtr(-hi),
                        special.ndtr(hi)-special.ndtr(lo))
    phi_l = np.exp(-lo*lo/2)/np.sqrt(2*np.pi)
    phi_h = np.exp(-hi*hi/2)/np.sqrt(2*np.pi)
    if k == 1:
        return phi_l-phi_h
    return incomplete(law, 0, lo, hi)+np.where(np.isfinite(lo), np.nan_to_num(lo)*phi_l, 0)-np.where(np.isfinite(hi), np.nan_to_num(hi)*phi_h, 0)


def conditional_batch(law, outer, omega, x):
    """Exact integration in last coordinate; accumulate positive outer rule."""
    ii, jj = np.triu_indices(len(x))
    coeff_i, coeff_j = x[ii, -1], x[jj, -1]
    result = np.zeros((2, len(ii)))
    for start in range(0, len(outer), 256):
        z = outer[start:start+256]@x[:, :-1].T
        zi, zj = z[:, ii], z[:, jj]
        weights = omega[start:start+256]
        for ia, threshold in enumerate((0., .5)):
            cuts = np.sort(np.stack([np.full_like(zi, law.lo), np.full_like(zi, law.hi),
                                     np.clip((threshold-zi)/coeff_i, law.lo, law.hi),
                                     np.clip((threshold-zj)/coeff_j, law.lo, law.hi)], axis=-1), axis=-1)
            lo, hi = cuts[..., :-1], cuts[..., 1:]
            mid = .5*(np.where(np.isfinite(lo), lo, 0)+np.where(np.isfinite(hi), hi, 0))
            mid = np.where(~np.isfinite(hi), lo+1, mid)
            mid = np.where(~np.isfinite(lo), hi-1, mid)
            ai = zi[..., None]+coeff_i[None, :, None]*mid > threshold
            aj = zj[..., None]+coeff_j[None, :, None]*mid > threshold
            s, t = ai*coeff_i[None, :, None], ai*(zi[..., None]-threshold)
            u, v = aj*coeff_j[None, :, None], aj*(zj[..., None]-threshold)
            integral = np.sum(s*u*incomplete(law, 2, lo, hi)+(s*v+t*u)*incomplete(law, 1, lo, hi)+t*v*incomplete(law, 0, lo, hi), axis=-1)
            result[ia] += weights@integral
    matrix = np.empty((2, len(x), len(x)))
    matrix[:, ii, jj] = matrix[:, jj, ii] = result
    return matrix


def split_nodes(law, x, order):
    cuts = [law.lo, law.hi, 0.]
    for a in x:
        for threshold in (0., .5):
            for bound in (law.lo, law.hi):
                if np.isfinite(bound):
                    cuts.append((threshold-a[1]*bound)/a[0])
            cuts.append(threshold/a[0])
    for a in x:
        for b in x:
            den = a[0]/a[1]-b[0]/b[1]
            if abs(den) > 1e-12:
                cuts.extend((s/a[1]-t/b[1])/den for s in (0., .5) for t in (0., .5))
    cuts = distinct_cuts(c for c in cuts if law.lo <= c <= law.hi)
    leg, lw = special.roots_legendre(order)
    lag, gw = special.roots_laguerre(order)
    rate = 1. if law.name == 'gamma1' else np.sqrt(2)
    zs, ws = [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if np.isfinite(lo) and np.isfinite(hi):
            z = lo+(leg+1)*(hi-lo)/2
            weight = lw*(hi-lo)/2*np.array([law.pdf(q) for q in z])
        else:
            z = lo+lag/rate if np.isfinite(lo) else hi-lag/rate
            weight = gw*np.exp(lag)/rate*np.array([law.pdf(q) for q in z])
        zs.extend(z); ws.extend(weight)
    ws = np.array(ws)
    return np.array(zs)[:, None], ws/ws.sum(), float(ws.sum())


def timed_batches(evaluator, x, repeats=3):
    evaluator(x[0])
    timings, matrices = [], []
    for repeat in range(repeats):
        elapsed, values = [], []
        start = perf_counter()
        for batch in x:
            values.append(evaluator(batch))
            elapsed.append(perf_counter()-start)
        timings.append(elapsed)
        if repeat == 0:
            matrices = np.stack(values, axis=1)
    return matrices, np.array(timings)
