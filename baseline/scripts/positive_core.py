"""Positive moment calibration and vectorized conditional kernel integration."""
import sys
sys.dont_write_bytecode = True
import math
import numpy as np
from scipy import optimize, special
from kernel_common import Basis, ACTS, X

CONFIGS = {
    'cal_hl_diag': ('hl', .75, .75),
    'cal_hl_right_frac': ('hl', .2, .75),
    'cal_hl_left_frac': ('hl', .75, .2),
    'cal_hinge': ('hinge', None, None),
    'cal_hermite': ('hermite', None, None),
    'cal_midpoint': ('hl', .5, .5),
}


class CalibrationBasis:
    def __init__(self, law, config):
        self.base = Basis(law, *config)
        self.mean = np.array([law.expectation(lambda z: self.base.raw(np.array([z]))[0, j], points=[.75])[0]
                              for j in range(1, len(self.base.gram))])
        covariance = self.base.gram[1:, 1:]-np.outer(self.mean, self.mean)
        values, vectors = np.linalg.eigh(covariance)
        if values[0] <= 0:
            raise ValueError('Population covariance is not positive definite')
        self.whiten = vectors/np.sqrt(values)
        self.dim = len(self.base.gram)**2

    def evaluate(self, w):
        a = np.column_stack([np.ones(len(w)), (self.base.raw(w[:, 0])[:, 1:]-self.mean)@self.whiten])
        b = np.column_stack([np.ones(len(w)), (self.base.raw(w[:, 1])[:, 1:]-self.mean)@self.whiten])
        return (a[:, :, None]*b[:, None, :]).reshape(len(w), -1)


def calibrate(design):
    """Max-ESS nonnegative quadrature. Fallback is always explicitly labelled."""
    n, p = design.shape
    target = np.zeros(p)
    target[0] = 1.
    lp = optimize.linprog(np.zeros(n), A_eq=design.T, b_eq=target, bounds=(0, None),
                          method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                                  'dual_feasibility_tolerance': 1e-9})
    uniform = np.full(n, 1/n)
    if not lp.success:
        return uniform, dict(status='infeasible' if lp.status == 2 else 'lp_failure', lp_status=int(lp.status),
                             success=False, residual=float(np.max(abs(design.T@uniform-target))),
                             stationarity=float('nan'), iterations=0, lp_message=lp.message), np.zeros(p)

    def fun(lam):
        pos = np.maximum(design@lam, 0)
        return .5*np.mean(pos*pos)-lam[0], design.T@pos/n-target

    initial = target.copy()
    result = optimize.minimize(fun, initial, jac=True, method='L-BFGS-B',
                               options={'gtol': 1e-11, 'ftol': 1e-15, 'maxiter': 1500, 'maxls': 40})
    lam = result.x
    iterations = int(result.nit)
    # Semismooth Newton polish of the convex dual; no primal clipping or ridge.
    for iteration in range(40):
        obj, grad = fun(lam)
        if np.max(abs(grad)) < 5e-12:
            break
        active = design@lam > 0
        hessian = design[active].T@design[active]/n
        delta = np.linalg.lstsq(hessian, -grad, rcond=1e-13)[0]
        if grad@delta >= 0:
            break
        step = 1.
        for _ in range(40):
            if fun(lam+step*delta)[0] <= obj+1e-4*step*(grad@delta)+1e-15:
                break
            step *= .5
        lam = lam+step*delta
        iterations += 1
    linear = design@lam
    weights = np.maximum(linear, 0)/n
    residual = float(np.max(abs(design.T@weights-target)))
    stationarity = float(max(np.max(np.maximum(linear-n*weights, 0)),
                             np.max(abs(weights*(n*weights-linear)))))
    success = bool(np.isfinite(weights).all() and residual <= 1e-8 and stationarity <= 1e-8)
    info = dict(status='calibrated' if success else 'qp_failure', lp_status=int(lp.status), success=success,
                residual=residual, stationarity=stationarity, iterations=iterations, lp_message=lp.message)
    if not success:
        info['failed_qp_residual'] = residual
        weights = uniform
        info['residual'] = float(np.max(abs(design.T@weights-target)))
    return weights, info, lam


def truncated_vector(law, k, lo, hi):
    lo, hi = np.maximum(lo, law.lo), np.minimum(hi, law.hi)
    if law.name.startswith('gamma') or law.name == 'beta25':
        a, b = law.mu+law.scale*lo, law.mu+law.scale*hi
        value = np.zeros_like(lo)
        for j in range(k+1):
            if law.name.startswith('gamma'):
                q = law.shape+j
                moment = math.exp(special.gammaln(q)-special.gammaln(law.shape))
                mass = special.gammainc(q, np.maximum(b, 0))-special.gammainc(q, np.maximum(a, 0))
            else:
                moment = math.exp(special.betaln(2+j, 5)-special.betaln(2, 5))
                mass = special.betainc(2+j, 5, np.clip(b, 0, 1))-special.betainc(2+j, 5, np.clip(a, 0, 1))
            value += math.comb(k, j)*(-law.mu)**(k-j)*moment*mass
        value /= law.scale**k
    elif law.name == 'laplace':
        value = np.zeros_like(lo)
        for side in (-1, 1):
            a, b = (np.maximum(lo, 0), np.maximum(hi, 0)) if side == 1 else (np.maximum(-hi, 0), np.maximum(-lo, 0))
            value += side**k*.5*math.factorial(k)/np.sqrt(2)**k*(special.gammainc(k+1, np.sqrt(2)*b)-special.gammainc(k+1, np.sqrt(2)*a))
    else:
        raise ValueError(law.name)
    return np.where(hi > lo, value, 0.)


def conditional_matrices(law, outer):
    """Return E[f f^T | W1=w] for all activations and all outer nodes."""
    output = np.empty((len(ACTS), len(outer), len(X), len(X)))
    for ia, (name, (knots, slopes, offsets)) in enumerate(ACTS.items()):
        for i, a in enumerate(X):
            for j in range(i, len(X)):
                b = X[j]
                cuts = [np.full_like(outer, law.lo), np.full_like(outer, law.hi)]
                for v in (a, b):
                    if v[1] != 0:
                        cuts.extend(np.clip((t-v[0]*outer)/v[1], law.lo, law.hi) for t in knots)
                cuts = np.sort(np.stack(cuts, axis=1), axis=1)
                lo, hi = cuts[:, :-1], cuts[:, 1:]
                both = np.isfinite(lo) & np.isfinite(hi)
                mid = .5*(np.where(np.isfinite(lo), lo, 0)+np.where(np.isfinite(hi), hi, 0))
                mid = np.where(~both & np.isfinite(lo), lo+1, mid)
                mid = np.where(~both & np.isfinite(hi), hi-1, mid)
                terms = []
                for v in (a, b):
                    ix = np.searchsorted(knots, v[0]*outer[:, None]+v[1]*mid, side='right')
                    s = np.take(slopes, ix)
                    terms.append((s*v[1], s*v[0]*outer[:, None]+np.take(offsets, ix)))
                (s, t), (u, v) = terms
                value = np.sum(s*u*truncated_vector(law, 2, lo, hi)+(s*v+t*u)*truncated_vector(law, 1, lo, hi)+t*v*truncated_vector(law, 0, lo, hi), axis=1)
                output[ia, :, i, j] = output[ia, :, j, i] = value
    return output
