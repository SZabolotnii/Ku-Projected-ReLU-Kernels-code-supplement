"""Known-law integration and native HL dictionaries for the first-layer pilot."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import math
import functools
from time import perf_counter
import numpy as np
from scipy import integrate, special, stats

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(__file__).resolve().parents[2] / 'vendor'
sys.path.insert(0, str(UPSTREAM / 'scripts'))
sys.path.insert(0, str(UPSTREAM))
import hl_patp_core as core

ANGLES = np.arange(8) * np.pi / 4
X = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1)
X[np.abs(X) < 1e-14] = 0
ACTS = {
    'relu': ([0.], [0., 1.], [0., 0.]),
    'shift_relu': ([.5], [0., 1.], [0., -.5]),
    'clip_relu': ([0., 1.], [0., 1., 0.], [0., 0., 1.]),
    'hardtanh': ([-1., 1.], [0., 1., 0.], [-1., 0., 1.]),
}
LAWS = ['gamma1', 'gamma4', 'laplace', 'beta25']
GRID = [.2, .35, .5, .75, 1.]


def distinct_cuts(values):
    result = []
    for value in sorted(set(values)):
        if not result or not (np.isfinite(value) and np.isfinite(result[-1])) or value-result[-1] > 1e-12*(1+abs(value)):
            result.append(value)
    return result


def activate(z, name):
    knots, slopes, offsets = ACTS[name]
    k = np.searchsorted(knots, z, side='right')
    return np.take(slopes, k) * z + np.take(offsets, k)


class Law:
    def __init__(self, name):
        self.name = name
        self.mu = 0.
        self.scale = 1.
        if name.startswith('gamma'):
            self.shape = float(name[5:])
            self.mu = self.shape
            self.scale = np.sqrt(self.shape)
            self.dist = stats.gamma(self.shape, loc=-self.mu/self.scale, scale=1/self.scale)
        elif name == 'beta25':
            self.mu = 2/7
            self.scale = np.sqrt(10/392)
            self.dist = stats.beta(2, 5, loc=-self.mu/self.scale, scale=1/self.scale)
        elif name == 'laplace':
            self.dist = stats.laplace(scale=1/np.sqrt(2))
        elif name == 'normal':
            self.dist = stats.norm()
        else:
            raise ValueError(name)
        self.lo, self.hi = self.dist.support()

    def ppf(self, u):
        return self.dist.ppf(u)

    def pdf(self, x):
        # Scalar formulas avoid scipy's distribution dispatch in quadrature.
        if self.name.startswith('gamma'):
            t = self.mu + self.scale*x
            if t <= 0:
                return 0.
            return self.scale * math.exp((self.shape-1)*math.log(t)-t-special.gammaln(self.shape))
        if self.name == 'beta25':
            t = self.mu + self.scale*x
            return self.scale*30*t*(1-t)**4 if 0 < t < 1 else 0.
        if self.name == 'laplace':
            return math.exp(-np.sqrt(2)*abs(x))/np.sqrt(2)
        return math.exp(-x*x/2)/np.sqrt(2*np.pi)

    def expectation(self, f, points=(), tol=2e-11):
        cuts = distinct_cuts(v for v in [self.lo, self.hi, 0., *points] if self.lo <= v <= self.hi)
        result = error = 0.
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            v, e = integrate.quad(lambda x: f(x)*self.pdf(x), lo, hi,
                                  epsabs=tol, epsrel=tol, limit=250)
            result += v
            error += e
        return result, error

    def truncated(self, k, lo, hi):
        """E[W**k 1{lo<W<hi}], k=0,1,2, via incomplete moments."""
        lo, hi = max(lo, self.lo), min(hi, self.hi)
        if hi <= lo:
            return 0.
        if self.name.startswith('gamma') or self.name == 'beta25':
            a, b = self.mu+self.scale*lo, self.mu+self.scale*hi
            value = 0.
            for j in range(k+1):
                if self.name.startswith('gamma'):
                    q = self.shape+j
                    moment = math.exp(special.gammaln(q)-special.gammaln(self.shape))
                    mass = special.gammainc(q, max(b, 0))-special.gammainc(q, max(a, 0))
                else:
                    moment = math.exp(special.betaln(2+j, 5)-special.betaln(2, 5))
                    mass = special.betainc(2+j, 5, np.clip(b, 0, 1))-special.betainc(2+j, 5, np.clip(a, 0, 1))
                value += math.comb(k, j)*(-self.mu)**(k-j)*moment*mass
            return value/self.scale**k
        if self.name == 'laplace':
            value = 0.
            for side in [-1, 1]:
                a, b = (max(lo, 0), max(hi, 0)) if side == 1 else (max(-hi, 0), max(-lo, 0))
                value += side**k*.5*math.factorial(k)/np.sqrt(2)**k * (
                    special.gammainc(k+1, np.sqrt(2)*b)-special.gammainc(k+1, np.sqrt(2)*a))
            return value
        mass = special.ndtr(hi)-special.ndtr(lo)
        pl = self.pdf(lo) if np.isfinite(lo) else 0.
        ph = self.pdf(hi) if np.isfinite(hi) else 0.
        if k == 0:
            return mass
        if k == 1:
            return pl-ph
        return mass+(lo*pl if np.isfinite(lo) else 0)-(hi*ph if np.isfinite(hi) else 0)

    def gauss(self, n):
        if self.name.startswith('gamma'):
            z, w = special.roots_genlaguerre(n, self.shape-1)
            return (z-self.mu)/self.scale, w/special.gamma(self.shape)
        if self.name == 'beta25':
            z, w = special.roots_jacobi(n, 4, 1)
            return ((z+1)/2-self.mu)/self.scale, w/w.sum()
        if self.name == 'laplace':
            z, w = special.roots_laguerre(n//2)
            return np.r_[-z, z]/np.sqrt(2), np.r_[w, w]/2
        return special.roots_hermitenorm(n)[0], special.roots_hermitenorm(n)[1]/np.sqrt(2*np.pi)


class ExactMoments:
    def __init__(self, law):
        self.law = law

    @functools.lru_cache(None)
    def nu(self, q, side):
        lo, hi = (0., self.law.hi) if side == 1 else (self.law.lo, 0.)
        return integrate.quad(lambda z: abs(z)**q*self.law.pdf(z), lo, hi,
                              epsabs=2e-11, epsrel=2e-11, limit=250)[0]

    def sigma(self, q):
        return self.nu(q, 1)-self.nu(q, -1)

    def nu_total(self, q):
        return self.nu(q, 1)+self.nu(q, -1)

    def raw_moment(self, k):
        return self.nu(k, 1)+(-1)**k*self.nu(k, -1)

    def c2(self):
        return 1.


def conditional_product(law, name, a, b, w):
    knots, slopes, offsets = ACTS[name]
    cuts = [law.lo, law.hi]
    for v in (a, b):
        if v[1] != 0:
            cuts.extend((t-v[0]*w)/v[1] for t in knots)
    cuts = distinct_cuts(v for v in cuts if law.lo <= v <= law.hi)
    total = 0.
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mid = (lo+hi)/2 if np.isfinite(lo) and np.isfinite(hi) else (hi-1 if np.isfinite(hi) else lo+1 if np.isfinite(lo) else 0.)
        terms = []
        for v in (a, b):
            j = np.searchsorted(knots, v[0]*w+v[1]*mid, side='right')
            terms.append((slopes[j]*v[1], slopes[j]*v[0]*w+offsets[j]))
        (s, t), (u, v) = terms
        total += s*u*law.truncated(2, lo, hi)+(s*v+t*u)*law.truncated(1, lo, hi)+t*v*law.truncated(0, lo, hi)
    return total


def reference_kernel(law, name, swap=False, tol=2e-10):
    x = X[:, ::-1] if swap else X
    K = np.empty((len(x), len(x)))
    maxerr = 0.
    for i, a in enumerate(x):
        for j in range(i, len(x)):
            b = x[j]
            points = []
            for v in (a, b):
                if v[0] != 0:
                    points.extend(t/v[0] for t in ACTS[name][0])
            if a[1] != 0 and b[1] != 0:
                denom = a[0]/a[1]-b[0]/b[1]
                if abs(denom) > 1e-12:
                    points.extend((s/a[1]-t/b[1])/denom for s in ACTS[name][0] for t in ACTS[name][0])
            value, error = law.expectation(lambda w: conditional_product(law, name, a, b, w), points, tol)
            K[i, j] = K[j, i] = value
            maxerr = max(maxerr, error)
    return K, maxerr


class Basis:
    def __init__(self, law, name, ap=None, am=None):
        self.name, self.ap, self.am = name, ap, am
        if name == 'hl':
            self.system = core.two_sided_system(ExactMoments(law), ap, am, order=2)
            gram = np.zeros((self.system.dimension+1,)*2)
            gram[0, 0] = 1.
            gram[1:, 1:] = self.system.F
        else:
            dim = 4
            gram = np.array([[law.expectation(lambda z: self.raw(np.array([z]))[0, i]*self.raw(np.array([z]))[0, j], points=[.75])[0]
                              for j in range(dim)] for i in range(dim)])
        self.gram = gram
        scale = np.sqrt(np.diag(gram))
        normalized = gram/scale[:, None]/scale[None, :]
        eig, vec = np.linalg.eigh(normalized)
        self.condition = float((eig[-1]/eig[0])**2) if eig[0] > 0 else float('inf')
        self.dim = len(gram)**2
        self.ok = self.condition <= 1e8
        self.transform = (vec/np.sqrt(np.maximum(eig, 1e-300)))/scale[:, None]

    def raw(self, z):
        if self.name == 'hl':
            return np.column_stack([np.ones(len(z)), core.evaluate_components(z, self.system.components, self.system.means).T])
        if self.name == 'hinge':
            return np.column_stack([np.ones(len(z)), z, np.maximum(z, 0), np.maximum(z-.75, 0)])
        return np.column_stack([np.ones(len(z)), z, z*z-1, z*z*z-3*z])

    def evaluate(self, w):
        a = self.raw(w[:, 0]) @ self.transform
        b = self.raw(w[:, 1]) @ self.transform
        return (a[:, :, None]*b[:, None, :]).reshape(len(w), -1)


def all_bases(law):
    bases = {}
    for key in [(ap, am) for ap in GRID for am in GRID]+['hinge', 'hermite']:
        start = perf_counter()
        basis = Basis(law, 'hl', *key) if isinstance(key, tuple) else Basis(law, key)
        basis.setup_seconds = perf_counter()-start
        bases[key] = basis
    return bases
