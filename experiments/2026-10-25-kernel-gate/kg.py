"""Kernel-gate run 2026-10-25: laws, panels, CF-inversion reference, QMC control, H5 score.

The candidate and the stochastic competitors are the frozen baseline functions
(baseline/scripts/projected_core.py, projected_refine.py); nothing here changes them.
New in this run: the deterministic CF-inversion reference/competitor, the panel
generators, the quadratic-CV control for the transfer laws, and the H5 geometry score.
"""
import importlib.util
import os
import sys
from math import factorial
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
import numpy as np
from scipy import special, stats

TRACK = Path(__file__).resolve().parents[2]
# The legacy kernel_common.py inserts an absolute upstream path; preload the pinned
# vendor copies first, as scripts/verify_import.py does.
for _name in ("aepd_core", "hl_patp_core"):
    _spec = importlib.util.spec_from_file_location(_name, TRACK / "vendor" / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
sys.path.insert(0, str(TRACK / "baseline" / "scripts"))
import projected_core as pc  # noqa: E402  frozen candidate + Sobol/conditional competitors
from projected_refine import replicate  # noqa: E402  frozen quadratic-CV Sobol

# Standardised laws: (kappa3, kappa4, kappa5, kappa6) of W_1, mean 0, variance 1.
CUMULANTS = {
    "normal": (0.0, 0.0, 0.0, 0.0),
    "gamma1": (2.0, 6.0, 24.0, 120.0),
    "gamma4": (1.0, 1.5, 3.0, 7.5),          # kappa_r = 4 (r-1)! / 2^r
    "laplace": (0.0, 3.0, 0.0, 30.0),
    "uniform": (0.0, -1.2, 0.0, 48 / 7),     # [-sqrt3, sqrt3]: kappa_r = 3^{r/2} B_r / r
}
# The frozen candidate reads its (kappa3, kappa4) from projected_core.CUMULANTS; the two
# transfer laws are added at run time, the file is not edited.
for _law, (_k3, _k4, _, _) in CUMULANTS.items():
    pc.CUMULANTS.setdefault(_law, (_k3, _k4))


def logcf(law, z):
    """(log|phi(z)|, arg phi(z)) of the standardised law, elementwise."""
    if law == "normal":
        return -z * z / 2, np.zeros_like(z)
    if law == "laplace":
        return -np.log1p(z * z / 2), np.zeros_like(z)
    if law == "gamma1":          # W = G - 1, G ~ Gamma(1)
        return -0.5 * np.log1p(z * z), np.arctan(z) - z
    if law == "gamma4":          # W = (G - 4)/2, G ~ Gamma(4)
        return -2 * np.log1p(z * z / 4), 4 * np.arctan(z / 2) - 2 * z
    if law == "uniform":
        r = np.sqrt(3) * z
        v = np.sinc(r / np.pi)   # sin(r)/r
        return np.log(np.abs(v)), np.where(v < 0, np.pi, 0.0)
    raise ValueError(law)


def dcf_over_cf(law, z):
    """phi'(z)/phi(z), complex."""
    if law == "normal":
        return -z + 0j
    if law == "laplace":
        return -z / (1 + z * z / 2) + 0j
    if law == "gamma1":
        return -z / (1 - 1j * z)
    if law == "gamma4":
        return -z / (1 - 0.5j * z)
    if law == "uniform":
        r = np.sqrt(3) * z
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(np.abs(r) < 1e-6, -z, np.sqrt(3) / np.tan(r) - 1 / z)
        return v + 0j
    raise ValueError(law)


def _nodes(N, sigma=1.0):
    u, w = np.polynomial.legendre.leggauss(N)
    u = (u + 1) / 2
    w = w / 2
    return sigma * u / (1 - u), w * sigma / (1 - u) ** 2


def kernel_cf(x, law, N, chunk=512):
    """Full matrix K_ij = E[(W'x_i)_+(W'x_j)_+] by characteristic-function inversion.

    U_+V_+ = (|U||V| + |U|V + U|V| + UV)/4 with |u| = (2/pi) int_0^inf (1-cos tu)/t^2 dt:
      E|U||V| = (4/pi^2) iint [1 - Re psi(t,0) - Re psi(0,s)
                               + (Re psi(t,s) + Re psi(t,-s))/2] / (t^2 s^2) dt ds,
      E|U|V   = -(2/pi) int Re E[V e^{itU}] / t^2 dt,  E[V e^{itU}] = -i d_s psi(t,s)|_{s=0},
    psi(t,s) = prod_k phi(t x_k + s y_k). Gauss-Legendre on t = u/(1-u), N nodes per axis.
    Rows of x need not be unit; the kernel is computed on unit rows and rescaled.
    """
    norms = np.linalg.norm(x, axis=1)
    x = x / norms[:, None]
    n, _ = x.shape
    t, wt = _nodes(N)
    la, ar = logcf(law, t[:, None, None] * x[None])               # N,n,d
    psi1 = np.exp(la.sum(-1) + 1j * ar.sum(-1))                    # psi_i(t), N,n
    g = dcf_over_cf(law, t[:, None, None] * x[None])               # N,n,d
    M = np.einsum("tik,jk->tij", g, x)                             # sum_k x_jk phi'/phi
    EV = (-1j * psi1[:, :, None] * M).real                          # Re E[U_j e^{itU_i}]
    B = -2 / np.pi * np.einsum("t,tij->ij", wt / t ** 2, EV)       # B_ij = E|U_i| U_j
    R1 = psi1.real
    ii, jj = np.triu_indices(n)
    acc = np.zeros(len(ii))
    ws = wt / t ** 2
    for lo in range(0, len(ii), chunk):
        a, b = ii[lo:lo + chunk], jj[lo:lo + chunk]
        xa, xb = x[a], x[b]
        part = np.zeros(len(a))
        for k in range(N):
            za = t[k] * xa
            out = 0.0
            for sign in (1.0, -1.0):
                z = za[None] + sign * t[:, None, None] * xb[None]   # N,P,d
                l2, a2 = logcf(law, z)
                out = out + 0.5 * (np.exp(l2.sum(-1)) * np.cos(a2.sum(-1)))
            num = 1 - R1[k, a][None] - R1[:, b] + out                 # N,P
            part += wt[k] / t[k] ** 2 * (ws @ num)
        acc[lo:lo + chunk] = part
    A = np.zeros((n, n))
    A[ii, jj] = acc
    A[jj, ii] = acc
    K = (4 / np.pi ** 2 * A + B + B.T + x @ x.T) / 4
    return K * np.outer(norms, norms)


def cf_reference(x, law, N=128):
    """Reference matrix and its relative quadrature error estimate u = ||K_N - K_{N/2}||/||K_N||."""
    fine = kernel_cf(x, law, N)
    coarse = kernel_cf(x, law, N // 2)
    return fine, float(np.linalg.norm(fine - coarse) / np.linalg.norm(fine))


def arccos_kernel(x):
    norms = np.linalg.norm(x, axis=1)
    r = np.clip((x / norms[:, None]) @ (x / norms[:, None]).T, -1, 1)
    return (np.sqrt(1 - r * r) + (np.pi - np.arccos(r)) * r) / (2 * np.pi) * np.outer(norms, norms)


def _ppf(u, law):
    if law in ("gamma1", "laplace", "normal"):
        return pc.transform(u, law)
    u = np.clip(u, .5 / 2 ** 30, 1 - .5 / 2 ** 30)
    if law == "gamma4":
        return (special.gammaincinv(4.0, u) - 4) / 2
    if law == "uniform":
        return np.sqrt(3) * (2 * u - 1)
    raise ValueError(law)


def qcv_sobol(x, law, n_points, seed):
    """Quadratic-control scrambled Sobol, as baseline replicate(); laws beyond the frozen
    transform use the inverse CDF above. Returns the corrected matrix."""
    if law in ("gamma1", "laplace", "normal"):
        return replicate(x[None], law, n_points, seed)[0][0]
    engine = stats.qmc.Sobol(x.shape[1], scramble=True, seed=seed)
    cov = x @ x.T
    rho = np.clip(cov, -1, 1)
    a = (1 + rho) / (2 * np.sqrt(2 * np.pi))
    b = np.sqrt(np.maximum(0, 1 - rho * rho)) / (4 * np.pi)
    c = .25 + np.arcsin(rho) / (2 * np.pi)
    k = np.zeros_like(cov); mean = np.zeros(len(x)); second = np.zeros_like(cov)
    for start in range(0, n_points, 8192):
        z = _ppf(engine.random(min(8192, n_points - start)), law) @ x.T
        f = np.maximum(z, 0)
        k += f.T @ f / n_points
        mean += z.sum(0) / n_points
        second += z.T @ z / n_points
    var = np.diag(second)
    corr = a * (mean[:, None] + mean[None]) + b * (var[:, None] + var[None] - 2) + c * (second - cov)
    return k - corr


def candidate(x, law, order=3):
    """Frozen second-order candidate (order 3) and its ablations: 0 Gaussian, 1 +kappa3, 2 +kappa4."""
    return pc.matrices(x, law, order)[order]


def panel(geometry, d, n, index):
    """Fresh input panel; seeds disjoint from every baseline panel (2026102401+d, 2026102500+...)."""
    gcode = {"dense": 0, "sparse": 1, "near": 2}[geometry]
    rng = np.random.default_rng([20261026, d, n, index, gcode])
    x = rng.normal(size=(n, d))
    if geometry == "sparse":
        k = max(2, d // 8)
        mask = np.zeros((n, d), bool)
        for i in range(n):
            mask[i, rng.choice(d, size=k, replace=False)] = True
        x = np.where(mask, x, 0.0)
    x /= np.linalg.norm(x, axis=1)[:, None]
    if geometry == "near":
        # a quarter of the rows are small rotations of other rows, plus one exact
        # duplicate and one antipode (rho = +-1 exactly)
        m = n // 4
        for i, eps in zip(range(m), np.geomspace(1e-3, 1e-1, m)):
            v = rng.normal(size=d)
            v -= (v @ x[i]) * x[i]
            v /= np.linalg.norm(v)
            x[n - 1 - i] = np.cos(eps) * x[i] + np.sin(eps) * v
        x[m] = x[0]
        x[m + 1] = -x[1]
    return x


def h5_score(x, law):
    """Frozen H5 geometry score: size of the neglected order-3 and order-4 Edgeworth terms.

    For each pair, with a = x_i, b = (x_j - rho x_i)/s (b = 0 when |rho| = 1), the
    order-r cumulant operator of the Edgeworth series has coefficient mass
    sum_{p+q=r} |kappa_pq|/(p! q!) <= |k_r| S_r/r!,  S_r = sum_k (|a_k|+|b_k|)^r.
    With D_r = |k_r| S_r / r!, the neglected order-3 term of exp(D_3 + D_4 + ...) is
    D_5 + D_3 D_4 + D_3^3/6 and the order-4 term is
    D_6 + D_4^2/2 + D_3 D_5 + D_3^2 D_4/2 + D_3^4/24.
    Panel score = ||[m_ij]||_F / ||K_G||_F (unit rows).
    """
    kap = dict(zip((3, 4, 5, 6), (abs(v) for v in CUMULANTS[law])))
    xx = x / np.linalg.norm(x, axis=1)[:, None]
    n = len(xx)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            rho = float(np.clip(xx[i] @ xx[j], -1, 1))
            a = xx[i]
            s2 = 1 - rho * rho
            b = (xx[j] - rho * a) / np.sqrt(s2) if s2 > 1e-24 else np.zeros_like(a)
            w = np.abs(a) + np.abs(b)
            D = {r: kap[r] * float(np.sum(w ** r)) / factorial(r) for r in (3, 4, 5, 6)}
            m3 = D[5] + D[3] * D[4] + D[3] ** 3 / 6
            m4 = D[6] + D[4] ** 2 / 2 + D[3] * D[5] + D[3] ** 2 * D[4] / 2 + D[3] ** 4 / 24
            m[i, j] = m[j, i] = m3 + m4
    return float(np.linalg.norm(m) / np.linalg.norm(arccos_kernel(xx)))


def allowance(K, ref, u):
    e = float(np.linalg.norm(K - ref) / np.linalg.norm(ref))
    return e, (e + u) / (1 - u)


def min_eig(K, ref):
    return float(np.linalg.eigvalsh(K)[0] / np.linalg.norm(ref))
