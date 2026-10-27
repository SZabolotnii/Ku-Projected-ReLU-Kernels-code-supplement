"""mc_kernel spec — confirmatory kernel gate, run 1, timing session 1 (SPEC.md, 2026-10-25).

Run:  python $AUTOSCI/tools/mc_kernel.py experiments/2026-10-25-kernel-gate/spec.py --resume
Every cell is one input panel (the unit of replication, BRIEF A5). Parts:
  gate      H3/H4/S1/S2/K4 — dense, (Gamma(1), d=64) and (Laplace, d=16), n=32,128, 10 panels,
            accuracy + time-to-target against every competitor;
  accuracy  full (law, d) grid at n=32 (W2), H2 d-sweep, H5 geometries — accuracy only.
"""
import os

# One BLAS thread (A6) must be set before numpy loads; numpy here uses Accelerate, which
# threadpoolctl cannot limit afterwards. timed() also records CPU/wall to prove it.
for _key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter, process_time  # noqa: E402

import numpy as np  # noqa: E402
from scipy import integrate
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg  # noqa: E402

RUN_ID = "kernel-gate-2026-10-25-s1"
TARGET = 0.003
PSD_TOL = -1e-10
U_MAX = 1e-4                 # CF quadrature estimate above this: REFERENCE_LIMITED
QMC_POINTS, QMC_REPS = 2 ** 20, 8
LADDER = {"sobol": range(10, 23), "qcv": range(10, 23), "conditional": range(10, 19)}
CF_LADDER = (8, 12, 16, 24, 32, 48, 64, 96)
REPS = 8
PANELS = 10


def seed(params, salt, rep=0):
    key = json.dumps(params, sort_keys=True) + salt
    return int(hashlib.sha256(key.encode()).hexdigest()[:7], 16) * 100 + rep


CPU_WALL = []                # CPU/wall of every timed call in the current cell


def timed(fn):
    fn()
    times = []
    for _ in range(5):
        t0, c0 = perf_counter(), process_time()
        fn()
        wall = perf_counter() - t0
        times.append(wall)
        CPU_WALL.append((process_time() - c0) / wall if wall > 0 else 1.0)
    return float(np.median(times))


def reference(params, x, law):
    """CF reference + independent QMC control. ok iff u <= U_MAX and |CF - QMC| <= 3 SE + u."""
    ref, u = kg.cf_reference(x, law)
    reps = np.array([kg.qcv_sobol(x, law, QMC_POINTS, seed(params, "qmc-control", r))
                     for r in range(QMC_REPS)])
    nrm = np.linalg.norm(ref)
    diff = float(np.linalg.norm(reps.mean(0) - ref) / nrm)
    three_se = float(3 * np.sqrt(reps.var(0, ddof=1).sum() / QMC_REPS) / nrm)
    ok = u <= U_MAX and diff <= three_se + u
    return ref, u, dict(u=u, qmc_diff=diff, qmc_3se=three_se, ref_ok=float(ok))


def accuracy_values(x, law, ref, u):
    out = {}
    for order, name in enumerate(("gaussian", "third", "fourth", "candidate")):
        K = kg.candidate(x, law, order)
        e, allow = kg.allowance(K, ref, u)
        out[f"{name}_error"] = e
        out[f"{name}_allowance"] = allow
        out[f"{name}_mineig"] = kg.min_eig(K, ref)
        out[f"{name}_maxabs"] = float(np.max(np.abs(K - ref)))
    out["h5_score"] = kg.h5_score(x, law)
    return out


def stochastic(method, x, law, N, s):
    if method == "sobol":
        return kg.pc.sobol_kernel(x[None], law, N, s)[0][0]
    if method == "qcv":
        return kg.qcv_sobol(x, law, N, s)
    return kg.pc.conditional_kernel(x, law, N, s)


def ladder(params, method, x, law, ref, u):
    """Smallest 2^k at which all REPS scrambles reach the target and stay PSD."""
    for k in LADDER[method]:
        worst, low = 0.0, 0.0
        for r in range(REPS):
            K = stochastic(method, x, law, 2 ** k, seed(params, method, r))
            worst = max(worst, kg.allowance(K, ref, u)[1])
            low = min(low, kg.min_eig(K, ref))
        if worst <= TARGET and low >= PSD_TOL:
            s = seed(params, method, 0)
            return dict(level=float(k), allowance=worst,
                        seconds=timed(lambda: stochastic(method, x, law, 2 ** k, s)))
    return dict(level=float("nan"), allowance=worst, seconds=float("nan"))


def cf_ladder(x, law, ref, u):
    for N in CF_LADDER:
        allow = kg.allowance(kg.kernel_cf(x, law, N), ref, u)[1]
        if allow <= TARGET:
            return dict(level=float(N), allowance=allow, seconds=timed(lambda: kg.kernel_cf(x, law, N)))
    return dict(level=float("nan"), allowance=allow, seconds=float("nan"))


def gate(params, x, law):
    ref, u, out = reference(params, x, law)
    out.update(accuracy_values(x, law, ref, u))
    CPU_WALL.clear()
    rng = np.random.default_rng(seed(params, "order"))
    comps = {}
    # randomised method order for the timing, same session, one BLAS thread (A6)
    for method in rng.permutation(["candidate", "gaussian", "sobol", "qcv", "conditional", "cf"]):
        if method == "candidate":
            out["candidate_seconds"] = timed(lambda: kg.candidate(x, law))
        elif method == "gaussian":
            out["gaussian_seconds"] = timed(lambda: kg.candidate(x, law, 0))
            comps["gaussian"] = dict(allowance=out["gaussian_allowance"], seconds=out["gaussian_seconds"],
                                     level=0.0)
        elif method == "cf":
            comps["cf"] = cf_ladder(x, law, ref, u)
        else:
            comps[method] = ladder(params, method, x, law, ref, u)
    for name, c in comps.items():
        for k, v in c.items():
            out[f"{name}_{k}"] = v
    eligible = {name: c["seconds"] for name, c in comps.items()
                if c["allowance"] <= TARGET and math.isfinite(c["seconds"])
                and (name != "gaussian" or out["gaussian_mineig"] >= PSD_TOL)}
    best = min(eligible.values()) if eligible else float("nan")
    out["best_competitor_seconds"] = best
    out["time_ratio"] = out["candidate_seconds"] / best if eligible else float("nan")
    cand_ok = out["candidate_allowance"] <= TARGET and out["candidate_mineig"] >= PSD_TOL
    out["candidate_ok"] = float(cand_ok)
    # S2 counts only panels whose reference is valid
    out["gaussian_fails"] = float(bool(out["ref_ok"]) and out["gaussian_allowance"] > TARGET)
    out["cpu_wall_max"] = float(max(CPU_WALL))
    # status: 0 PASS, 1 NO_SPEED_GAIN, 2 COMPETITOR_LIMITED, 3 CANDIDATE_FAIL, 4 REFERENCE_LIMITED
    if not out["ref_ok"]:
        status = 4
    elif not cand_ok:
        status = 3
    elif not eligible:
        status = 2
    else:
        status = 0 if out["time_ratio"] <= 0.8 else 1
    out["status"] = float(status)
    out["pass"] = float(status == 0)
    return out


# --------------------------------------------------------------------------- controls
def _he(n):
    c = np.zeros(n + 1)
    c[n] = 1
    return np.polynomial.hermite_e.herme2poly(c)


def _t_quad(p, q, rho):
    s = math.sqrt(1 - rho * rho)
    hp, hq = _he(p)[::-1], _he(q)[::-1]
    f = lambda z2, z1: z1 * (rho * z1 + s * z2) * np.polyval(hp, z1) * np.polyval(hq, z2) \
        * norm.pdf(z1) * norm.pdf(z2)
    return integrate.dblquad(f, 0, 12, lambda z1: -rho * z1 / s, lambda z1: 14,
                             epsabs=1e-14, epsrel=1e-13)[0]


def _t_closed(p, q, rho):
    """thm:closed for any (p, q), written independently of the frozen implementation."""
    s = math.sqrt(1 - rho * rho)
    c = rho / s
    J = [math.gamma((k + 1) / 2) * (2 * s * s) ** ((k + 1) / 2) / (4 * math.pi) for k in range(p + q + 6)]
    I = [0.25 + math.asin(rho) / (2 * math.pi), kg.pc.PHI0 / 2 + c * J[0]]
    for k in range(2, p + q + 6):
        I.append((k - 1) * I[k - 2] + c * J[k - 1])
    P = np.polynomial.polynomial
    up = P.polymul([0, 1], _he(p))
    if q == 0:
        a = P.polymul(up, [0, rho])
        return sum(a[k] * I[k] for k in range(len(a))) + s * sum(up[k] * J[k] for k in range(len(up)))
    if q == 1:
        return s * sum(up[k] * I[k] for k in range(len(up)))
    a = P.polymul(up, _he(q - 2) * np.array([(-c) ** k for k in range(q - 1)]))
    return s * sum(a[k] * J[k] for k in range(len(a)))


def control(params):
    kind = params["control"]
    if kind == "algebra":            # H1 / S4: every T_pq, p+q <= 6, against quadrature
        frozen, closed = 0.0, 0.0
        for rho in (-0.95, -0.6, -0.2, 0.0, 0.3, 0.7, 0.95):
            used = kg.pc.hermite_integrals(np.array([rho]))
            for p in range(7):
                for q in range(7 - p):
                    quad = _t_quad(p, q, rho)
                    closed = max(closed, abs(_t_closed(p, q, rho) - quad))
                    if (p, q) in used:
                        frozen = max(frozen, abs(float(used[p, q][0]) - quad))
        return dict(frozen_vs_quad=frozen, closed_vs_quad=closed)
    if kind == "gaussian":           # exact law: candidate and CF must give the arc-cosine kernel
        x = kg.panel("dense", 64, 32, 999)
        exact = kg.arccos_kernel(x)
        nrm = np.linalg.norm(exact)
        ref, u = kg.cf_reference(x, "normal")
        return dict(candidate_vs_arccos=float(np.linalg.norm(kg.candidate(x, "normal") - exact) / nrm),
                    cf_vs_arccos=float(np.linalg.norm(ref - exact) / nrm), u=u)
    if kind == "structure":          # symmetry, homogeneity, rho -> +-1 continuity (prop:symhom, prop:limits)
        x = kg.panel("dense", 16, 32, 998)
        law = "gamma1"
        K = kg.candidate(x, law)
        perm = np.random.default_rng(1).permutation(len(x))
        sym = float(np.max(np.abs(kg.candidate(x[perm], law) - K[np.ix_(perm, perm)])))
        lam = np.random.default_rng(2).uniform(0.5, 2.0, len(x))
        hom = float(np.max(np.abs(kg.candidate(x * lam[:, None], law) - K * np.outer(lam, lam))))
        v = x[1] - (x[1] @ x[0]) * x[0]
        v /= np.linalg.norm(v)
        lim = 0.0
        for sign in (1.0, -1.0):
            eps = 1e-6
            pair = np.stack([x[0], sign * np.cos(eps) * x[0] + np.sin(eps) * v])
            exact_end = np.stack([x[0], sign * x[0]])
            lim = max(lim, abs(kg.candidate(pair, law)[0, 1] - kg.candidate(exact_end, law)[0, 1]))
        return dict(symmetry=sym, homogeneity=hom, limit_gap=lim)
    if kind == "qmc":                # CF and QMC agree on a non-Gaussian panel
        x = kg.panel("dense", 16, 32, 997)
        _, u, out = reference(params, x, "laplace")
        out["qmc_excess"] = max(0.0, out["qmc_diff"] - out["qmc_3se"] - u)
        return out
    raise ValueError(kind)


CONTROLS = [
    dict(name="algebra: T_pq vs quadrature, p+q<=6",
         params=dict(control="algebra"),
         expect={"frozen_vs_quad": (0.0, 1e-10), "closed_vs_quad": (0.0, 1e-10)}),
    dict(name="gaussian law: candidate and CF equal arc-cosine",
         params=dict(control="gaussian"),
         expect={"candidate_vs_arccos": (0.0, 1e-13), "cf_vs_arccos": (0.0, 5e-5)}),
    dict(name="symmetry, homogeneity, rho->+-1 limits",
         params=dict(control="structure"),
         expect={"symmetry": (0.0, 1e-12), "homogeneity": (0.0, 1e-12), "limit_gap": (0.0, 1e-5)}),
    dict(name="CF reference agrees with independent QMC",
         params=dict(control="qmc"),
         expect={"qmc_excess": (0.0, 1e-12), "ref_ok": (1.0, 0.0)}),
]

GRID = []
for law, d in (("gamma1", 64), ("laplace", 16)):
    for n in (32, 128):
        for i in range(PANELS):
            GRID.append(dict(part="gate", law=law, d=d, n=n, geometry="dense", panel=i))
for law in ("gamma1", "laplace", "gamma4", "uniform"):
    for d in (16, 64, 128):
        if (law, d) in (("gamma1", 64), ("laplace", 16)):
            continue                 # these n=32 panels are the gate cells above
        for i in range(PANELS):
            GRID.append(dict(part="accuracy", law=law, d=d, n=32, geometry="dense", panel=i))
for d in (32, 256):                  # H2 d-sweep: with d=16, 64, 128 above, five values of d
    for i in range(PANELS):
        GRID.append(dict(part="accuracy", law="gamma1", d=d, n=32, geometry="dense", panel=i))
for law in ("gamma1", "laplace"):    # H5 geometries; dense comes from the cells above
    for d in (16, 64):
        for geometry in ("sparse", "near"):
            for i in range(PANELS):
                GRID.append(dict(part="accuracy", law=law, d=d, n=32, geometry=geometry, panel=i))


def cell(params, memo):
    if "control" in params:
        return control(params)
    x = kg.panel(params["geometry"], params["d"], params["n"], params["panel"])
    law = params["law"]
    if params["part"] == "gate":
        return gate(params, x, law)
    ref, u, out = reference(params, x, law)
    out.update(accuracy_values(x, law, ref, u))
    return out
