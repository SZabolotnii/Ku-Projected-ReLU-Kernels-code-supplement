"""mc_kernel spec — run 2, H6 on sklearn digits (protocol frozen in 2026-10-25-kernel-gate/SPEC.md).

Run:  python $AUTOSCI/tools/mc_kernel.py experiments/2026-10-26-h6-digits/spec_h6.py --resume
One cell per law (Gamma(1), Laplace). Reference and ladder evaluations run on WORKERS
processes; every timed call runs afterwards in this process, one BLAS thread, nothing else
of this run in parallel.
"""
import os

for _key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter, process_time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h6  # noqa: E402
from h6 import kg  # noqa: E402

RUN_ID = "h6-digits-2026-10-26"
RULE = 0.01                  # H6: relative test-MSE difference against the reference
U_MAX = 1e-4
QMC_POINTS, QMC_REPS = 2 ** 20, 8
LADDER = range(10, 21)       # Sobol and quadratic-CV Sobol, 2^10 .. 2^20 points
CF_LADDER = (8, 12, 16, 24, 32)
REPS = 8
WORKERS = 9
SLOW = 30.0                  # a timed call slower than this is measured once, not 1 + 3


def seed(params, salt, rep=0):
    key = json.dumps(params, sort_keys=True) + "h6" + salt
    return int(hashlib.sha256(key.encode()).hexdigest()[:7], 16) * 100 + rep


CPU_WALL = []


def timed(fn):
    """Median wall time of fn; cheap calls get a warm-up and 3 repeats, slow ones 1 run."""
    times = []
    for _ in range(4):
        t0, c0 = perf_counter(), process_time()
        fn()
        wall = perf_counter() - t0
        CPU_WALL.append((process_time() - c0) / wall if wall > 0 else 1.0)
        times.append(wall)
        if times[0] > SLOW:
            return float(times[0])
    return float(np.median(times[1:]))


def rel(a, b):
    return abs(a - b) / b


def stochastic(method, x, law, N, s):
    if method == "sobol":
        return kg.pc.sobol_kernel(x[None], law, N, s)[0][0]
    return kg.qcv_sobol(x, law, N, s)


def _ladder_job(args):
    method, x, y, n_train, law, N, s = args
    return h6.krr_mse(stochastic(method, x, law, N, s), y, n_train)


def ladder(params, method, x, y, n_train, law, mse_ref):
    """Smallest 2^k at which all REPS scrambles give a test MSE within RULE of the reference."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS) as pool:
        for k in LADDER:
            jobs = [(method, x, y, n_train, law, 2 ** k, seed(params, method, r)) for r in range(REPS)]
            worst = max(rel(m, mse_ref) for m in pool.map(_ladder_job, jobs))
            if worst <= RULE:
                return k, worst
    return None, worst


def control(params):
    kind = params["control"]
    if kind == "parallel_cf":        # pair-parallel CF is kg.kernel_cf
        x = kg.panel("dense", 64, 24, 996)
        a = kg.kernel_cf(x, "gamma1", 32)
        b = h6.kernel_cf_pairs(x, "gamma1", 32, workers=4, chunk=64)
        return dict(parallel_vs_kg=float(np.linalg.norm(a - b) / np.linalg.norm(a)))
    if kind == "split":              # frozen preprocessing and split
        x, y, n_train = h6.digits()
        test = y[n_train:]
        strat = max(abs(np.sum(test == c) - h6.TEST_SIZE * np.sum(y == c)) for c in range(10))
        return dict(n_train=float(n_train), n_test=float(len(test)), strat_gap=float(strat),
                    norm_gap=float(np.max(np.abs(np.linalg.norm(x, axis=1) - 1))))
    if kind == "gaussian":           # Gaussian law: candidate = arc-cosine kernel on the data
        x, y, n_train = h6.digits()
        exact = kg.arccos_kernel(x)
        K = kg.candidate(x, "normal")
        return dict(candidate_vs_arccos=float(np.linalg.norm(K - exact) / np.linalg.norm(exact)),
                    mse_gap=rel(h6.krr_mse(K, y, n_train), h6.krr_mse(exact, y, n_train)))
    raise ValueError(kind)


CONTROLS = [
    dict(name="pair-parallel CF equals kg.kernel_cf",
         params=dict(control="parallel_cf"), expect={"parallel_vs_kg": (0.0, 1e-12)}),
    dict(name="digits split and preprocessing as frozen",
         params=dict(control="split"),
         expect={"n_train": (1257.0, 0.0), "n_test": (540.0, 0.0), "strat_gap": (0.0, 1.0),
                 "norm_gap": (0.0, 1e-12)}),
    dict(name="Gaussian law: candidate equals arc-cosine on digits",
         params=dict(control="gaussian"),
         expect={"candidate_vs_arccos": (0.0, 1e-13), "mse_gap": (0.0, 1e-10)}),
]

GRID = [dict(part="h6", law=law) for law in ("gamma1", "laplace")]


def cell(params, memo):
    if "control" in params:
        return control(params)
    law = params["law"]
    x, y, n_train = h6.digits()
    ref, coarse = h6.cf_reference_cached(x, law, WORKERS)
    nrm = np.linalg.norm(ref)
    u = float(np.linalg.norm(ref - coarse) / nrm)
    reps = h6.qcv_parallel(x, law, QMC_POINTS, [seed(params, "qmc-control", r) for r in range(QMC_REPS)],
                           WORKERS)
    diff = float(np.linalg.norm(reps.mean(0) - ref) / nrm)
    three_se = float(3 * np.sqrt(reps.var(0, ddof=1).sum() / QMC_REPS) / nrm)
    out = dict(u=u, qmc_diff=diff, qmc_3se=three_se, ref_ok=float(u <= U_MAX and diff <= three_se + u))
    mse_ref = h6.krr_mse(ref, y, n_train)
    out["mse_ref"] = mse_ref
    out["mse_u"] = rel(h6.krr_mse(coarse, y, n_train), mse_ref)   # N=64 vs N=128, on the metric
    out["mse_qmc"] = rel(h6.krr_mse(reps.mean(0), y, n_train), mse_ref)
    Y = np.eye(10)[y]
    for order, name in enumerate(("gaussian", "third", "fourth", "candidate")):
        K = kg.candidate(x, law, order)
        e, allow = kg.allowance(K, ref, u)
        out[f"{name}_allowance"] = allow
        out[f"{name}_mineig"] = kg.min_eig(K, ref)
        m = h6.krr_mse(K, y, n_train)
        out[f"{name}_mse"] = m
        out[f"{name}_mse_rel"] = rel(m, mse_ref)
    out["h6"] = float(bool(out["ref_ok"]) and out["candidate_mse_rel"] <= RULE
                      and out["gaussian_mse_rel"] > RULE)
    # descriptive only: test accuracy of the argmax prediction
    for name, K in (("ref", ref), ("candidate", kg.candidate(x, law)), ("gaussian", kg.candidate(x, law, 0))):
        Ktr = K[:n_train, :n_train]
        alpha = np.linalg.solve(Ktr + 1e-3 * np.trace(Ktr) / n_train * np.eye(n_train), Y[:n_train])
        out[f"{name}_test_acc"] = float(np.mean(np.argmax(K[n_train:, :n_train] @ alpha, 1) == y[n_train:]))

    # S5 (reported, no verdict): total runtime = kernel + KRR, against competitors reaching RULE
    levels = {}
    for method in ("sobol", "qcv"):
        k, worst = ladder(params, method, x, y, n_train, law, mse_ref)
        levels[method] = k
        out[f"{method}_level"] = float("nan") if k is None else float(k)
        out[f"{method}_worst_mse_rel"] = worst
    cf_level, cf_rel = None, float("nan")
    for N in CF_LADDER:
        cf_rel = rel(h6.krr_mse(h6.kernel_cf_pairs(x, law, N, WORKERS), y, n_train), mse_ref)
        if cf_rel <= RULE:
            cf_level = N
            break
    out["cf_level"] = float("nan") if cf_level is None else float(cf_level)
    out["cf_mse_rel"] = cf_rel
    CPU_WALL.clear()
    out["candidate_seconds"] = timed(lambda: h6.krr_mse(kg.candidate(x, law), y, n_train))
    secs = {}
    if out["gaussian_mse_rel"] <= RULE:
        secs["gaussian"] = timed(lambda: h6.krr_mse(kg.candidate(x, law, 0), y, n_train))
    for method in ("sobol", "qcv"):
        if levels[method] is not None:
            s, N = seed(params, method, 0), 2 ** levels[method]
            secs[method] = timed(lambda: h6.krr_mse(stochastic(method, x, law, N, s), y, n_train))
    if cf_level is not None:
        secs["cf"] = timed(lambda: h6.krr_mse(kg.kernel_cf(x, law, cf_level), y, n_train))
    for name in ("gaussian", "sobol", "qcv", "cf"):
        out[f"{name}_seconds"] = secs.get(name, float("nan"))
    best = min(secs.values()) if secs else float("nan")
    out["best_competitor_seconds"] = best
    out["time_ratio"] = out["candidate_seconds"] / best if secs else float("nan")
    out["cpu_wall_max"] = float(max(CPU_WALL))
    return out
