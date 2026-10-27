"""Run 2 (H6): digits data, a pair-parallel CF-inversion reference, kernel ridge regression.

The protocol is the one frozen in experiments/2026-10-25-kernel-gate/SPEC.md (run 2).
Kernels come from that run's kg.py, unchanged: the candidate, the QMC control and the
CF inversion. `kernel_cf_pairs` is kg.kernel_cf with its pair loop split across processes,
the same formulas and nodes. The control `parallel_cf` in spec_h6.py shows it equals
kg.kernel_cf to rounding. The split is needed because one full digits matrix at N = 128
costs about 22 CPU-hours.
"""
import hashlib
import multiprocessing as mp
import os
import sys
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-25-kernel-gate"))
import kg  # noqa: E402

CACHE = HERE / "cache"            # reference matrices: a cache keyed by input hash, never a record
SPLIT_SEED = 20261025
TEST_SIZE = 0.3


def digits():
    """(x, y, n_train): train rows first. Standardised on the training split, rows unit norm."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    data = load_digits()
    xtr, xte, ytr, yte = train_test_split(data.data, data.target, test_size=TEST_SIZE,
                                          random_state=SPLIT_SEED, stratify=data.target)
    scaler = StandardScaler().fit(xtr)
    x = scaler.transform(np.vstack([xtr, xte]))
    x /= np.linalg.norm(x, axis=1)[:, None]
    return x, np.concatenate([ytr, yte]), len(xtr)


def krr_mse(K, y, n_train):
    """Test MSE of kernel ridge regression on one-hot targets, lambda = 1e-3 tr(K_train)/n_train."""
    Y = np.eye(10)[y]
    Ktr = K[:n_train, :n_train]
    lam = 1e-3 * np.trace(Ktr) / n_train
    alpha = np.linalg.solve(Ktr + lam * np.eye(n_train), Y[:n_train])
    pred = K[n_train:, :n_train] @ alpha
    return float(np.mean((pred - Y[n_train:]) ** 2))


_W = {}


def _init(x, law, N):
    t, wt = kg._nodes(N)
    la, ar = kg.logcf(law, t[:, None, None] * x[None])
    _W.update(x=x, law=law, t=t, wt=wt, R1=np.exp(la.sum(-1)) * np.cos(ar.sum(-1)))


def _pairs(job):
    """A-term of kg.kernel_cf for the pairs (a, b); same loop, same order of summation."""
    a, b = job
    x, law, t, wt, R1 = _W["x"], _W["law"], _W["t"], _W["wt"], _W["R1"]
    ws = wt / t ** 2
    xa, xb = x[a], x[b]
    part = np.zeros(len(a))
    for k in range(len(t)):
        za = t[k] * xa
        out = 0.0
        for sign in (1.0, -1.0):
            z = za[None] + sign * t[:, None, None] * xb[None]
            l2, a2 = kg.logcf(law, z)
            out = out + 0.5 * (np.exp(l2.sum(-1)) * np.cos(a2.sum(-1)))
        num = 1 - R1[k, a][None] - R1[:, b] + out
        part += wt[k] / t[k] ** 2 * (ws @ num)
    return part


def kernel_cf_pairs(x, law, N, workers, chunk=512):
    """kg.kernel_cf(x, law, N) with the pair loop on `workers` processes."""
    norms = np.linalg.norm(x, axis=1)
    x = x / norms[:, None]
    n = len(x)
    t, wt = kg._nodes(N)
    la, ar = kg.logcf(law, t[:, None, None] * x[None])
    psi1 = np.exp(la.sum(-1) + 1j * ar.sum(-1))
    B = np.zeros((n, n))
    for k in range(N):                      # kg.kernel_cf's einsum, one node at a time (memory)
        M = kg.dcf_over_cf(law, t[k] * x) @ x.T
        B += -2 / np.pi * wt[k] / t[k] ** 2 * (-1j * psi1[k][:, None] * M).real
    ii, jj = np.triu_indices(n)
    jobs = [(ii[lo:lo + chunk], jj[lo:lo + chunk]) for lo in range(0, len(ii), chunk)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(x, law, N)) as pool:
        acc = np.concatenate(pool.map(_pairs, jobs, chunksize=4))
    A = np.zeros((n, n))
    A[ii, jj] = acc
    A[jj, ii] = acc
    K = (4 / np.pi ** 2 * A + B + B.T + x @ x.T) / 4
    return K * np.outer(norms, norms)


def cf_reference_cached(x, law, workers):
    """(K_128, K_64), each computed once and cached on disk under the hash of (x, law, N)."""
    CACHE.mkdir(exist_ok=True)
    tag = hashlib.sha256(x.tobytes()).hexdigest()[:12]
    out = []
    for N in (128, 64):
        path = CACHE / f"{law}_N{N}_{tag}.npy"
        if path.exists():
            out.append(np.load(path))
            continue
        K = kernel_cf_pairs(x, law, N, workers)
        np.save(path, K)
        out.append(K)
    return out


def _qcv(args):
    x, law, n_points, seed = args
    return kg.qcv_sobol(x, law, n_points, seed)


def qcv_parallel(x, law, n_points, seeds, workers):
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, len(seeds))) as pool:
        return np.array(pool.map(_qcv, [(x, law, n_points, s) for s in seeds]))
