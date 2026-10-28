"""Post hoc (2026-10-28): the spectrum of the training kernel in run 2 (H6, digits).

Not part of the registered study. A referee asked why the order-1 truncation (+kappa_3)
inflates the digits test MSE by about 537% although it reduces the kernel error. Kernel
ridge regression solves (K_train + lambda I) alpha = Y with lambda = 1e-3 tr(K_train)/n_train;
if K_train has an eigenvalue below -lambda, that system is indefinite. This script reports,
for each kernel of run 2, the smallest eigenvalue of K_train next to lambda.

It reruns nothing that run 2 measured. The inputs, the kernels and the CF reference are
those of run 2: h6.digits(), kg.candidate() at orders 0-3 (frozen code) and the cached
N=128 reference of h6.cf_reference_cached(). A control first recomputes run 2's stored
`*_mineig` (full matrix, divided by ||K_ref||_F) and aborts on any mismatch, so the matrices
here are the ones run 2 used. A missing cache aborts too: rebuilding it costs about
22 CPU-hours and is not done here.

Run from the track root:  python experiments/2026-10-28-h6-spectrum/spectrum.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
H6DIR = HERE.parent / "2026-10-26-h6-digits"
RUN2 = H6DIR / "results" / "h6-digits-2026-10-26"
sys.path.insert(0, str(H6DIR))
import h6  # noqa: E402  (puts the kernel-gate folder on the path and imports kg)

kg = h6.kg
OUT = HERE / "results" / "h6-spectrum-2026-10-28"
NAMES = ("gaussian", "third", "fourth", "candidate")


def stored(law):
    files = sorted(RUN2.glob(f"law={law}_part=h6__*.json"))
    if len(files) != 1:
        raise SystemExit(f"expected one run-2 result for {law}, found {len(files)}")
    return files[0], json.loads(files[0].read_text())["values"]


def main():
    x, y, n_train = h6.digits()
    tag = hashlib.sha256(x.tobytes()).hexdigest()[:12]
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for law in ("gamma1", "laplace"):
        path = h6.CACHE / f"{law}_N128_{tag}.npy"
        if not path.exists():
            raise SystemExit(f"cached reference missing: {path.name}")
        ref = np.load(path)
        src, run2 = stored(law)
        vals = dict(n_train=n_train, ref_cache=path.name,
                    ref_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    ref_norm=float(np.linalg.norm(ref)))
        for order, name in enumerate(NAMES):
            K = kg.candidate(x, law, order)
            full = kg.min_eig(K, ref)
            if abs(full - run2[f"{name}_mineig"]) > 1e-9 * max(1.0, abs(full)):
                raise SystemExit(f"control failed: {law} {name} min_eig {full} != run 2 {run2[name + '_mineig']}")
            Ktr = K[:n_train, :n_train]
            lam = 1e-3 * np.trace(Ktr) / n_train
            lo = float(np.linalg.eigvalsh(Ktr)[0])
            vals[f"{name}_train_mineig"] = lo
            vals[f"{name}_lambda"] = float(lam)
            vals[f"{name}_mineig_over_lambda"] = lo / lam
            vals[f"{name}_shifted_mineig"] = lo + lam
        cell = dict(params=dict(law=law, part="spectrum"), values=vals,
                    control=f"stored *_mineig of {src.name} reproduced to 1e-9",
                    note="post hoc, outside the registered study")
        name = f"law={law}_part=spectrum.json"
        (OUT / name).write_text(json.dumps(cell, indent=1, sort_keys=True) + "\n")
        written.append(name)
    lines = [f"{hashlib.sha256((OUT / n).read_bytes()).hexdigest()}  {n}" for n in written]
    (OUT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    for n in written:
        v = json.loads((OUT / n).read_text())["values"]
        print(n, {k: f"{v[k]:.3g}" for k in v if k.endswith(("mineig_over_lambda", "shifted_mineig"))})


if __name__ == "__main__":
    main()
