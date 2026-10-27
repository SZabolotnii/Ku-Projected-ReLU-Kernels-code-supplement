"""Freeze the H5 threshold on exploratory baseline panels only, before any fresh panel exists.

Calibration panels: the 18 generalization panels (n=32; Gamma(1), Laplace; d=16,64,128)
and the 24 primary-study panels (n=8; dense/sparse/near; d=4,16,32,64), all already
seen in baseline/. Labels use the CF reference and validity rule of this run (spec.reference), not the old
QMC means.
Rule: a panel is predicted accurate iff h5_score <= tau; tau maximises Youden's J
(sensitivity + specificity - 1) for the label allowance <= 0.003, ties to the smaller tau.
Writes h5_rule.json once; refuses to overwrite.
"""
import spec  # first: sets the one-thread BLAS environment before numpy loads
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import kg  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "h5_rule.json"
TARGET = 0.003


def panels():
    gen = kg.TRACK / "baseline/evidence/projected_generalization"
    for path in sorted(gen.glob("reference_*.npz")):
        law, d, p = path.stem.split("_")[1:]
        yield f"generalization:{law}_{d}_{p}", law, np.load(path)["inputs"]
    for law in ("gamma1", "laplace"):
        for d in (4, 16, 32, 64):
            for gi, geometry in enumerate(("dense", "sparse", "near")):
                yield f"primary:{law}_d{d}_{geometry}", law, kg.pc.make_inputs(d)[gi]


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT.name} exists: the rule is frozen")
    rows = []
    for name, law, x in panels():
        # same validity rule as the test labels: CF u <= 1e-4 and QMC agreement
        ref, u, diag = spec.reference(dict(calibration=name), x, law)
        e, allow = kg.allowance(kg.candidate(x, law), ref, u)
        rows.append(dict(panel=name, law=law, d=x.shape[1], n=len(x), u=u, ref_ok=diag["ref_ok"], error=e,
                         allowance=allow, accurate=bool(diag["ref_ok"] and allow <= TARGET),
                         score=kg.h5_score(x, law)))
        print(rows[-1], flush=True)
    scores = np.array([r["score"] for r in rows])
    label = np.array([r["accurate"] for r in rows])
    best = None
    for tau in np.unique(scores):
        pred = scores <= tau
        sens = pred[label].mean() if label.any() else 0.0
        specif = (~pred[~label]).mean() if (~label).any() else 0.0
        j = sens + specif - 1
        if best is None or j > best[0] + 1e-12:
            best = (j, float(tau), float(sens), float(specif))
    j, tau, sens, specif = best
    rule = dict(
        frozen="2026-10-25", target=TARGET, tau=tau, youden_j=j, sensitivity=sens, specificity=specif,
        rule="predict accurate iff kg.h5_score(x, law) <= tau",
        kg_sha256=hashlib.sha256((HERE / "kg.py").read_bytes()).hexdigest(),
        spec_sha256=hashlib.sha256((HERE / "spec.py").read_bytes()).hexdigest(),
        calibration=rows,
    )
    OUT.write_text(json.dumps(rule, indent=2) + "\n")
    print("tau", tau, "J", j, "sens", sens, "spec", specif)


if __name__ == "__main__":
    main()
