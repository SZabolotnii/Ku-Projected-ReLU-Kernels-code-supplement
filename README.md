# Code supplement — Second-order cumulant corrections for ReLU random-feature kernels

Code, frozen specifications, result files and the Lean project behind the article
"Second-order cumulant corrections for ReLU random-feature kernels under non-Gaussian
weights: accuracy, cost and limits".

Every number, table and figure of the article is produced by
`paper/scripts/make_tex.py` from the result files stored here. The script never reruns
an experiment. Running it regenerates `paper/numbers.tex`, `paper/tables/*.tex`,
`paper/figures/*.pdf` and `experiments/paper-derived/DERIVED.json`, and the output
matches the files the manuscript was built from byte for byte.

## Layout

| Path | What it is | Article |
|---|---|---|
| `experiments/2026-10-25-kernel-gate/` | Run 1, the kernel study: `SPEC.md` (pre-registered 25 October 2026), `kg.py` (candidate, competitors, reference, panels), `spec.py` / `spec_s2.py` (timing sessions 1 and 2), `calibrate_h5.py` and the frozen `h5_rule.json`, `analyze.py`, `results/`, `analysis/` | §4, Tables 1–3, Figures 1–2, Appendix B |
| `experiments/2026-10-26-h6-digits/` | Run 2, the handwritten-digits benchmark of H6: `SPEC.md`, `h6.py`, `spec_h6.py`, `results/` | §4.5, digits table |
| `experiments/paper-derived/DERIVED.json` | every derived value the article prints (medians, ranges, counts, slopes), keyed by name | all sections |
| `baseline/scripts/`, `baseline/evidence/` | the exploratory series that selected the two primary cells, and the panels H5 was calibrated on; `projected_core.py` and `projected_refine.py` hold the frozen candidate and competitors that `kg.py` imports | §4.1, §4.5 |
| `baseline/*_SPEC.md` | specifications of the exploratory steps | §4.1 |
| `vendor/` | copies of two helper modules the baseline scripts import | — |
| `theory/statements.tex` | the statements and proofs, as copied into the article | §3, Appendix A |
| `Lean/`, `lakefile.lean`, `lake-manifest.json`, `lean-toolchain` | Lean 4 project (v4.26.0, Mathlib) checking the steps listed in Appendix A | Appendix A |
| `paper/scripts/make_tex.py` | numbers, tables and figures of the article from the stored results | all |

## Reproduce

Python 3.13 with the versions in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python paper/scripts/make_tex.py                               # numbers, tables, figures from stored results
python experiments/2026-10-25-kernel-gate/analyze.py --s2      # verdicts of run 1 from the cell files
```

To rerun a run from scratch, use `spec.py`, `spec_s2.py` and `h6.py`. They set one BLAS
thread before NumPy loads, run the controls first, and abort on a failed control. Run 1
took about 5.8 hours of wall time on one machine. The H6 characteristic-function
reference is computed once and cached under `experiments/2026-10-26-h6-digits/cache/`,
which is not included here; a rerun recomputes it.

Lean: `lake build` in the repository root.

## Changes against the working repository

The supplement is a copy of the research repository's tracked files, with the manuscript,
review notes and internal planning documents left out. Two files were edited so that
the copy runs on its own:

- `baseline/scripts/kernel_common.py`: `UPSTREAM` pointed to an absolute path on the
  author's machine. It now points to `vendor/`, which holds the same modules.
- `experiments/*/results/*/RUN.json`: the `spec` field held an absolute path. It is now
  relative to the repository root. Nothing else in these files changed, and the result
  files they describe are unchanged.

## License

MIT, see `LICENSE`.
