# Run SPEC — confirmatory kernel gate (run 1) and frozen H6 protocol (run 2)

Frozen 2026-10-25, before the confirmatory grid ran. Author decisions taken on
2026-10-25: CF-inversion reference with an independent QMC control; H6 as a
separate run 2 whose protocol is frozen here; H5 threshold calibrated on
exploratory baseline panels only. Hypotheses, success and kill criteria are those of
`BRIEF.md`; nothing below relaxes them.

## Status of the cells

The primary cells (dense Gamma(1) d=64, dense Laplace d=16) were chosen after the
exploratory baseline series (`baseline/`, Spec.md §4). This run is the first
confirmatory test of them, and it reports the whole (law, d) grid so the choice can be
judged against the cells that were not chosen (critique W2).

## Method under test

The frozen second-order candidate `matrices(x, law, 3)` of
`baseline/scripts/projected_core.py`, unchanged; ablations are orders 0 (Gaussian K_G),
1 (+κ3) and 2 (+κ4). The transfer laws Gamma(4) and the standardised uniform enter
through their exact (κ3, κ4), added to the frozen cumulant table at run time
(`kg.py`), not by editing the file.

## Reference

- Primary: deterministic 2-D characteristic-function inversion (`kg.kernel_cf`),
  U₊V₊ = (|U||V| + |U|V + U|V| + UV)/4 with |u| = (2/π)∫(1−cos tu)/t² dt; Gauss–Legendre
  on t = u/(1−u), N = 128 nodes per axis. The reference error u is the relative
  Frobenius difference between N = 128 and N = 64. On the Gaussian law the reference
  reproduces the arc-cosine kernel to 4.3e-6 with u = 1.3e-5, so u overstates the error
  about threefold.
- Control: independent quadratic-CV scrambled Sobol, 8 scrambles × 2^20 points, seeds
  disjoint from every method seed.
- A panel is REFERENCE_LIMITED, and counts as a failure (BRIEF S1), unless
  u ≤ 1e-4 and ‖QMC − CF‖/‖CF‖ ≤ 3·SE_QMC + u.
- Allowance (e+u)/(1−u) as in BRIEF A4. With the CF reference, u is a quadrature
  difference rather than 3 SE; it remains an empirical criterion, not a proved bound.

## Grid (240 cells + 4 controls; one cell = one input panel, BRIEF A5)

| Part | Laws | d | n | Geometry | Panels | Measures |
|---|---|---|---|---|---|---|
| gate (H3, H4, S1, S2, K2–K4) | Gamma(1) d=64; Laplace d=16 | — | 32, 128 | dense | 10 | accuracy + time-to-target |
| (law, d) grid | Gamma(1), Laplace, Gamma(4), uniform | 16, 64, 128 | 32 | dense | 10 | accuracy |
| H2 d-sweep | Gamma(1) | 32, 256 (+16, 64, 128 above) | 32 | dense | 10 | accuracy |
| H5 geometries | Gamma(1), Laplace | 16, 64 | 32 | sparse, near | 10 | accuracy |

Panels (`kg.panel`): rows N(0, I_d), normalised; seed `[20261026, d, n, index, geometry]`,
disjoint from every baseline seed; the same panel serves every law at a given (d, n).
Sparse: max(2, d/8) random non-zero coordinates per row. Near: a quarter of the rows
rotated from other rows by angles geomspace(1e-3, 1e-1), plus one exact duplicate and
one antipode.

## Competitors and time-to-target (gate cells)

Gaussian K_G; scrambled Sobol and quadratic-CV Sobol on N = 2^10…2^22; conditional
Sobol (one coordinate integrated exactly) on N = 2^10…2^18; CF inversion on
N = 8, 12, 16, 24, 32, 48, 64, 96 nodes. A stochastic competitor is eligible at the
smallest N at which all 8 scrambles reach allowance ≤ 0.003 with min eigenvalue
/‖R‖ ≥ −1e-10; CF at the smallest N that reaches 0.003. No eligible N within the
range: that competitor is limited, not timed. The closest stage-4 prior art (Wu et al.,
arXiv 2605.05179) takes expectations over Gaussian inputs for fixed weights and does not
apply to an expectation over weights. CF inversion (critique W17) is therefore the
strongest applicable competitor added (BRIEF Scope, K4).

Timing (A6): full matrix including node generation, one BLAS thread, 1 warm-up +
5 repeats, median; methods in a seeded random order per panel; one session. numpy here
uses Accelerate, which threadpoolctl cannot limit, so `VECLIB_MAXIMUM_THREADS=1` (and the
OpenBLAS/OMP equivalents) are set before numpy loads. Every cell records the largest
CPU/wall ratio of its timed calls (`cpu_wall_max`), so a multi-threaded timing is visible.
Time ratio = candidate / fastest eligible competitor.

Panel status: REFERENCE_LIMITED → CANDIDATE_FAIL (allowance > 0.003 or not PSD) →
COMPETITOR_LIMITED (no competitor eligible) → PASS (ratio ≤ 0.8) / NO_SPEED_GAIN.

## Decision rules (computed by `analyze.py` from the cell files)

- **S1 / H3 / H4**, per (law, d, n): ≥ 9/10 panels PASS. Both n of a cell must pass.
  The verdict needs the same result in timing session 2 (`spec_s2.py`, same panels,
  run on another day; accuracy is deterministic and is recomputed, not copied).
- **S2**: K_G allowance > 0.003 on ≥ 9/10 panels of the cell. Only panels with a valid
  reference count as K_G failures.
- **K2**: S1 fails in either primary cell (either session). **K3**: in every one of
  the 12 (law, d) cells where the candidate passes accuracy on ≥ 9/10, K_G also does.
  **K4**, per primary cell (both n): S1 holds with competitors {K_G, Sobol,
  quadratic-CV Sobol} but fails once conditional Sobol and CF inversion are added.
- **H2 falsifier**: OLS of log e on log d over the 50 dense Gamma(1) n=32 panels
  (d = 16, 32, 64, 128, 256; e = candidate relative error against CF), excluding and
  counting REFERENCE_LIMITED panels; falsified if the 95% upper confidence bound of the
  slope is above −1.2. Panels with e < 10u are reported as reference-resolution-limited,
  and the fit is repeated without them.
- **H5**: rule frozen in `h5_rule.json` by `calibrate_h5.py` before this grid: predict
  accurate iff `kg.h5_score ≤ τ`, τ maximising Youden's J on the 42 baseline panels.
  Calibration labels use the same validity rule as the test labels. Test set: the 120
  n=32 panels of Gamma(1)/Laplace × d 16, 64 × dense/sparse/near. Label accurate =
  reference ok and allowance ≤ 0.003. Falsified if AUC(−score) ≤ 0.75, if the AUC is
  undefined (one class empty), or if any panel predicted accurate has allowance > 0.006.
  `analyze.py` refuses to run if `kg.py` or `spec.py` changed after the rule was frozen.
  Sparse panels with few non-zero coordinates can be REFERENCE_LIMITED (u slightly above
  1e-4); they are labelled inaccurate, as S1 requires, and their count is reported.
- **H1 / S4**: control `algebra` — every T_pq with p+q ≤ 6, ρ ∈ {±0.95, …}, frozen and
  independent closed forms against adaptive quadrature, ≤ 1e-10; control `structure`
  checks symmetry, homogeneity and the ρ → ±1 limits. A miss aborts the run.

## Run 2 — H6 protocol (frozen now, run only after run 1)

Dataset `sklearn.datasets.load_digits` (1797 × 64, bundled, no download); features
standardised on the training split, then each row normalised; split
`train_test_split(test_size=0.3, random_state=20261025, stratify=y)`. Kernel ridge
regression on one-hot targets with λ = 1e-3·tr(K_train)/n_train; metric = test mean
squared error. Laws Gamma(1) and Laplace. Reference kernel: CF inversion (N = 128,
same validity rule). H6 holds iff for each law |MSE_cand − MSE_ref|/MSE_ref ≤ 1% and
|MSE_G − MSE_ref|/MSE_ref > 1%; total runtime against the fastest competitor reaching
the same metric is reported for S5.

## Deviations disclosed before the run

- A dry test computed one confirmatory panel (gate, Laplace d=16, n=32, panel 0;
  PASS) and the four controls before this file was frozen. No frozen parameter depends
  on it.
- The author was offered a CF tolerance target of 1e-6. Convergence is O(N^-2),
  because of cancellation near t, s → 0, so the frozen validity bound is u ≤ 1e-4,
  with observed u ≈ 1.3e-5.
- Conditional Sobol is capped at 2^18 for cost; beyond the cap it is limited, not a
  loser. This is stated in the report.
