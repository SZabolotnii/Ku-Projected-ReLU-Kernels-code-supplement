# Run SPEC — run 2, H6 on sklearn digits

Frozen 2026-10-26, before any digits kernel other than the Gaussian control was computed.
The decision rule is the one frozen on 2026-10-25 in
`experiments/2026-10-25-kernel-gate/SPEC.md` (section "Run 2 — H6 protocol"); it is
restated here, not changed. This file adds only what that section left open: the
implementation of the reference, the controls, and the S5 competitor ladders.

## Protocol (as frozen 2026-10-25)

- Data: `sklearn.datasets.load_digits` (1797 × 64, bundled with scikit-learn 1.9.0, no
  download). Split `train_test_split(test_size=0.3, random_state=20261025, stratify=y)`;
  1257 training, 540 test rows. Features standardised with a scaler fitted on the
  training split, then every row normalised to unit length.
- Model: kernel ridge regression on one-hot targets, λ = 1e-3·tr(K_train)/n_train;
  metric = test mean squared error.
- Laws: Gamma(1) and Laplace (standardised weights).
- Reference: CF inversion, N = 128, u = relative Frobenius difference to N = 64. The
  validity rule is the one used in run 1: u ≤ 1e-4 and ‖QMC − CF‖/‖CF‖ ≤ 3·SE + u, with
  the QMC control being quadratic-CV Sobol, 8 × 2^20 points.
- **H6 holds** for a law iff the reference is valid,
  |MSE_cand − MSE_ref|/MSE_ref ≤ 1 %, and |MSE_G − MSE_ref|/MSE_ref > 1 %. A law whose
  reference is invalid is undetermined, not a pass. H6 needs both laws.

## Implementation (fixed now)

- `h6.kernel_cf_pairs` evaluates the pair loop of `kg.kernel_cf` on 9 processes. It uses
  the same nodes and formulas, and the B term is accumulated one node at a time for
  memory. A full digits matrix at N = 128 costs about 22 CPU-hours. The control
  `parallel_cf` requires agreement with `kg.kernel_cf` to ≤ 1e-12; the dry test gave
  1.2e-16.
- The full 1797 × 1797 matrix is computed for every method, including the test-test
  block that KRR does not use, so that every method does the same work.
- Reference matrices are cached in `cache/`, keyed by the input hash. The cache is not
  evidence. The cell JSON in `results/` is the record.
- The descriptive extras are not part of the rule:
  - `mse_u`: the MSE difference between N = 64 and N = 128;
  - `mse_qmc`: the MSE of the QMC mean against the reference;
  - the argmax test accuracy;
  - the ablations at orders 1 and 2.

## Controls (a miss aborts the run)

- `parallel_cf`: parallel CF equals `kg.kernel_cf` (Gamma(1), d = 64, 24 rows, N = 32), ≤ 1e-12.
- `split`:
  - 1257 / 540 rows;
  - every class in the test split within 1 row of 30 %;
  - row norms 1 to ≤ 1e-12.
- `gaussian`: under the normal law the candidate equals the arc-cosine kernel on the digits
  rows (≤ 1e-13) and gives the same test MSE (≤ 1e-10).

## S5 (reported, no verdict)

Total runtime is kernel construction plus KRR, on one BLAS thread, in the main process,
after every worker pool is closed. A call taking ≤ 30 s gets a warm-up and 3 repeats
(median); a slower call is measured once. A competitor counts when it reaches the
metric within 1 % of MSE_ref:

- **K_G**: if within 1 %.
- **Scrambled Sobol and quadratic-CV Sobol, 2^10 … 2^20 points**: the smallest level at
  which all 8 scrambles are within 1 %.
- **CF inversion, N ∈ {8, 12, 16, 24, 32}**: the smallest N within 1 %.

Conditional Sobol is not run. Its pair-batched evaluation needs O(256·n²) memory, about
3 GB per batch at n = 1797. This is stated in the report.
