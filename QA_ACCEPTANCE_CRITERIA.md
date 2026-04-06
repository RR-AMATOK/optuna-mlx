# ============================================================================
# OPTUNA-MLX QA ACCEPTANCE CRITERIA
# Product Owner -> QA Team
# ============================================================================
# Created: 2026-04-05
# Author: PO (Claude)
# Status: ACTIVE
# ============================================================================

# QA Acceptance Criteria — All Phases

## General Testing Principles

1. **Parity first, performance second** (ADR-003). Every test must prove MLX
   produces the same results as the original backend before we measure speed.
2. **Dual-run testing:** For each numerical test, run with both the original
   backend (torch/numpy) and MLX. Compare outputs.
3. **Deterministic seeds:** All randomized tests must use fixed seeds so
   results are reproducible across runs.
4. **Tolerance tiers:** Different operations tolerate different floating-point
   drift. Use the tier appropriate to the operation, not a blanket tolerance.

---

## Tolerance Tiers

| Tier | rtol | atol | Used For |
|------|------|------|----------|
| **EXACT** | 1e-12 | 1e-14 | Simple math (erf, exp, log) |
| **TIGHT** | 1e-10 | 1e-12 | Kernel values, dot products |
| **STANDARD** | 1e-8 | 1e-10 | Cholesky, solve_triangular, posterior |
| **MODERATE** | 1e-6 | 1e-8 | Log marginal likelihood, composite |
| **LOOSE** | 1e-3 | 1e-5 | Optimizer output (L-BFGS-B result) |
| **STATISTICAL** | N/A | N/A | KS test p > 0.01 for distributions |

---

## Phase 0: Environment & Infrastructure

### AC-0.1: MLX Installation
- [ ] `import mlx.core as mx` succeeds
- [ ] `mx.default_device()` reports a GPU device (not cpu)
- [ ] `mx.array([1.0], dtype=mx.float64)` creates a float64 array

### AC-0.2: Float64 GPU Performance
- [ ] float64 Cholesky (100x100) runs on GPU
- [ ] float64 is no more than 5x slower than float32 for Cholesky
- [ ] If float64 is >5x slower, document and escalate to PO for ADR-009 decision

### AC-0.3: MLX API Coverage
- [ ] All APIs listed in NOTES.md "Available in MLX" section are confirmed present
- [ ] Missing APIs are documented with workaround plan

### AC-0.4: Utility Module
- [ ] `from optuna._mlx import is_mlx_available` works
- [ ] `is_mlx_available()` returns True on Apple Silicon with MLX installed
- [ ] `is_mlx_available()` returns False when MLX is not installed
- [ ] `import optuna` does NOT fail when MLX is not installed

### AC-0.5: pyproject.toml
- [ ] `pip install -e ".[mlx]"` installs mlx>=0.5.0
- [ ] `pip install -e ".[optional]"` on macOS installs mlx
- [ ] `pip install -e ".[optional]"` on Linux does NOT attempt to install mlx
- [ ] `pip install -e .` (no extras) does NOT install mlx

### AC-0.6: Baseline Benchmarks
- [ ] Benchmark results exist for all 7 operations listed in DEV_BRIEF.md 0D
- [ ] Results saved to `benchmarks/results_torch_baseline.json`
- [ ] Results include median, mean, std, n_runs, hardware info

---

## Phase 1: Gaussian Process Module

### AC-1.1: Zero Torch Dependency
- [ ] `grep -r "import torch" optuna/_gp/` returns zero matches
- [ ] `grep -r "torch\." optuna/_gp/` returns zero matches (excluding comments)
- [ ] `python -c "import optuna._gp.gp"` succeeds without torch installed
- [ ] `python -c "import optuna._gp.acqf"` succeeds without torch installed
- [ ] `python -c "import optuna._gp.prior"` succeeds without torch installed
- [ ] `python -c "import optuna._gp.optim_mixed"` succeeds without torch installed

### AC-1.2: Matern 5/2 Kernel Parity
**Test:** Compare `matern52_kernel(sqd)` for MLX vs original torch for:
- [ ] sqd = 0.0 (edge case: zero distance) — Tier TIGHT
- [ ] sqd = 1e-15 (near-zero) — Tier TIGHT
- [ ] sqd = [0.01, 0.1, 1.0, 10.0, 100.0] (typical range) — Tier TIGHT
- [ ] sqd = 1e6 (large distance, kernel -> 0) — Tier TIGHT
- [ ] Random 100x100 matrix of squared distances — Tier TIGHT
- [ ] Gradient: `mx.grad(matern52_kernel)(sqd)` vs torch manual backward — Tier STANDARD

### AC-1.3: Kernel Matrix Parity
**Test:** `GPRegressor.kernel()` with same inputs produces same matrix.
- [ ] 2 params, 10 trials — Tier TIGHT
- [ ] 10 params, 50 trials — Tier TIGHT
- [ ] 50 params, 100 trials — Tier TIGHT
- [ ] Mixed categorical + continuous params — Tier TIGHT

### AC-1.4: Cholesky & Cache Matrix Parity
**Test:** `GPRegressor._cache_matrix()` produces same L and inv_Y.
- [ ] 10 trials — Tier STANDARD
- [ ] 50 trials — Tier STANDARD
- [ ] 100 trials — Tier STANDARD
- [ ] Near-singular case (low noise_var) — verify it either succeeds with same
      result or fails gracefully with same error handling

### AC-1.5: Posterior Parity
**Test:** `GPRegressor.posterior(x)` for single and batched x.
- [ ] Single point prediction — Tier STANDARD
- [ ] Batch of 10 points — Tier STANDARD
- [ ] Batch of 100 points — Tier STANDARD
- [ ] Variance is non-negative for all inputs — absolute check
- [ ] Mean is within training data range (sanity check) — loose check

### AC-1.6: Marginal Log Likelihood Parity
**Test:** `GPRegressor.marginal_log_likelihood()` same GPR -> same MLL.
- [ ] 10 trials, 2 params — Tier MODERATE
- [ ] 50 trials, 10 params — Tier MODERATE
- [ ] Differentiability: `mx.grad(mll_fn)(params)` produces finite gradients — check

### AC-1.7: Kernel Fitting Parity
**Test:** `fit_kernel_params(X, Y, is_categorical, ...)` -> similar fitted params.
- [ ] Simple 2D problem — Tier LOOSE (optimizer path-dependent)
- [ ] 10D problem — Tier LOOSE
- [ ] Fitted MLL is within 1% of torch-fitted MLL — relative check
- [ ] Convergence: optimizer reports success — boolean check

### AC-1.8: Acquisition Function Parity
**Test:** Each acqf with same GP posterior produces same values.
- [ ] `standard_logei(z)` for z in [-30, -10, -1, 0, 1, 10] — Tier STANDARD
- [ ] `logei(mean, var, f0)` — Tier STANDARD
- [ ] `UCB.eval_acqf(x)` — Tier STANDARD
- [ ] `LCB.eval_acqf(x)` — Tier STANDARD
- [ ] `LogPI.eval_acqf(x)` — Tier STANDARD
- [ ] `LogEHVI.eval_acqf(x)` (multi-objective) — Tier STANDARD
- [ ] `eval_acqf_with_grad`: value matches, gradient matches — Tier STANDARD

### AC-1.9: Special Functions Parity
- [ ] `erfc_mlx(x)` vs `scipy.special.erfc(x)` for x in [-5, -1, 0, 1, 5, 20] — Tier TIGHT
- [ ] `erfcx_mlx(x)` vs `scipy.special.erfcx(x)` for same range — Tier TIGHT
- [ ] `log_ndtr_mlx(x)` vs `scipy.special.log_ndtr(x)` for x in [-30, -5, 0, 5] — Tier STANDARD
- [ ] Tail behavior: `erfc_mlx(20.0)` should not be exactly 0.0 — precision check

### AC-1.10: Optimization Pipeline Parity
**Test:** `optimize_acqf_mixed()` end-to-end.
- [ ] Same study state -> suggested params are within 5% of torch suggestion
      (the pipeline has multiple random elements so exact match is not expected)
- [ ] 2048-candidate batch evaluation completes without error
- [ ] Result is a valid point within search space bounds

### AC-1.11: Edge Cases
- [ ] Infinite objective values handled (`warn_and_convert_inf`) — same behavior
- [ ] Single trial (minimum data) — no crash, reasonable posterior
- [ ] Two trials — no crash
- [ ] 1000+ trials — no OOM, completes in reasonable time
- [ ] All categorical params — correct kernel computation
- [ ] All continuous params — correct kernel computation
- [ ] Mixed params — correct kernel computation

### AC-1.12: Integration Test
- [ ] `study.optimize(objective, n_trials=50)` with `GPSampler` completes
- [ ] Optimized value is reasonable (within 10% of torch-backend result on same problem)
- [ ] No warnings about NaN or Inf during optimization
- [ ] Memory usage is stable (no leak over 200 trials)

### AC-1.13: Existing Test Suite
- [ ] `pytest tests/gp_tests/` — ALL PASS
- [ ] `pytest tests/samplers_tests/test_gp.py` — ALL PASS
- [ ] No new test failures introduced in unrelated test files

### AC-1.14: Performance Benchmarks (Phase 1)
Run same benchmarks as baseline (AC-0.6) with MLX backend.
- [ ] `kernel()` — record speedup factor vs torch
- [ ] `_cache_matrix()` — record speedup factor
- [ ] `posterior()` — record speedup factor
- [ ] `marginal_log_likelihood()` — record speedup factor
- [ ] `fit_kernel_params()` — record speedup factor
- [ ] `eval_acqf` on 2048 candidates — record speedup factor
- [ ] End-to-end `study.optimize()` — record speedup factor
- [ ] Save to `benchmarks/results_mlx_phase1.json`
- [ ] **No regression:** MLX must not be >20% slower than torch on ANY operation
      (if it is, escalate to PO — may indicate a bug, not a real regression)

### AC-1.15: PO Sign-off Gate
- [ ] All AC-1.1 through AC-1.14 pass
- [ ] DEV_BRIEF.md "Definition of Done" checklist complete
- [ ] CHANGELOG_MLX.md updated with Phase 1 changes
- [ ] No open blockers or known issues

---

## Phase 2: TPE Sampler

### AC-2.1: erf Replacement
- [ ] `_erf.py` either deleted or reduced to a thin wrapper around `mx.erf()`
- [ ] `mx.erf(x)` vs `math.erf(x)` for 1000 random values — Tier EXACT
- [ ] Performance: MLX erf on 10,000 elements is faster than current Python loop

### AC-2.2: Truncated Normal Parity
- [ ] `_ndtr()`: MLX vs `scipy.stats.norm.cdf()` — Tier TIGHT
- [ ] `_log_ndtr()`: MLX vs `scipy.stats.norm.logcdf()` — Tier STANDARD
- [ ] `_log_gauss_mass()`: MLX vs original numpy — Tier TIGHT
- [ ] `ppf()`: MLX vs `scipy.stats.truncnorm.ppf()` — Tier STANDARD
- [ ] `rvs()`: KS test against known truncated normal — Tier STATISTICAL
- [ ] `logpdf()`: MLX vs original numpy — Tier STANDARD
- [ ] `_ndtri_exp()` Newton iteration converges for all test inputs

### AC-2.3: Probability Distributions Parity
- [ ] `log_pdf()` (THE HOT PATH): MLX vs numpy for 256 samples, 100 components — Tier STANDARD
- [ ] `sample()`: Statistical validity (KS test) — Tier STATISTICAL
- [ ] `_unique_inverse_2d()`: Correct deduplication — exact match

### AC-2.4: Parzen Estimator Parity
- [ ] `compute_sigmas()`: MLX vs numpy — Tier TIGHT
- [ ] Weight normalization: weights sum to 1.0 — exact check
- [ ] Categorical distribution computation: correct probabilities

### AC-2.5: TPE Integration
- [ ] `study.optimize(objective, n_trials=100)` with `TPESampler` completes
- [ ] Optimization trajectory is comparable to numpy backend
- [ ] `pytest tests/samplers_tests/tpe_tests/` — ALL PASS

### AC-2.6: TPE Performance
- [ ] `log_pdf` with 256 samples, 100 components — record speedup
- [ ] `log_pdf` with 1024 samples, 500 components — record speedup
- [ ] End-to-end TPESampler `study.optimize()` — record speedup
- [ ] Save to `benchmarks/results_mlx_phase2.json`

### AC-2.7: PO Sign-off Gate
- [ ] All AC-2.1 through AC-2.6 pass
- [ ] CHANGELOG_MLX.md updated
- [ ] No open blockers

---

## Phase 3: Hypervolume

### AC-3.1: Hypervolume Parity
- [ ] `_compute_2d()`: MLX vs numpy — Tier TIGHT
- [ ] `_compute_3d()`: MLX vs numpy for 10, 50, 200 points — Tier TIGHT
- [ ] `compute_hypervolume()`: Full dispatch logic correct for 2D, 3D, ND
- [ ] Recursive WFG (`_compute_hv`) produces correct results for 4D, 5D

### AC-3.2: Hypervolume Edge Cases
- [ ] Single point — correct
- [ ] Two points (one dominates) — correct
- [ ] All points on Pareto front — correct
- [ ] Reference point at boundary — correct

### AC-3.3: Hypervolume Performance
- [ ] Benchmark for n_points in [10, 50, 200, 1000] — record speedup
- [ ] `pytest tests/hypervolume_tests/` — ALL PASS

### AC-3.4: PO Sign-off Gate
- [ ] All AC-3.x pass, CHANGELOG updated

---

## Phase 4: Multi-Objective Pareto Front

### AC-4.1: Pareto Front Parity
- [ ] `_is_pareto_front_nd()`: MLX vs numpy for 100, 500, 2000 trials — exact match
- [ ] `_is_pareto_front_2d()`: MLX vs numpy — exact match
- [ ] `_calculate_nondomination_rank()`: same ranks for same inputs

### AC-4.2: Multi-Objective Integration
- [ ] Multi-objective `study.optimize()` with 2-3 objectives completes
- [ ] Pareto front matches numpy backend
- [ ] `pytest tests/study_tests/test_multi_objective.py` — ALL PASS

### AC-4.3: PO Sign-off Gate
- [ ] All AC-4.x pass, CHANGELOG updated

---

## Phase 5: NSGAII

### AC-5.1: Crossover Parity
- [ ] SBX, BLX-alpha, SPX, UNDX, VSBX produce statistically similar offspring
- [ ] Elite selection matches numpy backend

### AC-5.2: NSGAII Integration
- [ ] `study.optimize()` with `NSGAIISampler` completes
- [ ] `pytest tests/samplers_tests/test_nsgaii.py` — ALL PASS

### AC-5.3: PO Sign-off Gate
- [ ] All AC-5.x pass, CHANGELOG updated

---

## Phase 6: Optimization

### AC-6.1: Performance Targets
- [ ] GP module: >= 2x speedup over torch on kernel computation (n>=100)
- [ ] TPE module: >= 2x speedup over numpy on log_pdf (n_samples>=256)
- [ ] End-to-end: >= 1.5x overall speedup on 100-trial study
- [ ] If targets not met, document bottlenecks and escalate to PO

### AC-6.2: Memory Efficiency
- [ ] No memory leak over 1000-trial study (RSS growth < 50MB)
- [ ] Peak memory usage within 2x of torch/numpy baseline

### AC-6.3: Final Regression Suite
- [ ] `pytest tests/` — full test suite passes (minus known platform-specific skips)
- [ ] All benchmark results saved and compared against baselines
- [ ] Performance regression report generated

### AC-6.4: Final PO Sign-off
- [ ] All phase gates passed
- [ ] README_MLX.md updated with final benchmarks
- [ ] CHANGELOG_MLX.md complete
- [ ] Ready for merge to master (or PR to upstream)

---

## QA Tooling Requirements

To execute these tests, QA needs:
1. Apple Silicon Mac (M1 or later) with macOS 13+
2. Python 3.9+ environment with: `pip install -e ".[mlx,test]"`
3. The original torch backend available for comparison: `pip install torch`
4. `pytest`, `scipy`, `numpy` (from test deps)
5. Access to `benchmarks/` scripts
6. Ability to run long benchmarks (some take 5-10 minutes)

## QA Reporting Format

For each phase, deliver:
```
Phase X QA Report
=================
Date: YYYY-MM-DD
Tester: [name]
Hardware: [chip, RAM, macOS version]
Python: [version]
MLX: [version]

AC-X.Y: PASS/FAIL [details if fail]
...

Benchmark Summary:
  [operation]: [time_mlx] vs [time_baseline] = [speedup]x

Open Issues:
  - [any failures, concerns, observations]

Recommendation: PASS / CONDITIONAL PASS / FAIL
```
