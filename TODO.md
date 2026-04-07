# ============================================================================
# OPTUNA-MLX TODO LIST
# "A to-do list is just a bucket list for people who aren't dying."
#  - Howard Wolowitz (probably while procrastinating on his thesis)
# ============================================================================
# Created: 2026-04-05
# Last Updated: 2026-04-05
# Project: Optuna-MLX (Apple Silicon GPU Acceleration)
# ============================================================================
# Legend: [ ] Todo  [x] Done
# ============================================================================

## Pre-Phase: QA Fix-Now Items (from QA_REPORT.md, triaged in PO_QA_RESPONSE.md)
- [x] Install MLX 0.31.1 + PyTorch 2.11.0 in dev environment (C-1)
- [x] Commit all documentation files to dev branch (C-2, L-2)
- [x] Fix all LOC counts in docs (-1 each, see H-1 for exact values)
- [x] Fix README_MLX.md architecture diagram — mark planned files clearly (M-4)
- [x] Add early import check to GPSampler.__init__ with clear error message (M-5)
- [x] Clarify ADR-009 status wording (L-4)

## Phase 0: Project Setup (The Pilot Episode) — COMPLETE
- [x] Analyze Optuna codebase and identify acceleration targets
- [x] Set up dev branch
- [x] Create project documentation (NOTES, DECISIONS, TODO, README)
- [x] Document architectural decisions (ADR-001 through ADR-010)
- [x] Deep code analysis: map every function/line to MLX equivalent
- [x] Set up MLX development environment and verify GPU access
  - [x] `pip install mlx` and verify `import mlx.core as mx`
  - [x] Verify `mx.default_device()` reports GPU
  - [x] Verify float64 support: CPU only via `mx.stream(mx.cpu)` (ADR-009)
  - [x] Benchmark float64 vs float32 on Cholesky (100x100 matrix)
- [x] Create MLX utility module (`optuna/_mlx/__init__.py`)
  - [x] Backend detection: `HAS_MLX` flag
  - [x] Lazy import pattern matching existing optuna style
  - [x] `get_array_module()` helper function
- [x] Add MLX to optional dependencies in pyproject.toml
  - [x] Add `"mlx>=0.5.0"` to `[project.optional-dependencies]`
  - [x] Create `mlx` extra: `mlx = ["mlx>=0.5.0"]`
- [x] Set up benchmark suite for before/after comparison
  - [x] GP kernel matrix computation (vary n_trials: 10, 50, 100, 500)
  - [x] GP acquisition function evaluation (vary n_candidates: 256, 1024, 4096)
  - [ ] TPE log_pdf computation (vary n_samples * n_components) — Phase 2
  - [ ] Pareto front determination (vary n_trials: 100, 500, 1000) — Phase 4
  - [x] Full study.optimize() end-to-end timing
- [x] Document ADR-011: Backend fallback strategy — Option B (hard fork, MLX-only)
- [x] Document ADR-012: Autograd pattern for _fit_kernel_params

## Phase 0.5: Autograd Risk Gate — PASSED
# Added 2026-04-05 by PO. See PO_ROADMAP_REVIEW.md GAP-7.
- [x] Prototype `mx.value_and_grad` with self-mutating closure
  - [x] Verified: mx.value_and_grad works with self-mutating closures in _fit_kernel_params
  - [x] PASS: proceeded to Phase 1

## Phase 1: Gaussian Process Module - "The GPU Experimentation" — COMPLETE
# Target: `optuna/_gp/`  |  ~1308 LOC across 4 files
# Commit: 4fe5234 (migration) + PO review fix commit (MF-1, MF-2, SF-1)

### 1-PRE. dtype Audit (from QA finding M-7 — PO mandated) — DONE
- [x] All `mx.array()` calls include explicit `dtype=mx.float64`

### 1A. prior.py — DONE
- [x] `torch` imports replaced with `mx` (lazy import pattern)
- [x] `torch.log()` -> `mx.log()`, type hints updated

### 1B. gp.py — DONE (most complex migration)
- [x] Matern52: `@mx.custom_function` + manual VJP (handles sqrt(0) singularity)
- [x] MLL: `@mx.custom_function` + analytical VJP (Rasmussen & Williams eq. 5.9)
- [x] `_solve_triangular_right()` helper for missing MLX `left=False`
- [x] Batched diagonal clamping via eye broadcasting (replaces `mx.diag` for 3D+)
- [x] GPRegressor.__init__, kernel, _cache_matrix, posterior, _fit_kernel_params,
      append_running_data, fit_kernel_params — all ported

### 1C. acqf.py — DONE
- [x] `_erfcx()` with asymptotic expansion (threshold 3.0, 8-term Horner)
- [x] `_log_ndtr()` with tail-safe branch for x < -5 (SF-1 fix)
- [x] Sobol QMC via `scipy.stats.qmc.Sobol` (ADR-010)
- [x] All acqf classes ported: LogEI, LogPI, UCB, LCB, LogEHVI,
      ConstrainedLogEI, ConstrainedLogEHVI

### 1D. optim_mixed.py — DONE
- [x] `_gradient_ascent_batched()` uses `mx.value_and_grad` + `mx.stream(mx.cpu)`
- [x] batched_lbfgsb.py unchanged (ADR-008)

### 1E. Tests — DONE
- [x] `test_gp.py` (29 tests) and `test_acqf.py` (32 tests) migrated, all pass
- [x] Zero torch imports in `optuna/_gp/`, `optuna/samplers/_gp/`, GP tests

### PO Review Fixes — DONE
- [x] MF-1: GPSampler import check updated — "(Apple Silicon only)" message
- [x] MF-2: ADR-011 decided — Option B (hard fork, MLX-only) in DECISIONS.md
- [x] SF-1: `_log_ndtr` tail-safe branch (erfcx-based, accurate to 1e-5 at x=-5)

### QA Bug Fixes — DONE
- [x] B-1: `fit_kernel_params` catches `np.linalg.LinAlgError` (study crash fix)
- [x] B-2: `standard_logei` stable branch extended to z < -4.5, NaN gradient fix
- [x] B-3: `_erfcx` threshold tuned to x > 3.5 with branch clamping

### Remaining for QA (see PO_PHASE1_REVIEW.md Q-1 through Q-10)
- [x] Q-1: 270/270 parametrized GP tests pass
- [ ] Q-2: `study.optimize()` with GPSampler, 50 trials (B-1 fix unblocks this)
- [ ] Q-3: Verify GPSampler works without torch installed (needs clean venv)
- [x] Q-4: `erfcx` accuracy at transition — fixed (B-3), rel_err < 1e-5 at boundary
- [x] Q-5: `_log_ndtr` accuracy — tail 1e-10, x=-5 now 5.5e-7
- [x] Q-6: Greenlet + MLX interaction — PASS
- [ ] Q-7: Memory stability over 200 trials (B-5 open, needs profiling)
- [ ] Q-8: MLX benchmark vs torch baseline (S-1: 3.21x slower, strategic)
- [x] Q-9: Numerical parity: kernel, posterior, MLL — PASS
- [ ] Q-10: Gradient parity: MLL grad, acqf grad

## Phase 2: TPE Sampler - "The Parzen Estimator Paradigm"
# Target: `optuna/samplers/_tpe/`  |  ~912 LOC across 4 files
# "Bazinga! These mixture models are begging for parallel execution."

### 2-PRE. Deferred QA Fixes (from QA_REPORT.md, triaged in PO_QA_RESPONSE.md)
- [ ] Handle y=0.0 edge case in `_ndtri_exp` — return inf (H-2)
- [ ] Add `scale > 0` validation to `rvs()` (H-4)
- [ ] Add `a > b` validation to `_log_gauss_mass()` (M-1)
- [ ] Add `q in [0,1]` bounds check to `ppf()` (M-2)

### 2A. Error Function (`_erf.py`, 142 LOC)

- [ ] **Replace entirely with MLX built-in**
  - [ ] `erf(x)` (line 133) -> `mx.erf(x)` (MLX has this!)
  - [ ] Remove the 2000-element threshold and piecewise polynomial code
  - [ ] Create thin wrapper: `def erf(x): return mx.erf(mx.array(x))` if needed
  - [ ] Verify accuracy against `math.erf` for edge cases
  - [ ] "143 lines of polynomial wizardry, replaced by one function call.
         That's what Sheldon calls 'elegance'."

### 2B. Truncated Normal (`_truncnorm.py`, 296 LOC)

- [ ] **_ndtr() / _ndtr_single()** (line 59-74)
  - [ ] Replace `_ndtr(a)` with `0.5 + 0.5 * mx.erf(a / sqrt(2))`
  - [ ] Remove `functools.lru_cache` on `_ndtr_single` (MLX vectorizes)

- [ ] **_log_ndtr() / _log_ndtr_single()** (line 77-105)
  - [ ] Replace `np.frompyfunc(_log_ndtr_single, 1, 1)` (Python-level loop!)
        with vectorized MLX implementation
  - [ ] Port the tail approximation (line 84-101) to MLX array ops
  - [ ] This is a significant speedup opportunity - current impl is very slow

- [ ] **_log_gauss_mass()** (line 112-148)
  - [ ] Port case_left/case_right/case_central to MLX with `mx.where()`
  - [ ] `np.logaddexp` -> `mx.logsumexp` on stacked values
  - [ ] `np.log1p(...)` -> `mx.log1p(...)`

- [ ] **_ndtri_exp()** (line 151-220)
  - [ ] Port Newton iteration to MLX
  - [ ] Challenge: variable convergence per element. Use fixed max iterations
        with masked update: `x = mx.where(converged, x, x - dx)`
  - [ ] `np.expm1` -> check MLX availability or `mx.exp(x) - 1`

- [ ] **ppf()** (line 223-265)
  - [ ] Port case_left/case_right with `mx.where()`
  - [ ] Calls `_log_gauss_mass`, `_ndtri_exp` - must be ported first

- [ ] **rvs()** (line 268-282)
  - [ ] `random_state.uniform()` -> `mx.random.uniform()`
  - [ ] Calls `ppf()` - must be ported first

- [ ] **logpdf()** (line 285-296)
  - [ ] `_norm_logpdf(x)` -> `-(x**2) / 2.0 - log(sqrt(2*pi))`
  - [ ] `np.select(...)` -> `mx.where()` chain

### 2C. Probability Distributions (`probability_distributions.py`, 223 LOC)

- [ ] **_MixtureOfProductDistribution.sample()** (line 86-152)
  - [ ] `rng.choice(len(weights), p=weights, size=batch_size)`
        -> custom MLX weighted sampling or numpy fallback for index selection
  - [ ] `np.cumsum(active_weights, axis=-1)` -> `mx.cumsum(..., axis=-1)`
  - [ ] Port truncnorm calls to MLX versions from 2B

- [ ] **_MixtureOfProductDistribution.log_pdf()** (line 154-223)
  - [ ] THE HOT PATH - most important TPE acceleration target
  - [ ] `np.take_along_axis` -> check MLX equivalent
  - [ ] Weighted log-pdf accumulation over mixture components
  - [ ] `np.logaddexp` pattern: `log(sum(exp(weighted_log_pdf - max)))` + max
  - [ ] Port calls to `_truncnorm.logpdf`, `_log_gauss_mass`

- [ ] **_unique_inverse_2d()** (line 54-70)
  - [ ] Uses `np.argsort`, `np.cumsum` - port to MLX
  - [ ] `np.unique` equivalent not in MLX - consider numpy fallback or
        custom sort-based unique

- [ ] **_log_gauss_mass_unique()** (line 73-79)
  - [ ] Wrapper around `_truncnorm._log_gauss_mass` with deduplication

### 2D. Parzen Estimator (`parzen_estimator.py`, 251 LOC)

- [ ] **_ParzenEstimator.__init__()** (line 38-78)
  - [ ] Weight normalization: `weights /= weights.sum()` -> MLX
  - [ ] `_calculate_distributions()` calls per parameter

- [ ] **compute_sigmas()** (line 186-228)
  - [ ] `np.argsort`, `np.maximum`, `np.clip` -> MLX equivalents
  - [ ] Bandwidth computation: moderate NumPy, runs once per param

- [ ] **_calculate_categorical_distributions()** (line 132-166)
  - [ ] `np.unique(return_inverse=True)` -> numpy fallback
  - [ ] Weight computation with distance function
  - [ ] Row normalization

### 2E. Testing & Validation (TPE Module)

- [ ] **Numerical parity tests**
  - [ ] erf: MLX vs math.erf (rtol=1e-12)
  - [ ] _ndtr: MLX vs scipy.stats.norm.cdf (rtol=1e-10)
  - [ ] _log_ndtr: MLX vs scipy.stats.norm.logcdf (rtol=1e-8)
  - [ ] _log_gauss_mass: MLX vs original numpy (rtol=1e-10)
  - [ ] ppf: MLX vs scipy.stats.truncnorm.ppf (rtol=1e-8)
  - [ ] rvs: Statistical tests (KS test against known distribution)
  - [ ] log_pdf: MLX vs original numpy (rtol=1e-8)
  - [ ] Full TPESampler study.optimize() comparison

- [ ] **Performance benchmarks (TPE)**
  - [ ] log_pdf with 256 samples, 100 mixture components
  - [ ] log_pdf with 1024 samples, 500 mixture components
  - [ ] rvs sampling (1000 samples from 100-component mixture)
  - [ ] End-to-end TPESampler study.optimize() comparison

- [ ] **PO Review checkpoint**
- [ ] **QA Sign-off checkpoint**

## Phase 3: Hypervolume - "The Volume Conjecture"
# CONDITIONAL — proceed only if Phase 1-2 speedups justify it (PO decision per QA I-5)
# QA found: 3D hypervolume with 200 points < 1ms. GPU overhead may negate benefit.
# Target: `optuna/_hypervolume/`

- [ ] **_compute_2d()** (wfg.py:8-13)
  - [ ] Simple dot product - minimal GPU benefit, port for completeness
  - [ ] `edge_length_x @ edge_length_y` -> `mx.inner(x, y)` or `x @ y`

- [ ] **_compute_3d()** (wfg.py:16-38)
  - [ ] `np.argsort` -> `mx.argsort`
  - [ ] `np.maximum.accumulate` on 2D array -> custom MLX scan or loop
        (MLX may not have accumulate; check `mx.cummax` or similar)
  - [ ] `np.dot(np.dot(z_delta, y_delta), x_delta)` -> direct `@` ops

- [ ] **_compute_hv()** (wfg.py:41-64) - Recursive, keep on CPU
  - [ ] Inner ops: `(ref - sols).prod(axis=-1)` -> MLX if array is large
  - [ ] `np.maximum(sols[:, None], sols)` -> `mx.maximum(...)` for broadcasting

- [ ] **compute_hypervolume()** (wfg.py:110-181) - Entry point, dispatch logic
  - [ ] Keep dispatch logic, port inner implementations

- [ ] **Numerical parity tests**
- [ ] **Performance benchmarks** (vary n_points: 10, 50, 200, 1000)
- [ ] **PO Review checkpoint**
- [ ] **QA Sign-off checkpoint**

## Phase 4: Multi-Objective - "The Pareto Frontstellation"
# CONDITIONAL — proceed only if justified (PO decision per QA I-6)
# QA found: Pareto front for 1000x3 took 1ms. Low ROI unless 10K+ trials.
# Target: `optuna/study/_multi_objective.py`
# Deferred QA fix: NaN handling in Pareto front computation (H-3)

- [ ] **_is_pareto_front_nd()** (line 127-148)
  - [ ] While loop: sequential (keep on CPU)
  - [ ] Inner comparison: `np.any(loss[remaining] < loss[top], axis=1)` -> MLX
        for large `remaining` arrays (1000+ trials)

- [ ] **_is_pareto_front_2d()** (line 151-156)
  - [ ] `np.minimum.accumulate` -> MLX scan if available, else numpy

- [ ] **_calculate_nondomination_rank()** (line 187-219)
  - [ ] Iterative Pareto peeling - sequential outer, parallel inner
  - [ ] `np.unique(return_inverse=True)` -> numpy (no MLX equivalent)

- [ ] **Numerical parity tests**
- [ ] **Performance benchmarks** (vary n_trials: 100, 500, 2000)
- [ ] **PO Review checkpoint**
- [ ] **QA Sign-off checkpoint**

## Phase 5: NSGAII - "The Crossover Episode"
# CONDITIONAL — same as Phases 3-4 (PO decision per QA I-6)
# Target: `optuna/samplers/nsgaii/`
# Lowest priority - small operations, minimal GPU benefit at typical scales.

- [ ] **Crossover operations** (_crossovers/ directory)
  - [ ] SBX, BLX-alpha, SPX, UNDX, VSBX - port if population > 1000
  - [ ] Otherwise, numpy overhead is likely < GPU transfer cost

- [ ] **Elite selection** (_elite_population_selection_strategy.py)
  - [ ] Delegates to `_fast_non_domination_rank` (Phase 4)

- [ ] **Numerical parity tests**
- [ ] **Performance benchmarks**
- [ ] **PO Review checkpoint**
- [ ] **QA Sign-off checkpoint**

## Phase 6: Optimization - "The Performance Maximization"
# "First make it work, then make it fast." - Kent Beck
# "First make it fast, then make it faster." - Sheldon Cooper

- [ ] Profile MLX operations and identify bottlenecks
  - [ ] `mx.eval()` placement optimization (lazy eval sweet spots)
  - [ ] Identify unnecessary CPU<->GPU round-trips
- [ ] Optimize memory transfer patterns
  - [ ] Minimize `np.array(mx_array)` conversions
  - [ ] Batch MLX evaluations before converting to numpy
- [ ] Implement lazy evaluation strategies
  - [ ] Delay `mx.eval()` until results are actually needed
  - [ ] Chain MLX operations without intermediate evaluations
- [ ] Batch operations where possible
  - [ ] Batch acquisition function evaluations across candidates
  - [ ] Batch kernel computations across trial pairs
- [ ] Explore MLX-specific optimizations
  - [ ] `mx.compile()` for hot functions
  - [ ] Memory pre-allocation for known shapes
  - [ ] Stream management for overlapping compute/transfer
- [ ] Final comprehensive benchmark suite
- [ ] Documentation and user guide

## Ongoing
- [ ] Keep NOTES.md updated with findings
- [ ] Keep DECISIONS.md updated with new ADRs
- [ ] Update README_MLX.md with progress
- [ ] Write tests for every new MLX function
- [ ] "Remember: Penny knock knock knock, code review review review" - Sheldon
