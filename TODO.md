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

## Phase 0: Project Setup (The Pilot Episode)
- [x] Analyze Optuna codebase and identify acceleration targets
- [x] Set up dev branch
- [x] Create project documentation (NOTES, DECISIONS, TODO, README)
- [x] Document architectural decisions (ADR-001 through ADR-010)
- [x] Deep code analysis: map every function/line to MLX equivalent
- [ ] Set up MLX development environment and verify GPU access
  - [ ] `pip install mlx` and verify `import mlx.core as mx`
  - [ ] Verify `mx.default_device()` reports GPU
  - [ ] Verify float64 support: `mx.array([1.0], dtype=mx.float64)`
  - [ ] Benchmark float64 vs float32 on Cholesky (100x100 matrix)
- [ ] Create MLX utility module (`optuna/_mlx/__init__.py`)
  - [ ] Backend detection: `HAS_MLX` flag
  - [ ] Lazy import pattern matching existing optuna style
  - [ ] `get_array_module()` helper function
- [ ] Add MLX to optional dependencies in pyproject.toml
  - [ ] Add `"mlx>=0.5.0"` to `[project.optional-dependencies]`
  - [ ] Create `mlx` extra: `mlx = ["mlx>=0.5.0"]`
- [ ] Set up benchmark suite for before/after comparison
  - [ ] GP kernel matrix computation (vary n_trials: 10, 50, 100, 500)
  - [ ] GP acquisition function evaluation (vary n_candidates: 256, 1024, 4096)
  - [ ] TPE log_pdf computation (vary n_samples * n_components)
  - [ ] Pareto front determination (vary n_trials: 100, 500, 1000)
  - [ ] Full study.optimize() end-to-end timing
- [ ] Document ADR-011: Backend fallback strategy (MLX -> torch -> error)
- [ ] Document ADR-012: Autograd pattern for _fit_kernel_params

## Phase 0.5: Autograd Risk Gate (BLOCKING - must pass before Phase 1)
# Added 2026-04-05 by PO. See PO_ROADMAP_REVIEW.md GAP-7.
- [ ] Prototype `mx.value_and_grad` with self-mutating closure
  - [ ] Create minimal SimpleGP class that mutates self.param in loss_fn
  - [ ] Verify mx.value_and_grad(loss_fn)(params) produces correct val + grad
  - [ ] If PASS: proceed to Phase 1
  - [ ] If FAIL: escalate to PO, implement flat-param pure-function alternative

## Phase 1: Gaussian Process Module - "The GPU Experimentation"
# Target: `optuna/_gp/`  |  ~1308 LOC across 4 files
# "Like replacing the warp drive - high impact, well-defined interface."

### 1-PRE. dtype Audit (from QA finding M-7 — PO mandated)
- [ ] List every `torch.from_numpy()` call in _gp/ (19 total per QA)
- [ ] Note which have explicit `dtype=torch.float64` vs implicit
- [ ] Ensure ALL `mx.array()` calls in migration include explicit `dtype=mx.float64`
- [ ] This is a PO sign-off gate for Phase 1

### 1A. Core GP Regressor (`gp.py`, 409 LOC)

- [ ] **Matern 5/2 Kernel** (gp.py:63-90)
  - [ ] Convert `Matern52Kernel(torch.autograd.Function)` to pure function
        `matern52_kernel(squared_distance) -> mx.array`
  - [ ] Forward: `mx.exp(-sqrt5d) * ((5/3) * sqd + sqrt5d + 1)`
  - [ ] Verify `mx.grad(matern52_kernel)` produces correct derivative
        (compare against manual: `(-5/6) * (sqrt5d + 1) * exp(-sqrt5d)`)
  - [ ] Handle `squared_distance=0` edge case (torch version has manual
        derivative to avoid zero division in autograd)

- [ ] **GPRegressor.__init__** (gp.py:93-118)
  - [ ] Replace `torch.Tensor` params with `mx.array`
  - [ ] `torch.from_numpy(is_categorical)` -> `mx.array(is_categorical)`
  - [ ] Squared diff: `(X.unsqueeze(-2) - X.unsqueeze(-3)).square_()`
        -> `mx.square(mx.expand_dims(X, -2) - mx.expand_dims(X, -3))`
  - [ ] Categorical handling: `.type(torch.float64)` -> `.astype(mx.float64)`

- [ ] **GPRegressor.kernel()** (gp.py:185-213)
  - [ ] Port squared distance computation with broadcasting
  - [ ] `sqd.matmul(inverse_squared_lengthscales)` -> `sqd @ isl`
  - [ ] Call new pure `matern52_kernel()` * `kernel_scale`

- [ ] **GPRegressor._cache_matrix()** (gp.py:124-149)
  - [ ] `self.kernel().detach().cpu().numpy()` -> `np.array(self.kernel())`
        (after `mx.eval()` to force computation)
  - [ ] Keep `np.linalg.cholesky` OR use `mx.linalg.cholesky`
  - [ ] Keep `scipy.linalg.solve_triangular` OR use `mx.linalg.solve_triangular`
  - [ ] Convert results back: `torch.from_numpy(...)` -> `mx.array(...)`
  - [ ] Detach gradients: `self.isl = mx.stop_gradient(self.isl)`

- [ ] **GPRegressor.posterior()** (gp.py:215-250)
  - [ ] `torch.linalg.vecdot(a, b)` -> `mx.sum(a * b, axis=-1)`
  - [ ] `torch.linalg.solve_triangular(L, rhs, upper, left)`
        -> check `mx.linalg.solve_triangular` API matches
  - [ ] `.clamp_min_(0.0)` -> `mx.maximum(var, 0.0)`
  - [ ] `.squeeze(0)` -> `mx.squeeze(arr, axis=0)`

- [ ] **GPRegressor.marginal_log_likelihood()** (gp.py:252-285)
  - [ ] `torch.eye(n, dtype=float64)` -> `mx.eye(n)` (check dtype)
  - [ ] `torch.linalg.cholesky(cov_Y_Y)` -> `mx.linalg.cholesky(cov_Y_Y)`
  - [ ] `L.diagonal().log().sum()` -> `mx.sum(mx.log(mx.diag(L)))`
  - [ ] `torch.linalg.solve_triangular(L, y[:,None], upper=False)`
        -> `mx.linalg.solve_triangular(L, y[:,None], upper=False)`
  - [ ] Must remain differentiable for `_fit_kernel_params()`!

- [ ] **GPRegressor._fit_kernel_params()** (gp.py:287-351)
  - [ ] Restructure `loss_func` as pure function for `mx.value_and_grad()`
  - [ ] `raw_params_tensor = torch.from_numpy(raw_params).requires_grad_(True)`
        -> just pass through, use `mx.grad()` externally
  - [ ] `loss.backward()` -> `loss, grad = mx.value_and_grad(loss_fn)(raw_params_mx)`
  - [ ] Bridge to scipy: `grad_np = np.array(grad); mx.eval(grad)`
  - [ ] Keep `scipy.optimize.minimize(method='l-bfgs-b')` (ADR-008)
  - [ ] `torch.exp(raw_params_tensor[:n_params])` -> `mx.exp(raw_params[:n_params])`

- [ ] **GPRegressor.append_running_data()** (gp.py:151-183)
  - [ ] Port matrix expansion for running trials
  - [ ] Keep scipy solve_triangular or port to MLX

- [ ] **fit_kernel_params()** (gp.py:354-409)
  - [ ] Entry point: `torch.from_numpy(X)` -> `mx.array(X)` throughout
  - [ ] `torch.ones(X.shape[1]+2, dtype=float64)` -> `mx.ones((X.shape[1]+2,))`
  - [ ] Error handling / fallback to default params: keep same logic

### 1B. Acquisition Functions (`acqf.py`, 402 LOC)

- [ ] **Special functions** (needed before porting acqf)
  - [ ] Implement `erfc_mlx(x)`: `1 - mx.erf(x)` with tail handling
  - [ ] Implement `erfcx_mlx(x)`: `mx.exp(mx.square(x)) * erfc_mlx(x)`
  - [ ] Implement `log_ndtr_mlx(x)`: `mx.log(0.5 * erfc_mlx(-x / sqrt(2)))`
  - [ ] Test against scipy.special.erfc/erfcx/log_ndtr for accuracy

- [ ] **standard_logei()** (acqf.py:65-87)
  - [ ] `torch.special.erfc(-_SQRT_HALF * z)` -> `erfc_mlx(-_SQRT_HALF * z)`
  - [ ] `.exp()` -> `mx.exp()`
  - [ ] `.log()` -> `mx.log()`
  - [ ] Small-z branch: `.erfcx()` -> `erfcx_mlx()`
  - [ ] `z[(small := z < -25)]` indexing: verify MLX boolean indexing works

- [ ] **logei(), logehvi()** (acqf.py:45-62, 90-92)
  - [ ] `torch.special.logsumexp` -> `mx.logsumexp`
  - [ ] `var.sqrt_()` -> `mx.sqrt(var)` (no in-place)
  - [ ] `sigma.log()` -> `mx.log(sigma)`
  - [ ] `.clamp_(min=..., max=...)` -> `mx.clip(arr, min, max)`

- [ ] **BaseAcquisitionFunc** (acqf.py:95-113)
  - [ ] `eval_acqf_no_grad()`: `torch.no_grad()` -> just call function (MLX
        doesn't track grads by default)
  - [ ] `eval_acqf_with_grad()`: `val.backward()` + `.grad`
        -> `val, grad = mx.value_and_grad(self.eval_acqf)(mx.array(x))`
        -> return `val.item(), np.array(grad)`

- [ ] **LogEI, LogPI, UCB, LCB, ConstrainedLogEI** (acqf.py:116-272)
  - [ ] Port each `eval_acqf()` method
  - [ ] `torch.zeros(shape, dtype=float64)` -> `mx.zeros(shape)`
  - [ ] `torch.sqrt(self._beta * var)` -> `mx.sqrt(self._beta * var)`
  - [ ] LogPI: `torch.special.log_ndtr(...)` -> `log_ndtr_mlx(...)`

- [ ] **LogEHVI** (acqf.py:275-350)
  - [ ] `_sample_from_normal_sobol()`: Keep `torch.quasirandom.SobolEngine`
        or use `scipy.stats.qmc.Sobol` + numpy as fallback (ADR-010)
  - [ ] `torch.erfinv(samples)` -> `mx.erfinv(samples)` (available in MLX!)
  - [ ] `torch.stack(Y_post, dim=-1)` -> `mx.stack(Y_post, axis=-1)`
  - [ ] Port `_get_non_dominated_box_bounds()` helper

- [ ] **ConstrainedLogEHVI** (acqf.py:353-402)
  - [ ] Composition of LogEHVI + LogPI list - follows from above

### 1C. Mixed Optimization (`optim_mixed.py`, 329 LOC)

- [ ] **_gradient_ascent_batched()** (optim_mixed.py:29-94)
  - [ ] Port `negative_acqf_with_grad()` inner function
  - [ ] `torch.from_numpy(next_params).requires_grad_(True)` ->
        use `mx.value_and_grad` on the batched acqf call
  - [ ] `neg_fvals.sum().backward()` -> functional grad approach
  - [ ] Keep `batched_lbfgsb.batched_lbfgsb()` call (ADR-008)
  - [ ] Feed numpy gradients to scipy: `np.array(mx_grad)`

- [ ] **_exhaustive_search()** (optim_mixed.py:97-118)
  - [ ] `acqf.eval_acqf_no_grad(all_params)` - already returns numpy
  - [ ] `np.argmax(fvals)` - keep as numpy (small arrays)

- [ ] **_discrete_line_search()** (optim_mixed.py:121-186)
  - [ ] Uses scipy.optimize.minimize_scalar (Brent) - keep on CPU
  - [ ] Inner `acqf.eval_acqf_no_grad` calls stay as-is

- [ ] **local_search_mixed_batched()** (optim_mixed.py:232-277)
  - [ ] Orchestration loop - keep structure, port inner computations

- [ ] **optimize_acqf_mixed()** (optim_mixed.py:280-329)
  - [ ] Top-level entry: keep numpy for sampling + roulette selection
  - [ ] `acqf.eval_acqf_no_grad(sampled_xs)` - this evaluates 2048 candidates
        at once. GPU parallelism shines here!

### 1D. Batched L-BFGS-B (`batched_lbfgsb.py`, 168 LOC)

- [ ] Keep mostly unchanged (ADR-008)
- [ ] Verify greenlet interleaving still works when acqf internally uses MLX
- [ ] The func_and_grad callback will compute grads via MLX, return numpy

### 1E. Prior Module (`prior.py`, 33 LOC) — ADDED by PO
# Was missing from original plan. See PO_ROADMAP_REVIEW.md GAP-1.

- [ ] Replace `import torch` with `import mlx.core as mx` (same lazy pattern)
- [ ] `torch.log()` -> `mx.log()` (lines 24, 30)
- [ ] `torch.Tensor` type hints -> `mx.array` (lines 19, 22)
- [ ] Must be done BEFORE gp.py (gp.py calls prior from _fit_kernel_params)

### 1F. Testing & Validation (GP Module)

- [ ] **Numerical parity tests**
  - [ ] Matern52 kernel: compare MLX vs torch output (rtol=1e-10)
  - [ ] Kernel matrix: same inputs -> same matrix (rtol=1e-10)
  - [ ] Cholesky: same input -> same L (rtol=1e-8, more lenient)
  - [ ] Posterior mean/var: same GPR -> same predictions (rtol=1e-8)
  - [ ] Marginal log likelihood: same GPR -> same value (rtol=1e-6)
  - [ ] fit_kernel_params: same X,Y -> similar params (rtol=1e-3, optimizer-dependent)
  - [ ] LogEI, UCB, LCB: same inputs -> same values (rtol=1e-8)
  - [ ] Full optimize_acqf_mixed: same study -> same suggested params

- [ ] **Gradient parity tests**
  - [ ] Compare `mx.grad(matern52)` vs torch manual derivative
  - [ ] Compare `mx.value_and_grad(logei)` vs `torch.backward()` result
  - [ ] Compare loss gradient in `_fit_kernel_params` (MLX vs torch)

- [ ] **Edge case tests**
  - [ ] Zero squared distance (Matern52 kernel singularity)
  - [ ] Near-singular kernel matrix (high noise required)
  - [ ] Infinite objective values (gp.py:47 `warn_and_convert_inf`)
  - [ ] Single trial, two trials, many trials (1000+)
  - [ ] All categorical params, all continuous, mixed

- [ ] **Performance benchmarks (GP)**
  - [ ] `GPRegressor.kernel()` — vary n_trials (10-1000), n_params (2-50)
  - [ ] `_cache_matrix()` — Cholesky timing
  - [ ] `posterior()` — vary batch size
  - [ ] `marginal_log_likelihood()` — full computation
  - [ ] `_fit_kernel_params()` — full optimization loop
  - [ ] `eval_acqf` on 2048 candidates (optimize_acqf_mixed hot path)
  - [ ] End-to-end GPSampler study.optimize() comparison

- [ ] **Additional tests (added by PO)**
  - [ ] Verify `optim_sample.py` still works (no torch, calls acqf)
  - [ ] Create dual-backend test fixture in `tests/gp_tests/conftest.py`
  - [ ] Verify greenlet + MLX lazy eval interaction (mx.eval before np.array)

- [ ] **PO Review checkpoint** - functionality matches original
- [ ] **QA Sign-off checkpoint** - accuracy + performance validated
  - [ ] See QA_ACCEPTANCE_CRITERIA.md AC-1.1 through AC-1.15

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
