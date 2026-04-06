# ============================================================================
# OPTUNA-MLX PROJECT NOTES
# "In a world where hyperparameters need optimizing, one framework
#  dared to use the GPU." - Sheldon Cooper, probably
# ============================================================================
# Created: 2026-04-05
# Last Updated: 2026-04-05
# Project: Optuna-MLX (Apple Silicon GPU Acceleration)
# Team: Ramos (Lead), Claude (Dev), PO, QA
# ============================================================================

## Project Overview

Optuna-MLX brings Apple Silicon GPU acceleration to the Optuna hyperparameter
optimization framework using Apple's MLX library. As Sheldon would say,
"It's not rocket science... it's GPU science, which is obviously superior."

## Codebase Analysis (2026-04-05)

### Architecture Summary
- Optuna is a mature hyperparameter optimization framework (~37,726 LOC)
- Core computation happens in: samplers, GP module, hypervolume, multi-objective
- Currently uses NumPy/SciPy for numerics with optional PyTorch for GP module
- Well-structured with clear separation of concerns (perfect for targeted acceleration)

### Key Computation Hotspots (Ranked by GPU Acceleration Potential)

#### 1. Gaussian Process Module (`optuna/_gp/`) - "The Low-Hanging Fruit, Leonard"

**Files & Key Functions:**

- `gp.py` (409 LOC) - Core GP regressor
  - `Matern52Kernel` (line 63): Custom `torch.autograd.Function` - computes
    `exp(-sqrt(5*d)) * (1/3*sqrt(5*d)^2 + sqrt(5*d) + 1)` with manual backward.
    MLX equivalent: `mlx.core.grad()` + pure function (no autograd.Function needed).
  - `GPRegressor.kernel()` (line 185): Builds O(n^2) squared distance matrix via
    `(X1.unsqueeze(-2) - X2.unsqueeze(-3)).square_()` then `sqd.matmul(inverse_squared_lengthscales)`.
    This is THE hottest path in the GP module. torch -> mlx: direct tensor op swap.
  - `GPRegressor._cache_matrix()` (line 124): Cholesky decomposition via `np.linalg.cholesky()`
    then `scipy.linalg.solve_triangular()` twice. MLX has `mx.linalg.cholesky()` and
    `mx.linalg.solve_triangular()` - near 1:1 replacement.
  - `GPRegressor.posterior()` (line 215): Computes mean/variance with `torch.linalg.vecdot`,
    `torch.linalg.solve_triangular`. Direct MLX swap possible.
  - `GPRegressor.marginal_log_likelihood()` (line 252): Cholesky + log determinant + quadratic
    form. Used in gradient-based hyperparameter fitting. Must preserve differentiability.
  - `GPRegressor._fit_kernel_params()` (line 287): Wraps torch tensors in scipy L-BFGS-B.
    Uses `loss.backward()` for gradient computation. MLX: use `mlx.core.grad()` to replace
    `torch.autograd` and `mlx.optimizers` or scipy for the L-BFGS-B loop.
  - `fit_kernel_params()` (line 354): Entry point. Creates GPRegressor with `torch.from_numpy()`.
    MLX: use `mx.array()` instead.

- `acqf.py` (402 LOC) - Acquisition functions
  - `standard_logei()` (line 65): Numerically stable log expected improvement using
    `torch.special.erfc`, `torch.special.erfcx`, `torch.erfinv`. MLX has `mx.erfinv()`
    but may need custom erfc/erfcx implementations or scipy fallback.
  - `logei()` (line 90): `standard_logei((mean - f0) / sqrt(var)) + log(sqrt(var))`.
  - `logehvi()` (line 45): Expected hypervolume improvement - `diff.log().sum()` with
    `torch.special.logsumexp()`. MLX has `mx.logsumexp()` - direct swap.
  - `LogEI.eval_acqf()` (line 153): Calls `gpr.posterior(x)` then `logei()`.
  - `UCB.eval_acqf()` (line 217): Simple `mean + sqrt(beta * var)`.
  - `LCB.eval_acqf()` (line 233): Simple `mean - sqrt(beta * var)`.
  - `BaseAcquisitionFunc.eval_acqf_with_grad()` (line 108): Uses `val.backward()` +
    `x_tensor.grad`. MLX: replace with `mlx.core.value_and_grad()`.
  - `_sample_from_normal_sobol()` (line 33): Uses `torch.quasirandom.SobolEngine` +
    `torch.erfinv`. MLX has no Sobol engine - need custom implementation or numpy fallback.

- `optim_mixed.py` (329 LOC) - Mixed-variable optimization
  - `_gradient_ascent_batched()` (line 29): Preconditioning + batched L-BFGS-B.
    Uses `acqf.eval_acqf(x_tensor)` then `.sum().backward()`. The preconditioning
    trick (scaling by lengthscales) is important for convergence.
  - `local_search_mixed_batched()` (line 232): Alternates continuous gradient ascent
    and discrete exhaustive/line search. Max 100 iterations with convergence check.
  - `optimize_acqf_mixed()` (line 280): Top-level entry. Samples 2048 preliminary
    candidates, picks top-10 via roulette selection, runs local search on each.

- `batched_lbfgsb.py` (168 LOC) - Batched L-BFGS-B with greenlet
  - Uses `greenlet` for cooperative multitasking of multiple scipy L-BFGS-B instances.
  - MLX opportunity: replace scipy L-BFGS-B with MLX-native optimizer, but scipy's
    L-BFGS-B is very mature. Consider keeping scipy but feeding it MLX gradients
    converted to numpy. Risk: frequent mx.array <-> np.array conversions.

- `scipy_blas_thread_patch.py` (49 LOC) - Thread workaround
  - Limits OpenBLAS threads to 1 for scipy >= 1.15.0.
  - MLX: This becomes unnecessary since MLX uses its own Metal compute pipeline.

**Torch -> MLX Translation Map (GP Module):**
```
torch.Tensor              -> mx.array
torch.from_numpy(x)       -> mx.array(x)
tensor.numpy()            -> np.array(tensor)
tensor.detach()           -> mx.stop_gradient(tensor) or just use the value
tensor.requires_grad_(T)  -> (not needed, use mx.grad() functional API)
torch.autograd.Function   -> (not needed, mx.grad() handles custom grad)
loss.backward()           -> mx.grad(loss_fn)(params)  # functional style
torch.linalg.cholesky     -> mx.linalg.cholesky
torch.linalg.solve_tri... -> mx.linalg.solve_triangular (check if available)
torch.linalg.vecdot       -> mx.sum(a * b, axis=-1)  # or manual dot
torch.eye                 -> mx.eye
torch.exp / log / sqrt    -> mx.exp / mx.log / mx.sqrt
tensor.unsqueeze(-2)      -> mx.expand_dims(arr, axis=-2)
tensor.square_()          -> mx.square(arr)  # MLX is functional, no in-place
tensor.matmul(other)      -> arr @ other  or  mx.matmul(arr, other)
tensor.clamp_min_(0)      -> mx.maximum(arr, 0)
torch.special.erfc        -> (custom or scipy fallback)
torch.special.erfcx       -> (custom or scipy fallback)
torch.erfinv              -> mx.erfinv (available in MLX)
torch.special.logsumexp   -> mx.logsumexp
torch.special.log_ndtr    -> (custom implementation needed)
torch.quasirandom.Sobol   -> (numpy/scipy fallback needed)
tensor.item()             -> arr.item()
tensor.cpu()              -> (no-op, MLX uses unified memory)
```

**Critical Risk Areas (GP):**
- In-place ops: PyTorch uses `.square_()`, `.clamp_min_()`. MLX is purely functional.
  Every in-place op must become `x = mx.square(x)`.
- `torch.autograd.Function` for Matern52Kernel: MLX has no equivalent class-based
  custom autograd. Must restructure as a pure function and use `mx.grad()`.
- `scipy.optimize.minimize` with torch gradients: Need to bridge MLX grad -> numpy
  for scipy. `mx.eval()` forces computation before conversion.
- Numerical precision: GP is sensitive to floating point. Must verify float64 support
  in MLX (MLX defaults to float32; need explicit dtype=mx.float64).

#### 2. TPE Sampler (`optuna/samplers/_tpe/`) - "The Parzen Estimator Paradigm"

**Files & Key Functions:**

- `parzen_estimator.py` (251 LOC) - Parzen estimator builder
  - `_ParzenEstimator.__init__()` (line 38): Builds mixture distribution from observations.
    Weight computation, sigma computation, mu/sigma arrays. All NumPy.
  - `_ParzenEstimator.sample()` (line 80): Delegates to `_MixtureOfProductDistribution.sample()`.
  - `_ParzenEstimator.log_pdf()` (line 84): Delegates to `_MixtureOfProductDistribution.log_pdf()`.
  - `compute_sigmas()` (line 186): Adaptive bandwidth computation. Uses `np.argsort`,
    `np.maximum`, `np.clip`. Moderate computation, runs once per parameter.
  - `_calculate_categorical_distributions()` (line 132): Categorical distance computations.
    Uses `np.unique`, `np.exp`, custom distance functions.
  - `_calculate_numerical_distributions()` (line 168): Sigma computation, creates batched
    truncated normal distributions.

- `probability_distributions.py` (223 LOC) - Distribution implementations
  - `_MixtureOfProductDistribution.sample()` (line 86): Core sampling loop.
    `rng.choice` for mixture component selection, then samples from each distribution type.
    Uses `_truncnorm.rvs()` for numerical params. Moderate parallelism opportunity.
  - `_MixtureOfProductDistribution.log_pdf()` (line 154): THE HOTTEST TPE PATH.
    Computes weighted log-pdf across all mixture components for all samples.
    Uses `_truncnorm.logpdf()`, `_log_gauss_mass()`, `np.logaddexp` pattern.
    Matrix shape: (n_samples, n_mixture_components) - large for many trials.
  - `_unique_inverse_2d()` (line 54): Deduplication for efficiency. Uses `np.argsort`,
    `np.cumsum`. Clever optimization to avoid redundant log_gauss_mass evaluations.
  - `_log_gauss_mass_unique()` (line 73): Wraps `_truncnorm._log_gauss_mass` with
    deduplication.

- `_truncnorm.py` (296 LOC) - Truncated normal (ported from SciPy)
  - `_log_gauss_mass()` (line 112): Log Gaussian probability mass. Handles left/right/central
    cases separately for numerical stability. Uses `_log_ndtr()`, `_ndtr()`, `np.log1p`.
  - `_ndtri_exp()` (line 151): Inverse of log_ndtr via Newton's method. Iterates up to
    100 times with convergence check. This is a potential bottleneck on GPU since
    iteration counts vary per element (divergent control flow).
  - `ppf()` (line 223): Percent point function (inverse CDF). Calls `_log_gauss_mass`
    and `_ndtri_exp`. Critical for sampling.
  - `rvs()` (line 268): Random variate sampling. `ppf(uniform_samples, a, b) * scale + loc`.
  - `logpdf()` (line 285): `_norm_logpdf(x) - _log_gauss_mass(a, b) - log(scale)`.
  - `_ndtr()` (line 72): Normal CDF using custom `erf()` implementation.
  - `_log_ndtr()` (line 104): Uses `np.frompyfunc(_log_ndtr_single, 1, 1)` - Python-level
    loop via frompyfunc. This is inherently slow and a prime GPU target.

- `_erf.py` (142 LOC) - Error function (ported from FreeBSD)
  - `erf()` (line 133): For < 2000 elements, uses `math.erf` per-element (Python loop).
    For >= 2000, uses vectorized polynomial approximation in 5 bins.
  - `_erf_right_non_big()` (line 112): Piecewise polynomial approximation using
    `numpy.polynomial.Polynomial`. Bins: tiny, small1, small2, med1, med2.
  - MLX note: MLX has `mx.erf()` built-in! This entire file becomes unnecessary.
    "Bazinga! That's 143 lines of code we never have to look at again." - Sheldon

**NumPy -> MLX Translation Map (TPE Module):**
```
np.array / np.ndarray     -> mx.array
np.argsort                -> mx.argsort
np.exp / log / log1p      -> mx.exp / mx.log / mx.log1p
np.maximum / minimum      -> mx.maximum / mx.minimum
np.clip                   -> mx.clip
np.sum / mean / cumsum    -> mx.sum / mx.mean / mx.cumsum
np.logaddexp              -> (custom: mx.logsumexp on stacked arrays)
np.unique                 -> (no MLX equivalent - numpy fallback needed)
np.frompyfunc             -> (no MLX equivalent - vectorize with mx ops)
np.random.RandomState     -> mx.random.key + mx.random.* functions
rng.choice                -> (custom weighted sampling in MLX)
rng.uniform               -> mx.random.uniform
rng.rand                  -> mx.random.uniform(shape=...)
```

**Critical Risk Areas (TPE):**
- `_log_ndtr_single`: Uses `functools.lru_cache` with Python float -> pure Python loop.
  Must rewrite as vectorized MLX operation.
- `_ndtri_exp`: Newton iteration with variable convergence. GPUs prefer uniform
  iteration counts. May need fixed iteration count (100) with masked updates.
- `np.unique` with `return_inverse=True`: Used extensively in deduplication. MLX has
  no unique function - must fallback to numpy or redesign.
- Random state: TPE uses `np.random.RandomState` for reproducibility. MLX uses
  `mx.random.key` (JAX-style). Must ensure reproducible results during parity testing.

#### 3. Hypervolume Calculations (`optuna/_hypervolume/`) - "The Volume Conjecture"

**Files & Key Functions:**

- `wfg.py` (182 LOC) - WFG hypervolume algorithm
  - `_compute_2d()` (line 8): `edge_length_x @ edge_length_y` - single dot product.
    Fast already, minimal GPU benefit unless called at scale.
  - `_compute_3d()` (line 16): O(n^2) with `np.maximum.accumulate` on 2D array,
    then `np.dot(np.dot(z_delta, y_delta), x_delta)`. Good GPU target for large n.
  - `_compute_hv()` (line 41): Recursive WFG. Special-cased for 1-2 points (no numpy).
    For 3+: `(reference_point - sorted_loss_vals).prod(axis=-1)` for inclusive HVs,
    then `np.maximum(sorted_loss_vals[:, np.newaxis], sorted_loss_vals)` for limited sols.
    Recursive calls to `_compute_exclusive_hv` - hard to parallelize on GPU due to recursion.
  - `_compute_exclusive_hv()` (line 67): `inclusive_hv - _compute_hv(limited_sols[on_front])`.
    Calls `_is_pareto_front` internally (links to multi-objective module).
  - `compute_hypervolume()` (line 110): Entry point. Dispatches to 2D/3D/ND implementations.

**GPU Consideration:** The recursive nature of WFG for >3D makes GPU parallelization
challenging. Best approach: accelerate the inner operations (dot products, accumulate,
Pareto front checks) while keeping the recursive structure on CPU. For 2D and 3D,
the algorithms are iterative and well-suited for GPU.

#### 4. Multi-Objective Pareto Front (`optuna/study/_multi_objective.py`) - "The Pareto Frontstellation"

**Files & Key Functions:**

- `_multi_objective.py` (262 LOC) - Pareto front and non-domination ranking
  - `_is_pareto_front_nd()` (line 127): Core O(n^2) algorithm. While loop removes
    dominated solutions iteratively. Uses `np.any(loss_values[remaining] < loss_values[top], axis=1)`.
    GPU potential: the comparison `loss_values[remaining] < loss_values[top]` is parallelizable
    but the while loop with shrinking `remaining_indices` is inherently sequential.
  - `_is_pareto_front_2d()` (line 151): O(n) with `np.minimum.accumulate`. Already fast.
  - `_is_pareto_front()` (line 171): Entry point. Uses `np.unique` with lexsort.
  - `_calculate_nondomination_rank()` (line 187): Iteratively peels Pareto fronts.
    While loop calling `_is_pareto_front` repeatedly. O(n^2 * k) where k = #ranks.
  - `_fast_non_domination_rank()` (line 49): Handles feasible/infeasible/NaN penalty.
    Three phases of `_calculate_nondomination_rank`.

**GPU Consideration:** The iterative peeling pattern (remove Pareto front, repeat) is
hard to parallelize. Best approach: accelerate the inner `_is_pareto_front_nd` comparison
kernel while keeping the outer loop on CPU. For large n (1000+), the O(n^2) comparison
dominates and benefits from GPU parallelism.

#### 5. NSGAII Sampler (`optuna/samplers/nsgaii/`) - "The Crossover Episode"

**Files & Key Functions:**

- `_crossover.py` (180 LOC) - Crossover coordination
  - `perform_crossover()` (line 84): Selects parents, applies crossover, validates bounds.
  - `_try_crossover()` (line 30): Transforms params, calls `crossover.crossover()`.
  - `_inlined_categorical_uniform_crossover()` (line 167): Simple mask-based swap.

- `_crossovers/` directory contains: SBX, BLX-alpha, SPX, UNDX, VSBX, Uniform
  - These are relatively small operations (per-generation, per-individual).
  - GPU benefit is modest unless population sizes are very large (1000+).

**GPU Consideration:** NSGAII is lowest priority. The crossover operations are small
and the overhead of GPU memory transfer would likely negate benefits for typical
population sizes (50-200). Worth revisiting only if users need very large populations.

---

### Existing Dependencies & Integration Points
- **NumPy** (required): Core dependency. Used everywhere. MLX replaces hot paths only.
- **SciPy** (optional): GP kernel fitting (L-BFGS-B), triangular solves, Brent optimization.
  Keep scipy for L-BFGS-B initially; feed it MLX-computed gradients via numpy conversion.
- **PyTorch** (optional): GP module only. Full replacement target for MLX.
  Key torch features used: autograd, linalg, special functions, quasirandom Sobol.
- **scikit-learn** (optional): Importance evaluation (random forests). Out of scope.
- **cmaes** (optional): External CMA-ES package. Out of scope.
- **greenlet** (optional): Cooperative multitasking for batched L-BFGS-B. Keep as-is
  since it's a CPU-side optimization for scipy interleaving.

### MLX API Capabilities & Gaps

**Available in MLX (confirmed):**
- Array operations: create, reshape, slice, broadcast, concatenate
- Math: exp, log, log1p, sqrt, square, abs, sign, clip, maximum, minimum
- Linear algebra: matmul, cholesky, solve, solve_triangular, inv, norm
- Reductions: sum, mean, max, min, argmax, argmin, cumsum, prod
- Special: erf, erfinv, logsumexp
- Random: uniform, normal, key-based PRNG
- Automatic differentiation: grad, value_and_grad (functional API)
- Data types: float16, float32, bfloat16, float64 (check availability)

**NOT available in MLX (need workarounds):**
- `np.unique` with return_inverse - need numpy fallback
- `torch.quasirandom.SobolEngine` - need numpy/scipy fallback
- `torch.special.erfc` / `erfcx` - may need custom implementation
- `torch.special.log_ndtr` - custom implementation needed
- `functools.lru_cache` patterns - redesign as vectorized ops
- `np.frompyfunc` - redesign as vectorized ops
- In-place operations (`.square_()`, `.clamp_min_()`) - use functional style

**MLX float64 Note:** MLX supports float64 but may not accelerate it on all GPU
hardware. Need to verify on target Apple Silicon. If float64 is slow on GPU,
consider: (a) computing on float32 where precision allows, (b) using float64
only for numerically sensitive operations (Cholesky, log-det), (c) mixed precision.
"As Sheldon would say, precision matters. Unlike Howard's fashion sense."

---

### Architecture Design Notes

**Backend Selection Strategy:**
The cleanest approach is a backend abstraction layer in `optuna/_mlx/`:
```python
# optuna/_mlx/__init__.py
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

def get_array_module():
    """Return mlx.core if available, else numpy."""
    return mx if HAS_MLX else np
```

However, for Phase 1 (GP module), since it already has a PyTorch abstraction
(lazy import, TYPE_CHECKING pattern), we can follow the same pattern:
```python
if TYPE_CHECKING:
    import mlx.core as mx
else:
    from optuna._imports import _LazyImport
    mx = _LazyImport("mlx.core")
```

**Grad API Difference (Critical):**
- PyTorch: Object-oriented. `tensor.requires_grad_(True)`, `loss.backward()`, `tensor.grad`.
- MLX: Functional. `grad_fn = mx.grad(loss_fn)`, `grads = grad_fn(params)`.
- This means `_fit_kernel_params` must be restructured:
  - Extract loss computation into a pure function
  - Use `mx.value_and_grad(loss_fn)(params)` to get both value and gradient
  - Feed gradient to scipy L-BFGS-B (after converting to numpy)

## Session Log

### 2026-04-05 - Project Kickoff
- Completed full codebase analysis
- Identified 5 major acceleration targets
- Set up dev branch for development
- Created project documentation structure
- "And so it begins... like a new season of our favorite show."

### 2026-04-05 - Deep Code Analysis
- Read all source files for GP module (gp.py, acqf.py, optim_mixed.py, batched_lbfgsb.py)
- Read all source files for TPE module (parzen_estimator.py, probability_distributions.py,
  _truncnorm.py, _erf.py)
- Read hypervolume (wfg.py) and multi-objective (_multi_objective.py) modules
- Read NSGAII crossover module (_crossover.py)
- Mapped every function, line number, and torch/numpy operation to MLX equivalent
- Identified critical risk areas: float64, in-place ops, autograd pattern, missing APIs
- Documented torch -> MLX and numpy -> MLX translation maps
- "I've read more code today than Raj has spoken words to women." - Howard
