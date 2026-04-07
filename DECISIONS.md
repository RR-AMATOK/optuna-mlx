# ============================================================================
# OPTUNA-MLX ARCHITECTURAL DECISIONS LOG
# "I made a decision tree once. It was a dark day." - Raj Koothrappali
# ============================================================================
# Created: 2026-04-05
# Last Updated: 2026-04-05
# Project: Optuna-MLX (Apple Silicon GPU Acceleration)
# ============================================================================

## Decision Record Format
Each decision follows: ID | Date | Decision | Rationale | Status | Alternatives Considered

---

## ADR-001: Use MLX as the GPU Backend (Not CUDA/Metal directly)
- **Date:** 2026-04-05
- **Decision:** Use Apple's MLX framework as the GPU acceleration layer
- **Rationale:** MLX is purpose-built for Apple Silicon, has a NumPy-like API
  making migration straightforward, supports lazy evaluation for memory
  efficiency, and has automatic differentiation. As Howard would say,
  "Why build a rocket when NASA already built one?"
- **Status:** ACCEPTED
- **Alternatives:** Raw Metal shaders (too low-level), MPS via PyTorch
  (extra dependency layer), JAX (not optimized for Apple Silicon)

## ADR-002: Harness Development Approach
- **Date:** 2026-04-05
- **Decision:** Follow a harness approach with PO and QA roles
- **Rationale:** The project modifies core computation paths in a mature
  framework. Structured review gates prevent regressions and ensure
  numerical accuracy is maintained. "Unlike Sheldon's roommate agreement,
  this process actually makes sense."
- **Status:** ACCEPTED
- **Alternatives:** Cowboy coding (rejected - too risky for numerical code)

## ADR-003: Functional Parity Before Optimization
- **Date:** 2026-04-05
- **Decision:** First achieve identical results with MLX backend, then optimize
- **Rationale:** Numerical accuracy must be verified before pursuing performance.
  A fast wrong answer is worse than a slow correct one. "As Sheldon says,
  'I'm not crazy, my mother had me tested.' We need to test our results too."
- **Status:** ACCEPTED
- **Alternatives:** Optimize during initial implementation (rejected - risk of
  masking bugs with performance changes)

## ADR-004: GP Module as First Migration Target
- **Date:** 2026-04-05
- **Decision:** Start with the Gaussian Process module (`optuna/_gp/`)
- **Rationale:** 
  1. Already uses PyTorch tensors - most natural MLX migration path
  2. Contains the heaviest matrix operations (O(n^2) kernel matrices)
  3. Has clear boundaries (kernel computation, Cholesky, acquisition functions)
  4. Existing torch code serves as a 1:1 translation template
  5. "It's like replacing the warp drive - high impact, well-defined interface."
- **Status:** ACCEPTED
- **Alternatives:** TPE first (more code to change, less torch precedent),
  Hypervolume first (smaller impact)

## ADR-005: Dev Branch Strategy
- **Date:** 2026-04-05
- **Decision:** All development on `dev` branch, push to fork (RR-AMATOK/optuna-mlx)
- **Rationale:** Keeps master clean as our stable reference point. Dev branch
  allows incremental work without polluting the main history.
  "Like Sheldon's spot on the couch - master is sacred."
- **Status:** ACCEPTED
- **Alternatives:** Feature branches per module (too fragmented for initial work)

## ADR-006: Acceleration Target Priority Order
- **Date:** 2026-04-05
- **Decision:** Implement MLX acceleration in this order:
  1. Gaussian Process module (torch replacement)
  2. TPE Sampler (numpy hot paths)
  3. Hypervolume calculations (matrix operations)
  4. Multi-objective Pareto front (parallel comparisons)
  5. NSGAII crossover operations (vectorized math)
- **Rationale:** Ordered by: existing torch code (easiest migration), computation
  intensity, and impact on overall optimization time. Each module has clear
  boundaries for isolated testing.
- **Status:** ACCEPTED
- **Alternatives:** Impact-only ordering (ignores migration difficulty)

## ADR-007: MLX Autograd Strategy - Functional over Object-Oriented
- **Date:** 2026-04-05
- **Decision:** Restructure torch autograd patterns into MLX's functional grad API
- **Rationale:** PyTorch uses OOP autograd: `tensor.requires_grad_(True)`,
  `loss.backward()`, `tensor.grad`. MLX uses functional: `mx.grad(fn)(params)`.
  This requires restructuring `GPRegressor._fit_kernel_params()` (gp.py:287)
  and `BaseAcquisitionFunc.eval_acqf_with_grad()` (acqf.py:108).
  The `Matern52Kernel(torch.autograd.Function)` class (gp.py:63) with custom
  forward/backward becomes a pure function - MLX's `mx.grad()` handles the
  derivative automatically since `mx.sqrt`, `mx.exp` are differentiable.
  "Sheldon doesn't need a class to be functional. Neither does MLX."
- **Status:** ACCEPTED
- **Key Changes Required:**
  1. `Matern52Kernel` class -> pure `matern52_kernel(squared_distance)` function
  2. `_fit_kernel_params` loss computation -> extract pure `loss_fn(raw_params)`
  3. `eval_acqf_with_grad` -> use `mx.value_and_grad(eval_acqf)(x)`
  4. Bridge: `value, grad = mx.value_and_grad(fn)(x); np_grad = np.array(grad)`
- **Alternatives:** Wrapping MLX in a torch-like autograd API (over-engineering)

## ADR-008: Keep scipy L-BFGS-B, Feed MLX Gradients
- **Date:** 2026-04-05
- **Decision:** Keep scipy's `fmin_l_bfgs_b` optimizer but feed it MLX-computed
  gradients converted to numpy arrays
- **Rationale:** scipy's L-BFGS-B is extremely mature and battle-tested.
  The optimizer itself is not the bottleneck - the gradient computation is.
  By computing gradients on GPU (MLX) and passing them to scipy (CPU), we
  get the best of both worlds. The conversion cost (`np.array(mx_grad)`) is
  minimal compared to the gradient computation itself.
  Applies to: `gp.py:_fit_kernel_params()` and `optim_mixed.py:_gradient_ascent_batched()`.
  The greenlet batching in `batched_lbfgsb.py` remains useful for interleaving
  multiple scipy optimizer instances.
  "Like ordering Thai food while doing physics - outsource what you can." - Raj
- **Status:** ACCEPTED
- **Alternatives:** 
  - MLX-native L-BFGS-B (doesn't exist yet)
  - Custom gradient descent in MLX (less robust, no line search)
  - Adam optimizer in MLX (wrong tool for bounded optimization)

## ADR-009: MLX float64 Policy
- **Date:** 2026-04-05
- **Decision:** Use float64 for all GP operations, verify GPU acceleration.
  If float64 is not GPU-accelerated, use a hybrid approach.
- **Rationale:** The GP module (especially Cholesky decomposition and log
  marginal likelihood) is highly sensitive to floating point precision.
  The existing code explicitly uses `torch.float64` everywhere (gp.py:112,
  gp.py:364, acqf.py:39). Downgrading to float32 risks numerical instability
  in: Cholesky (gp.py:132,281), triangular solves (gp.py:136-138), and
  log determinant (gp.py:282).
  MLX supports float64 but Metal GPU may not accelerate it on all chips.
  If benchmarks show float64 is slow on GPU, consider:
  (a) float32 for kernel matrix, float64 for Cholesky/solve
  (b) float32 with iterative refinement for Cholesky
  (c) CPU fallback for precision-critical operations
  "Precision is like Sheldon's spot - non-negotiable."
- **Status:** ACCEPTED — REFINED after Work Item 0A benchmark (2026-04-05).
  **Benchmark result: float64 is NOT supported on GPU in MLX 0.31.1. Cholesky
  is CPU-only even for float32.** Revised strategy:
  - GPU float32 for kernel matrix element-wise ops (squared distances, Matern52)
  - CPU float64 for Cholesky, solve_triangular, log-det (precision-critical)
  - Use `stream=mx.cpu` for linalg ops, default GPU stream for array math
  - Compute kernel on GPU in float32, cast to float64 before Cholesky
  - MLX CPU float64 Cholesky matches numpy/torch speed (~0.04ms for 100x100)
  - GPU float32 matmul is fast (0.54ms for 500x500 vs 0.65ms CPU float64)
- **Alternatives:** float32 everywhere (rejected - numerical instability risk)

## ADR-010: Handle Missing MLX Special Functions
- **Date:** 2026-04-05
- **Decision:** For special functions missing from MLX, use this priority:
  1. Use MLX built-in if available (mx.erf, mx.erfinv, mx.logsumexp)
  2. Implement in pure MLX ops if mathematically feasible (erfc, log_ndtr)
  3. Fall back to numpy/scipy for rare-call-path functions (SobolEngine)
- **Rationale:** Key missing functions identified from code analysis:
  - `torch.special.erfc(x)` (acqf.py:77): Used in `standard_logei()`.
    Can implement as `1 - mx.erf(x)` for moderate x, need careful handling
    for large x where `1 - erf(x)` loses precision.
  - `torch.special.erfcx(x)` (acqf.py:85): Used in logEI small-z branch.
    `erfcx(x) = exp(x^2) * erfc(x)`. Can implement but needs numerical care.
  - `torch.special.log_ndtr(x)` (acqf.py:203): Used in LogPI. Can implement
    as `log(0.5 * erfc(-x/sqrt(2)))` with care for tails.
  - `torch.quasirandom.SobolEngine` (acqf.py:37): Used in EHVI sampling.
    NumPy/SciPy fallback is fine - this is called once per acqf evaluation,
    not in the inner loop. Keep as numpy.
  "You can't always get what you want, but sometimes you can implement it." - Mick Jagger, as quoted by Leonard
- **Status:** ACCEPTED
- **Alternatives:** Implement all in MLX (too much effort for edge cases),
  always fall back to numpy (defeats purpose of GPU acceleration)

## ADR-011: Backend Fallback Strategy (MLX -> torch -> error)
- **Date:** 2026-04-05
- **Updated:** 2026-04-06
- **Decision:** ACCEPTED — Option B (hard fork, MLX-only)
- **Context:** The plan replaces all torch code in `_gp/` with MLX. But Optuna
  currently works on Linux/Windows with torch. If we remove torch entirely,
  non-macOS users lose GP sampler access.
- **Option A:** Backend abstraction layer. Try MLX first,
  fall back to torch, raise ImportError if neither available. This preserves
  backward compatibility. Cost: moderate complexity (two code paths to maintain).
- **Option B (CHOSEN):** Hard fork. MLX-only `_gp/`. Non-macOS users use upstream Optuna.
  Cost: simple code, but limits adoption.
- **Option C:** Dual-file strategy. Keep original torch files as `_gp_torch/`,
  add new MLX files as `_gp_mlx/`, select at import time. Cost: code duplication.
- **Rationale:** This is a fork targeting Apple Silicon. Maintaining dual backends
  doubles the code surface for marginal benefit — non-macOS users already have
  upstream Optuna with full torch support. GPSampler.__init__ raises a clear
  ImportError with "(Apple Silicon only)" when MLX is unavailable. Can revisit
  if there's demand for dual-backend (e.g., MPS via torch on macOS).
- **Added by:** PO, see PO_ROADMAP_REVIEW.md GAP-5
- "As Sheldon would say, 'It's not a choice, it's a series of escalating
  commitments.' But unlike his Roommate Agreement, we can actually change this."

## ADR-012: Autograd Pattern for _fit_kernel_params
- **Date:** 2026-04-05
- **Decision:** PENDING - needs prototype validation (Phase 0.5 risk gate)
- **Context:** `gp.py:_fit_kernel_params()` (lines 287-351) has a loss function
  that mutates `self` attributes (inverse_squared_lengthscales, kernel_scale,
  noise_var) then calls `self.marginal_log_likelihood()`. PyTorch handles this
  via `.requires_grad_(True)` + `.backward()`. MLX needs `mx.value_and_grad(fn)`.
- **Question:** Does `mx.value_and_grad` work when the function closure mutates
  attributes of a captured object? If yes, the migration is straightforward.
  If no, we need to refactor to a flat parameter vector approach.
- **Approach A:** Keep self-mutation pattern, wrap in closure for mx.value_and_grad.
  Simplest migration, but may not work with MLX's tracing-based autograd.
- **Approach B:** Extract all differentiable params into a flat `mx.array`,
  compute loss as a pure function of that array. Assign back to self after.
  More work but guaranteed to work with any autograd system.
- **Status:** OPEN - blocked on Phase 0.5 prototype
- **Added by:** PO, see PO_ROADMAP_REVIEW.md GAP-7
- "The definition of insanity is doing the same backward pass and expecting
  different gradients." - Einstein, if he used autograd
