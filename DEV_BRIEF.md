# ============================================================================
# OPTUNA-MLX DEV BRIEF
# Product Owner -> Dev Team
# ============================================================================
# Created: 2026-04-05
# Author: PO (Claude)
# Status: ACTIVE
# ============================================================================

# Dev Brief: Phase 0 Remaining + Phase 1 Kickoff

## Priority: Do these in exact order. Do not skip ahead.

---

## WORK ITEM 0A: MLX Environment Verification

**Goal:** Confirm MLX works on this machine before writing any code.

**Steps:**
```bash
# 1. Install MLX
pip install mlx>=0.5.0

# 2. Verify import and GPU
python -c "
import mlx.core as mx
print('MLX version:', mx.__version__)
print('Default device:', mx.default_device())
a = mx.array([1.0, 2.0, 3.0])
print('Basic array:', a)
print('dtype:', a.dtype)
"

# 3. Verify float64 (CRITICAL - see ADR-009)
python -c "
import mlx.core as mx
a = mx.array([1.0], dtype=mx.float64)
print('float64 supported:', a.dtype)
b = mx.linalg.cholesky(mx.eye(3, dtype=mx.float64))
print('float64 Cholesky works:', b.dtype)
"

# 4. Verify key APIs exist
python -c "
import mlx.core as mx
print('mx.erf:', hasattr(mx, 'erf'))
print('mx.erfinv:', hasattr(mx, 'erfinv'))
print('mx.logsumexp:', hasattr(mx, 'logsumexp'))
print('mx.linalg.cholesky:', hasattr(mx.linalg, 'cholesky'))
print('mx.linalg.solve_triangular:', hasattr(mx.linalg, 'solve_triangular'))
print('mx.grad:', hasattr(mx, 'grad'))
print('mx.value_and_grad:', hasattr(mx, 'value_and_grad'))
print('mx.stop_gradient:', hasattr(mx, 'stop_gradient'))
print('mx.compile:', hasattr(mx, 'compile'))
"

# 5. Benchmark float64 vs float32 Cholesky (100x100)
python -c "
import mlx.core as mx
import time
import numpy as np

for dtype_name, dtype in [('float32', mx.float32), ('float64', mx.float64)]:
    n = 100
    A_np = np.random.randn(n, n)
    A_np = A_np @ A_np.T + n * np.eye(n)
    A = mx.array(A_np, dtype=dtype)
    mx.eval(A)

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        L = mx.linalg.cholesky(A)
        mx.eval(L)
        times.append(time.perf_counter() - t0)
    print(f'{dtype_name}: median={np.median(times)*1000:.2f}ms, '
          f'mean={np.mean(times)*1000:.2f}ms')
"
```

**Report back:** Device name, float64 support (yes/no), float64 Cholesky timing vs float32, any missing APIs.

**Blocker if:** float64 is not supported or is >10x slower than float32. In that case, escalate to PO for ADR-009 decision.

---

## WORK ITEM 0B: Create `optuna/_mlx/` Utility Module

**Goal:** Backend detection and lazy import infrastructure.

**File: `optuna/_mlx/__init__.py`**

Follow the exact same lazy import pattern used in `optuna/_gp/gp.py:32-41`:

```python
"""MLX backend utilities for Optuna.

Provides backend detection and helper functions for MLX GPU acceleration
on Apple Silicon.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import mlx.core as mx
else:
    try:
        import mlx.core as mx
        HAS_MLX = True
    except ImportError:
        HAS_MLX = False
        mx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def is_mlx_available() -> bool:
    """Check if MLX is available."""
    return HAS_MLX


def get_default_device() -> str:
    """Return the MLX default device name, or 'cpu' if MLX unavailable."""
    if HAS_MLX:
        return str(mx.default_device())
    return "cpu"
```

**Key constraints:**
- Do NOT import torch in this module
- Do NOT create array_ops.py, linalg.py, random.py, grad.py yet (those come in Phase 1)
- Keep it minimal - just detection for now

---

## WORK ITEM 0C: Add MLX to `pyproject.toml`

**File:** `pyproject.toml`

**Edit 1** - Add `mlx` extra after line 89 (after the `optional` list closing bracket):
```toml
mlx = [
  "mlx>=0.5.0",  # optuna/_mlx. Apple Silicon GPU acceleration.
]
```

**Edit 2** - Add mlx to `optional` list (after line 86, the greenlet line):
```toml
  "mlx>=0.5.0; sys_platform == 'darwin'",  # optuna/_mlx. Apple Silicon GPU.
```

**Why platform guard:** MLX only works on macOS. Non-mac installs should not pull it.

---

## WORK ITEM 0D: Baseline Benchmark Suite

**Goal:** Capture BEFORE numbers so we can measure MLX improvement.

**File: `benchmarks/bench_gp.py`** (new file)

Benchmark these operations using the CURRENT torch backend:
1. `GPRegressor.kernel()` — n_trials in [10, 50, 100, 500], n_params in [2, 10, 50]
2. `GPRegressor._cache_matrix()` — same sizes
3. `GPRegressor.posterior()` — batch sizes [1, 10, 100]
4. `GPRegressor.marginal_log_likelihood()` — n_trials in [10, 50, 100]
5. `fit_kernel_params()` — full optimization, n_trials in [10, 50, 100]
6. `eval_acqf` on 2048 candidates (the `optimize_acqf_mixed` hot path)
7. End-to-end `study.optimize()` with GPSampler, 100 trials

**Output format:** JSON with median/mean/std for each benchmark.

**Important:** Run on the same Apple Silicon machine that will run MLX. Save results to `benchmarks/results_torch_baseline.json`.

---

## WORK ITEM 1A: Start GP Module Migration — `prior.py` (Warmup)

**Why start here:** `prior.py` is 33 lines with only 5 torch ops. Perfect warmup to validate the pattern before touching gp.py.

**File:** `optuna/_gp/prior.py` (33 lines)

**Current torch usage (exact):**
- Line 7: `import torch` (TYPE_CHECKING)
- Line 13: `torch = _LazyImport("torch")`
- Line 19: `-> torch.Tensor` return type
- Line 22: `x: torch.Tensor`, `-> torch.Tensor`
- Line 24: `torch.log(x)`
- Line 30: `torch.log()`
- Line 31-32: calls to `gamma_log_prior` (returns torch.Tensor)

**MLX replacement:**
```python
if TYPE_CHECKING:
    import mlx.core as mx
    from optuna._gp import gp
else:
    from optuna._imports import _LazyImport
    mx = _LazyImport("mlx.core")

def default_log_prior(gpr: gp.GPRegressor) -> mx.array:
    def gamma_log_prior(x: mx.array, concentration: float, rate: float) -> mx.array:
        return (concentration - 1) * mx.log(x) - rate * x
    # rest stays the same — gpr attributes become mx.array instead of torch.Tensor
```

**Test:** `tests/gp_tests/test_gp.py` creates GPRegressor and uses prior. After changing prior.py, all existing GP tests must still pass (once gp.py is also migrated).

**Note:** prior.py CANNOT be tested in isolation until gp.py is also migrated (it depends on GPRegressor attributes being mx.array). So this is a "change and hold" — commit but don't expect tests to pass yet.

---

## WORK ITEM 1B: GP Module Migration — `gp.py` (The Big One)

**File:** `optuna/_gp/gp.py` (409 lines, ~50 torch ops)

**Migration order within gp.py (do these in sequence):**

### Step 1: Change imports (lines 32-41)
```python
if TYPE_CHECKING:
    from collections.abc import Callable
    import mlx.core as mx
    import scipy
else:
    from optuna._imports import _LazyImport
    mx = _LazyImport("mlx.core")
    scipy = _LazyImport("scipy")
```
Remove `torch` entirely.

### Step 2: Matern52Kernel class -> pure function (lines 63-90)
**Current:** `class Matern52Kernel(torch.autograd.Function)` with `forward()` and `backward()`
**Replace with:**
```python
def matern52_kernel(squared_distance: mx.array) -> mx.array:
    sqrt5d = mx.sqrt(5.0 * mx.maximum(squared_distance, 0.0))
    return mx.exp(-sqrt5d) * (1.0 + sqrt5d + (5.0 / 3.0) * squared_distance)
```
MLX's `mx.grad()` handles the derivative automatically. The manual backward in lines 84-89 is no longer needed.

### Step 3: GPRegressor.__init__ (lines 93-118)
Replace all `torch.Tensor` types with `mx.array`.
- Line 108: `(X.unsqueeze(-2) - X.unsqueeze(-3)).square_()` -> `mx.square(mx.expand_dims(X, -2) - mx.expand_dims(X, -3))`
- Line 111: `.type(torch.float64)` -> `.astype(mx.float64)`

### Step 4: GPRegressor._cache_matrix (lines 124-149)
- Line 122: `.detach().cpu().numpy()` -> `np.array(self.kernel())` (after `mx.eval()`)
- Lines 128-129: Keep `np.linalg.cholesky` OR use `mx.linalg.cholesky`
  - **Decision needed:** If float64 Cholesky is fast on GPU (from Work Item 0A), use MLX. Otherwise keep numpy.
- Lines 142-149: `torch.from_numpy()` -> `mx.array()`, `.detach()` -> `mx.stop_gradient()`

### Step 5: GPRegressor.kernel() (lines 185-213)
- Line 207: `.square_()` -> `mx.square()`
- Line 209: `.type(torch.float64)` -> `.astype(mx.float64)`
- Line 212: `.matmul()` -> `@` operator
- Line 213: `Matern52Kernel.apply()` -> `matern52_kernel()`

### Step 6: GPRegressor.posterior() (lines 215-250)
- Line 232: `torch.linalg.vecdot(a, b)` -> `mx.sum(a * b, axis=-1)`
- Line 234-236: `torch.linalg.solve_triangular()` -> `mx.linalg.solve_triangular()`
- Line 245: `.clamp_min_(0.0)` -> `mx.maximum(var, 0.0)`

### Step 7: GPRegressor.marginal_log_likelihood() (lines 252-285)
- Line 280: `torch.eye(n, dtype=torch.float64)` -> `mx.eye(n)`
- Line 281: `torch.linalg.cholesky()` -> `mx.linalg.cholesky()`
- Line 282: `.diagonal().log().sum()` -> `mx.sum(mx.log(mx.diag(L)))`
- Line 283: `torch.linalg.solve_triangular()` -> `mx.linalg.solve_triangular()`
- **CRITICAL:** This function must remain differentiable for `_fit_kernel_params()`

### Step 8: GPRegressor._fit_kernel_params() (lines 287-351) — HARDEST PART
**Current pattern (lines 310-327):**
```python
raw_params_tensor = torch.from_numpy(raw_params).requires_grad_(True)
with torch.enable_grad():
    # ... build params from raw_params_tensor ...
    loss = -self.marginal_log_likelihood() - log_prior(self)
loss.backward()
return loss.detach().cpu().numpy(), raw_params_tensor.grad.detach().cpu().numpy()
```
**MLX pattern:**
```python
def loss_fn(raw_params_mx):
    # ... build params from raw_params_mx (same math) ...
    # ... assign to self ...
    return -(self.marginal_log_likelihood() + log_prior(self))

loss_val, grad_val = mx.value_and_grad(loss_fn)(mx.array(raw_params))
mx.eval(loss_val, grad_val)
return np.array(loss_val), np.array(grad_val)
```
**Key challenge:** `mx.value_and_grad` needs a pure function. The current code mutates `self` inside the loss function. This works in MLX if the function signature takes the differentiable params as arguments and the self-mutation is for intermediate state only.

### Step 9: fit_kernel_params() entry point (lines 354-409)
- Line 364: `torch.ones(..., dtype=torch.float64)` -> `mx.ones((...,))`
- Lines 369-371, 388-390: `torch.from_numpy()` -> `mx.array()`

### Step 10: append_running_data() (lines 151-183)
- `torch.cat()` -> `mx.concatenate()`
- `.cpu().numpy()` -> `np.array()`

---

## WORK ITEM 1C: GP Module Migration — `acqf.py`

**File:** `optuna/_gp/acqf.py` (402 lines, ~50 torch ops)

**Do after gp.py is working.** Key changes:
- Replace `torch` imports with `mx` (same pattern as gp.py)
- `torch.special.erfc()` -> implement `1.0 - mx.erf(x)` (for moderate x) or custom
- `torch.special.erfcx()` -> implement `mx.exp(mx.square(x)) * (1.0 - mx.erf(x))`
- `torch.special.log_ndtr()` -> implement `mx.log(0.5 * (1.0 - mx.erf(-x / mx.sqrt(mx.array(2.0)))))`
- `torch.quasirandom.SobolEngine` -> use `scipy.stats.qmc.Sobol` (numpy fallback, per ADR-010)
- `BaseAcquisitionFunc.eval_acqf_with_grad()` (line 108-113): `val.backward()` + `.grad` -> `mx.value_and_grad(self.eval_acqf)(mx.array(x))`
- `torch.no_grad()` context -> just call function directly (MLX doesn't track grads by default)

---

## WORK ITEM 1D: GP Module Migration — `optim_mixed.py`

**File:** `optuna/_gp/optim_mixed.py` (329 lines, 6 torch ops)

**Lightest touch.** Only 6 torch operations, all in `_gradient_ascent_batched()` lines 64-68:
- Line 64: `torch.from_numpy().requires_grad_(True)` -> pass to `mx.value_and_grad`
- Lines 65-68: `eval_acqf` + `.backward()` + `.grad` -> functional grad approach

---

## WORK ITEM 1E: Update Tests

**Files:**
- `tests/gp_tests/test_gp.py` (195 lines) — change `import torch` to `import mlx.core as mx`
- `tests/gp_tests/test_acqf.py` (143 lines) — same
- `tests/samplers_tests/test_gp.py` (162 lines) — same

**Replace all:**
- `torch.Tensor` -> `mx.array`
- `torch.from_numpy()` -> `mx.array()`
- `torch.set_grad_enabled()` -> remove (MLX doesn't need it)
- Tensor assertions -> `np.testing.assert_allclose(np.array(mlx_result), expected, rtol=...)`

**Tolerances (from TODO.md):**
- Kernel values: rtol=1e-10
- Cholesky: rtol=1e-8
- Posterior mean/var: rtol=1e-8
- Marginal log likelihood: rtol=1e-6
- fit_kernel_params: rtol=1e-3 (optimizer-dependent)
- LogEI, UCB, LCB: rtol=1e-8

---

## Definition of Done (Phase 1)

Dev considers Phase 1 done when:
1. Zero `import torch` in `optuna/_gp/` directory
2. All files use `import mlx.core as mx` via lazy import
3. All existing GP tests pass with `import mlx.core as mx`
4. `python -c "import optuna; optuna.samplers.GPSampler"` works without torch installed
5. A simple `study.optimize()` with GPSampler completes successfully

**Then:** Tag PO for review. PO will validate against acceptance criteria. QA follows.
