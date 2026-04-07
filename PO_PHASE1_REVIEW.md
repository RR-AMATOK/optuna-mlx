# ============================================================================
# OPTUNA-MLX PO PHASE 1 REVIEW
# Product Owner Sign-off Assessment
# ============================================================================
# Created: 2026-04-06
# Author: PO (Claude)
# Scope: Review of commit 4fe5234 "feat: Phase 1 — Migrate GP module from
#        PyTorch to MLX" and supporting commits
# ============================================================================

# Phase 1 PO Review: GP Module Migration

## Review Method

Audited the actual committed code against:
- DEV_BRIEF.md work items 1A-1E
- QA_ACCEPTANCE_CRITERIA.md AC-1.1 through AC-1.15
- ADR-001 through ADR-012
- PO_ROADMAP_REVIEW.md gaps and risks

---

## 1. Completeness Check: DEV_BRIEF Work Items

### Work Item 0A-0D (Phase 0): COMPLETE
- [x] MLX installed (0.31.1), PyTorch installed (2.11.0)
- [x] `optuna/_mlx/__init__.py` created with `HAS_MLX`, `is_mlx_available()`, `get_default_device()`
- [x] `pyproject.toml` updated with MLX dependency
- [x] Benchmark baseline captured (`benchmarks/results_torch_baseline.json`)

### Work Item 1A (prior.py): COMPLETE
- [x] `torch` imports replaced with `mx` (lazy import pattern)
- [x] `torch.log()` -> `mx.log()`
- [x] Type hints: `torch.Tensor` -> `mx.array`
- [x] Clean, minimal change. Matches dev brief exactly.

### Work Item 1B (gp.py): COMPLETE — with notable architectural decisions

**Step 1 (imports):** Done. `torch` -> `mx` via `_LazyImport("mlx.core")`.

**Step 2 (Matern52Kernel):** Done — BUT dev went beyond the brief.
- Brief said: "pure function, `mx.grad()` handles derivative automatically"
- Actual: Used `@mx.custom_function` + `@matern52_kernel.vjp` with manual VJP
- **Reason:** The sqrt(0) singularity issue. `mx.grad(mx.sqrt(0))` = infinity,
  same problem the original torch code had. Dev correctly identified this and
  implemented the same fix (manual derivative) using MLX's custom VJP API.
- **PO verdict:** APPROVED. This was the right call. The brief's assumption
  that "MLX's `mx.grad()` handles the derivative automatically" was wrong for
  this specific case. Dev adapted correctly.

**Step 3 (GPRegressor.__init__):** Done.
- `mx.expand_dims` replaces `unsqueeze`
- `mx.square` replaces `.square_()`
- `.astype(mx.float64)` replaces `.type(torch.float64)`
- Categorical handling preserved with `mx.where`

**Step 4 (_cache_matrix):** Done.
- Kept numpy/scipy for Cholesky and solve_triangular (CPU float64)
- `np.array(self.kernel())` for conversion
- `mx.stop_gradient()` replaces `.detach()`
- **Aligned with ADR-009 refined strategy** (CPU float64 for linalg)

**Step 5 (kernel):** Done.
- `@` operator replaces `.matmul()`
- `mx.expand_dims` replaces `.unsqueeze()`
- `matern52_kernel()` replaces `Matern52Kernel.apply()`

**Step 6 (posterior):** Done — with a clever solution.
- `mx.sum(a * b, axis=-1)` replaces `torch.linalg.vecdot`
- `_solve_triangular_right()` helper created for `left=False` case
  (MLX lacks the `left` parameter — this was RISK-2 in PO_ROADMAP_REVIEW.md)
- Diagonal clamping via eye broadcasting (replaces in-place `.clamp_min_()`)
- **PO verdict:** APPROVED. The `_solve_triangular_right` helper is clean and
  well-documented. The diagonal clamping approach is more complex than the
  original but necessary due to MLX's functional style.

**Step 7 (marginal_log_likelihood):** Done — with MAJOR architectural decision.
- Dev created `_differentiable_mll()` with `@mx.custom_function` + VJP
- Implements Rasmussen & Williams eq. 5.9 analytical gradient
- **This was not in the brief.** The brief assumed `mx.linalg.cholesky` would
  be differentiable. It's NOT in MLX 0.31.1.
- **PO verdict:** APPROVED. This is the highest-quality piece of engineering in
  the entire migration. The analytical MLL gradient is well-known (R&W 2006),
  correctly implemented, and solves what would have been a total blocker.

**Step 8 (_fit_kernel_params):** Done.
- `mx.value_and_grad(loss_fn)(raw_params_mx)` replaces torch autograd
- Self-mutation pattern works (Phase 0.5 risk gate answered: IT WORKS)
- `mx.eval()` before numpy conversion
- scipy L-BFGS-B preserved (ADR-008)
- **PO verdict:** APPROVED. The autograd risk (GAP-7, ADR-012) is resolved.

**Step 9 (fit_kernel_params entry):** Done.
- `mx.array(X, dtype=mx.float64)` replaces `torch.from_numpy()`
- `mx.ones((...,), dtype=mx.float64)` replaces `torch.ones()`
- `mx.stream(mx.cpu)` context manager wraps the entire function
- Error handling / fallback preserved

**Step 10 (append_running_data):** Done.
- `mx.concatenate` replaces `torch.cat`
- numpy/scipy preserved for Cholesky (matches _cache_matrix approach)

### Work Item 1C (acqf.py): COMPLETE

- [x] Imports changed to MLX
- [x] `_erfcx()` implemented with asymptotic expansion for large x — good
- [x] `_log_ndtr()` implemented via `mx.erf` — clean one-liner
- [x] Sobol QMC via `scipy.stats.qmc.Sobol` (ADR-010 compliant)
- [x] `eval_acqf_with_grad()` uses `mx.value_and_grad` — correct
- [x] `eval_acqf_no_grad()` drops `torch.no_grad()` context (not needed in MLX)
- [x] `mx.stream(mx.cpu)` used in eval methods — aligned with ADR-009
- [x] All LogEI, LogPI, UCB, LCB, LogEHVI, ConstrainedLogEHVI ported

### Work Item 1D (optim_mixed.py): COMPLETE

- [x] `mx.value_and_grad` replaces torch autograd in `_gradient_ascent_batched`
- [x] `mx.stream(mx.cpu)` context
- [x] numpy conversion with `mx.eval()` before `np.array()`
- [x] Only 6 torch operations replaced — minimal, correct change

### Work Item 1E (tests): COMPLETE (per commit message: 61 tests pass, 0 regressions)

- [x] `test_gp.py` and `test_acqf.py` migrated from torch to MLX
- [x] Zero torch imports in test files

---

## 2. Risk Resolution Check

### RISK-1 (float64 on GPU): RESOLVED
- **Finding:** float64 NOT supported on GPU in MLX 0.31.1
- **Dev response:** Mixed precision strategy — GPU float32 for element-wise,
  CPU float64 for linalg. Implemented via `mx.stream(mx.cpu)`.
- **ADR-009 updated:** Yes, with benchmark data.
- **PO verdict:** Correct response. The strategy is sound.

### RISK-2 (solve_triangular API): RESOLVED
- **Finding:** MLX lacks `left=False` parameter
- **Dev response:** Created `_solve_triangular_right()` helper using transpose trick
- **PO verdict:** Clean solution. Well-documented.

### RISK-3 (Greenlet + MLX): IMPLICITLY RESOLVED
- `mx.eval()` is called before every `np.array()` conversion, which forces
  lazy evaluation. Greenlet switches after numpy conversion are safe.
- **PO verdict:** Acceptable. Should be explicitly tested by QA.

### GAP-5 (Backend fallback — ADR-011): NOT YET RESOLVED
- The current code has NO fallback to torch. If MLX is not installed,
  `GPSampler` will fail with `ModuleNotFoundError` on `mlx.core`.
- The early import check (M-5 fix) was for TORCH. Now that torch is removed,
  we need the same check for MLX.
- **Status:** OPEN ISSUE. See action items below.

### GAP-7 (Autograd risk — ADR-012): RESOLVED
- `mx.value_and_grad` works with self-mutating closures.
- **PO verdict:** Phase 0.5 risk gate PASSED.

---

## 3. Architectural Quality Assessment

### Strengths

1. **Custom VJPs are excellent.** The `_differentiable_mll` with analytical
   Rasmussen & Williams gradient is production-quality numerical code. This
   was the hardest problem in the migration and was solved elegantly.

2. **`_solve_triangular_right` helper.** Clean abstraction for a missing API.
   Well-documented with the mathematical identity.

3. **Consistent use of `mx.stream(mx.cpu)`.** The mixed precision strategy
   (GPU for element-wise, CPU for linalg) is applied consistently across
   all files via stream context managers.

4. **dtype discipline.** Every `mx.array()` creation includes explicit
   `dtype=mx.float64`. The M-7 dtype audit concern is addressed.

5. **Zero torch residue.** Only comments reference torch, for historical context.
   No actual torch imports or usage remain.

### Concerns

1. **`_erfcx` asymptotic expansion** (acqf.py:33-47): Uses 8 terms of a
   divergent asymptotic series. For x near the transition point (x ~ 4),
   accuracy may be suboptimal. QA should verify against `scipy.special.erfcx`
   across the transition region (x in [3, 6]).

2. **`_log_ndtr` is simplified** (acqf.py:64-66): The implementation is
   `mx.log(0.5 * (1 + mx.erf(x * sqrt(0.5))))`. This loses precision for
   large negative x (where `erf(x) ≈ -1`, so `1 + erf(x) ≈ 0`). The original
   torch version `torch.special.log_ndtr` handles this with a tail expansion.
   QA should test with x = -30, -20, -10 to verify if this matters for the
   acqf use cases (it may not, since LogPI clips inputs).

3. **No `_log_ndtr` tail handling for LogPI** (acqf.py): LogPI calls `_log_ndtr`
   directly. If z is very negative, the simplified implementation may return
   `-inf` where the torch version returned a finite (very negative) value.
   This could affect constraint handling in edge cases.

4. **`_differentiable_mll` computes L_inv as a full matrix** (gp.py:132):
   `L_inv = mx.linalg.solve_triangular(L, mx.eye(n), upper=False)` then
   `W = L_inv.T @ L_inv`. For large n, this is O(n^3) in memory and compute.
   The original torch code used `L.diagonal().log().sum()` and separate solves.
   For n <= 500 (typical GP), this is fine. For larger problems, may need
   optimization in Phase 6.

---

## 4. PO Checklist: Definition of Done (from DEV_BRIEF.md)

1. [x] Zero `import torch` in `optuna/_gp/` directory — VERIFIED
2. [x] All files use `import mlx.core as mx` via lazy import — VERIFIED
3. [x] All existing GP tests pass with MLX — CLAIMED (61 pass, 0 regressions)
4. [ ] `python -c "import optuna; optuna.samplers.GPSampler"` works without
       torch installed — **NOT VERIFIED** (needs QA; torch is still installed)
5. [ ] Simple `study.optimize()` with GPSampler completes successfully —
       **NOT VERIFIED** (needs QA end-to-end test)

---

## 5. PO Decision

### Phase 1 Status: CONDITIONAL PASS

The code migration is complete and high quality. The architectural decisions
(custom VJPs, mixed precision, solve_triangular helper) are sound. However,
three items must be resolved before full sign-off:

### MUST FIX (blocking QA sign-off)

**MF-1: GPSampler import check needs updating for MLX**
The Pre-Phase fix (M-5) added a check for torch. Now that GP uses MLX,
the check should verify MLX is available, not torch. Without this, users
on Linux (where MLX doesn't exist) get an unhelpful `ModuleNotFoundError`.

```python
# sampler.py should check for mlx.core, not torch
try:
    import mlx.core
except ImportError:
    raise ImportError(
        "GPSampler requires MLX (Apple Silicon only). "
        "Install with: pip install optuna[mlx]"
    ) from None
```

**MF-2: ADR-011 (backend fallback) needs a decision**
Currently the GP module is MLX-only. Non-macOS users lose access to GPSampler.
Options per ADR-011:
- Option A: Backend abstraction (MLX preferred, torch fallback)
- Option B: Hard fork (MLX-only, non-macOS uses upstream)

**PO recommendation:** Option B for now. This is a fork; non-macOS users should
use upstream Optuna. We can revisit if there's demand for dual-backend.

### SHOULD FIX (before QA sign-off)

**SF-1: `_log_ndtr` tail accuracy**
Add a tail-safe branch for x < -5:
```python
def _log_ndtr(x):
    # For large negative x, use log(erfc(-x/sqrt2)/2) = log(erfc) - log(2)
    safe = x > -5.0
    standard = mx.log(0.5 * (1.0 + mx.erf(x * _SQRT_HALF)))
    # Tail: use erfcx-based computation to avoid log(0)
    tail = -0.5 * mx.square(x) - _LOG_SQRT_2PI + mx.log(_erfcx(-x * _SQRT_HALF)) - mx.log(mx.array(2.0))
    return mx.where(safe, standard, tail)
```

### QA ITEMS (for QA to verify)

| # | Item | AC Reference | Priority |
|---|------|-------------|----------|
| Q-1 | Run all 61 GP tests independently | AC-1.13 | HIGH |
| Q-2 | `study.optimize()` with GPSampler, 50 trials | AC-1.12 | HIGH |
| Q-3 | Verify GPSampler works without torch installed | AC-1.1 | HIGH |
| Q-4 | `erfcx` accuracy at transition region (x ∈ [3,6]) | AC-1.9 | HIGH |
| Q-5 | `_log_ndtr` accuracy for x = -30, -20, -10 | AC-1.9 | HIGH |
| Q-6 | Greenlet + MLX interaction (batched L-BFGS-B) | AC-1.10 | MEDIUM |
| Q-7 | Memory stability over 200 trials | AC-1.12 | MEDIUM |
| Q-8 | MLX benchmark vs torch baseline | AC-1.14 | MEDIUM |
| Q-9 | Numerical parity: kernel, posterior, MLL | AC-1.2-1.6 | HIGH |
| Q-10 | Gradient parity: MLL grad, acqf grad | AC-1.8 | HIGH |

---

## 6. Updated Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| Pre-Phase (QA Fix-Now) | COMPLETE | All 8 items done |
| Phase 0 (Setup) | COMPLETE | MLX env, utility module, pyproject, benchmarks |
| Phase 0.5 (Risk Gate) | PASSED | mx.value_and_grad + self-mutation works |
| **Phase 1 (GP Module)** | **CONDITIONAL PASS** | Code done, MF-1/MF-2 + QA pending |
| Phase 2 (TPE) | NOT STARTED | Blocked on Phase 1 QA sign-off |
| Phases 3-5 | CONDITIONAL | Proceed only if justified |
| Phase 6 (Optimization) | NOT STARTED | After Phase 2 |

---

## 7. Action Items for Dev

| # | Action | Priority | Blocks |
|---|--------|----------|--------|
| 1 | Update GPSampler import check for MLX (MF-1) | HIGH | QA |
| 2 | Decide ADR-011: recommend Option B (hard fork) (MF-2) | HIGH | QA |
| 3 | Add `_log_ndtr` tail-safe branch (SF-1) | MEDIUM | QA sign-off |
| 4 | Update TODO.md: mark Phase 0, 0.5, 1 items as done | LOW | — |
| 5 | Update CHANGELOG_MLX.md with Phase 1 changes | LOW | — |

## 8. Action Items for QA

Execute Q-1 through Q-10 above. Report format per QA_ACCEPTANCE_CRITERIA.md.
Special attention to Q-4 and Q-5 (special function accuracy at boundaries).
