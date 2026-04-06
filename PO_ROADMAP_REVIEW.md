# ============================================================================
# OPTUNA-MLX ROADMAP REVIEW
# Product Owner Audit
# ============================================================================
# Created: 2026-04-05
# Author: PO (Claude)
# Status: ACTIVE
# ============================================================================

# Roadmap Review: Plan vs Reality

## Executive Summary

After auditing the actual codebase against the documented plan in TODO.md,
DECISIONS.md, and NOTES.md, the plan is **solid but has 7 gaps and 3 risks**
that need addressing before Phase 1 development begins.

**Overall verdict:** Plan is APPROVED with amendments below.

---

## 1. GAPS FOUND

### GAP-1: `prior.py` missing from TODO.md
**Finding:** `optuna/_gp/prior.py` (33 lines) has 5 torch operations (torch.log,
torch.Tensor types) but is NOT listed anywhere in TODO.md Phase 1 tasks.

**Impact:** If dev migrates gp.py and acqf.py but forgets prior.py, the GP module
will still depend on torch at runtime. prior.py is called from gp.py:323 via
`log_prior(self)` in the loss function.

**Resolution:** Added to DEV_BRIEF.md as Work Item 1A (warmup). Must also be
added to TODO.md under Phase 1.

### GAP-2: `search_space.py` Sobol dependency
**Finding:** `optuna/_gp/search_space.py` (226 lines) uses `scipy.stats.qmc.Sobol`
(line noted in exploration). While it has NO torch dependency, it IS part of the
GP pipeline and uses scipy for Sobol sampling.

**Impact:** None for Phase 1 (no torch to remove). But the NOTES.md lists
`torch.quasirandom.SobolEngine` as used in acqf.py:37 — the replacement plan
correctly targets scipy as the fallback. search_space.py already uses scipy
Sobol, so we have a working pattern to copy.

**Resolution:** No action needed. Document as reference for acqf.py Sobol replacement.

### GAP-3: `optim_sample.py` not mentioned in TODO.md
**Finding:** `optuna/_gp/optim_sample.py` (23 lines) has NO torch dependency.
It's pure numpy and calls `acqf.eval_acqf_no_grad()`.

**Impact:** Zero. No migration needed. But it should be verified that it still
works after acqf.py is migrated (since the acqf return types change from
torch.Tensor to mx.array, but `eval_acqf_no_grad` already returns numpy).

**Resolution:** Add to Phase 1E testing: verify optim_sample.py still works.

### GAP-4: `scipy_blas_thread_patch.py` fate unclear
**Finding:** NOTES.md says "This becomes unnecessary since MLX uses its own
Metal compute pipeline." But TODO.md doesn't mention removing or bypassing it.

**Impact:** Low. The patch is harmless if left in (it limits OpenBLAS threads,
which doesn't affect MLX). But it's called from gp.py:27 as a context manager
wrapping scipy operations.

**Resolution:** Keep as-is for Phase 1. If we remove scipy calls from
`_cache_matrix()` (replacing numpy Cholesky with MLX Cholesky), revisit whether
this patch is still needed. Not a blocker.

### GAP-5: No fallback strategy if MLX is not installed
**Finding:** The plan assumes MLX will replace torch entirely in `_gp/`. But
Optuna currently works with torch as an optional dep. If someone installs
optuna WITHOUT mlx (e.g., on Linux), the GP sampler will break.

**Impact:** HIGH. This is a breaking change for non-macOS users.

**Resolution:** Two options:
- **Option A (Recommended):** Keep torch as a fallback. If MLX is available, use
  MLX. If not, fall back to torch. If neither, raise ImportError (current behavior).
  This requires a backend selection layer.
- **Option B:** Make this a hard fork. MLX-only. Non-macOS users use upstream Optuna.

**Decision needed:** This is a new ADR (ADR-011). PO recommends Option A for
Phase 1 (maintain compatibility), with Option B as a future consideration.

### GAP-6: Test infrastructure for dual-backend testing
**Finding:** Current GP tests directly `import torch`. The QA acceptance criteria
require comparing torch vs MLX outputs. But the test files don't have any
infrastructure for running the same test with different backends.

**Impact:** Medium. QA can't efficiently validate parity without test fixtures
that parameterize over backends.

**Resolution:** Create a `conftest.py` in `tests/gp_tests/` with:
```python
@pytest.fixture(params=["mlx", "torch"])
def backend(request):
    if request.param == "mlx":
        pytest.importorskip("mlx.core")
    elif request.param == "torch":
        pytest.importorskip("torch")
    return request.param
```
Dev should implement this as part of Work Item 1E.

### GAP-7: Missing ADR for `mx.value_and_grad` + self-mutation pattern
**Finding:** The most complex migration in gp.py is `_fit_kernel_params()`
(lines 287-351), where the loss function mutates `self` (assigns new values
to `self.inverse_squared_lengthscales`, `self.kernel_scale`, `self.noise_var`)
and then calls `self.marginal_log_likelihood()`.

MLX's `mx.value_and_grad()` expects a pure function. The self-mutation pattern
may or may not work with MLX's autograd. This is the single highest-risk code
change in the entire project and there's no ADR documenting the approach.

**Impact:** HIGH. If `mx.value_and_grad` doesn't support closures that mutate
captured objects, the entire `_fit_kernel_params` restructuring plan fails.

**Resolution:** Need ADR-011 (or ADR-012) documenting:
1. Whether `mx.value_and_grad` works with closures that capture `self`
2. A fallback plan: extract all GP state into a flat parameter array, compute
   loss as a pure function of that array, then assign back to self after.
3. Dev should prototype this FIRST before migrating the rest of gp.py.

---

## 2. RISKS IDENTIFIED

### RISK-1: float64 Performance on Apple Silicon (MEDIUM)
**ADR-009 status:** "ACCEPTED (pending float64 benchmark)"

The benchmark hasn't been run yet. MLX documentation suggests float64 may
fall back to CPU on some operations or be significantly slower on GPU.

**Mitigation:** Work Item 0A explicitly requires this benchmark. If float64 is
slow, we have three fallback options documented in ADR-009. This risk is
well-managed.

### RISK-2: `mx.linalg.solve_triangular` API compatibility (MEDIUM)
**Finding:** torch's `solve_triangular` has signature:
```python
torch.linalg.solve_triangular(A, B, upper=False, left=True)
```
MLX's may differ. The `left` parameter (whether to solve AX=B or XA=B) is
critical for the posterior computation in gp.py:234-236.

**Mitigation:** Dev must verify the exact MLX API signature in Work Item 0A.
If `left` parameter is missing, implement as transpose trick:
`solve(A.T, B.T).T` for the right-solve case.

### RISK-3: Greenlet + MLX interaction (LOW)
**Finding:** `batched_lbfgsb.py` uses greenlet coroutines to interleave multiple
scipy L-BFGS-B instances. The callback from scipy calls acqf functions that
will now use MLX internally.

MLX uses lazy evaluation. If a greenlet switch happens between an MLX operation
and its `mx.eval()`, the computation graph could be in an unexpected state.

**Mitigation:** Ensure `mx.eval()` is called before returning numpy values from
any callback that bridges MLX -> scipy. This is already implied by the
`np.array(mx_result)` conversion (which forces evaluation), but should be
explicitly verified.

---

## 3. PLAN ALIGNMENT WITH ADRs

| ADR | Alignment | Notes |
|-----|-----------|-------|
| ADR-001 (MLX backend) | ALIGNED | All migration targets use MLX |
| ADR-002 (Harness approach) | ALIGNED | PO, QA, Dev roles defined |
| ADR-003 (Parity first) | ALIGNED | QA criteria enforce parity before perf |
| ADR-004 (GP module first) | ALIGNED | Phase 1 = GP module |
| ADR-005 (Dev branch) | ALIGNED | All work on dev branch |
| ADR-006 (Priority order) | ALIGNED | Phases 1-5 match priority |
| ADR-007 (Functional autograd) | ALIGNED | Dev brief covers this |
| ADR-008 (Keep scipy L-BFGS-B) | ALIGNED | batched_lbfgsb.py unchanged |
| ADR-009 (float64 policy) | PENDING | Needs benchmark (Work Item 0A) |
| ADR-010 (Missing special funcs) | ALIGNED | erfc/erfcx/log_ndtr plan in brief |

**New ADRs needed:**
- **ADR-011:** Backend fallback strategy (MLX -> torch -> error)
- **ADR-012:** `mx.value_and_grad` with self-mutating closures approach

---

## 4. TIMELINE CONCERNS

The TODO.md has no dates. As PO, I'm not estimating timelines (per instructions),
but I note the dependency chain:

```
0A (env) ─────────────> 0B (utility) ──> 0C (pyproject) ──> 0D (benchmark)
                                                                    │
    ┌───────────────────────────────────────────────────────────────┘
    ▼
1A (prior.py) ──> 1B (gp.py) ──> 1C (acqf.py) ──> 1D (optim_mixed.py) ──> 1E (tests)
                      │
                      └── Step 8 (_fit_kernel_params) is the critical path risk
```

**Critical path:** Work Item 0A (float64 benchmark) and 1B Step 8 (autograd
restructuring) are the two items most likely to surface blockers.

**Recommendation:** Dev should prototype 1B Step 8 immediately after 0A passes.
If the `mx.value_and_grad` + self-mutation pattern doesn't work, we need to
know before investing in the rest of the migration.

---

## 5. SCOPE CREEP WATCH

Items that are NOT in scope for Phase 1 but might tempt the dev:
- Performance optimization (lazy eval, mx.compile) — Phase 6 only
- TPE migration — Phase 2 only
- New features (e.g., MLX-native L-BFGS-B) — out of scope entirely
- Mixed precision strategies — only if ADR-009 forces it
- CI/CD changes — out of scope until integration test suite is solid

**PO directive:** If dev encounters something that "would be easy to fix while
we're here," they must check with PO first. No drive-by improvements.

---

## 6. AMENDMENTS TO TODO.md

The following should be added to TODO.md:

### Under Phase 1:
```
### 1F. Prior Module (`prior.py`, 33 LOC)
- [ ] Replace torch imports with MLX (5 torch ops)
- [ ] torch.log() -> mx.log()
- [ ] torch.Tensor types -> mx.array types
```

### Under Phase 1E:
```
- [ ] Verify optim_sample.py still works (no torch, calls acqf)
- [ ] Create dual-backend test fixture in tests/gp_tests/conftest.py
- [ ] Prototype mx.value_and_grad with self-mutating closure (RISK GATE)
```

### New ADR entries for DECISIONS.md:
```
## ADR-011: Backend Fallback (MLX -> torch -> error)
## ADR-012: Autograd pattern for _fit_kernel_params
```

---

## PO APPROVAL

**Roadmap status:** APPROVED WITH AMENDMENTS

**Conditions:**
1. Add GAP-1 (prior.py) to TODO.md before dev starts Phase 1
2. Dev prototypes `mx.value_and_grad` closure pattern (GAP-7) as first code task
3. ADR-011 (backend fallback) decision made before Phase 1 completion
4. Work Item 0A (float64 benchmark) must pass before any Phase 1 work begins

**Next action:** Dev starts Work Item 0A. Report results. PO reviews and greenlights Phase 1.
