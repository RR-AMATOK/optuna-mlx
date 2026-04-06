# ============================================================================
# OPTUNA-MLX REPRIORITIZATION ANALYSIS
# Product Owner Decision
# ============================================================================
# Created: 2026-04-05
# Author: PO (Claude)
# Status: DECISION MADE
# ============================================================================

# Reprioritization: Does the Plan Need to Change?

## Analysis Method

Compared the original priority order (ADR-006) against actual code analysis:
- Counted exact torch/numpy operations per file
- Assessed migration difficulty per file
- Evaluated impact on end users
- Identified hidden dependencies and risks

---

## Original Priority (ADR-006)

1. GP module (torch replacement) — ~1308 LOC
2. TPE sampler (numpy hot paths) — ~912 LOC
3. Hypervolume (matrix operations)
4. Multi-objective Pareto front
5. NSGAII crossover operations

---

## Actual Code Reality

### GP Module (`_gp/`) — Confirmed as #1

| File | Lines | Torch Ops | Difficulty | Impact |
|------|-------|-----------|------------|--------|
| gp.py | 409 | ~50 | HIGH (autograd restructure) | Critical |
| acqf.py | 402 | ~50 | HIGH (special functions) | Critical |
| optim_mixed.py | 329 | 6 | LOW | Medium |
| prior.py | 33 | 5 | LOW | Required |
| batched_lbfgsb.py | 168 | 0 | NONE | None |
| search_space.py | 226 | 0 | NONE | None |
| optim_sample.py | 23 | 0 | NONE | None |
| scipy_blas_thread_patch.py | 48 | 0 | NONE | None |

**Effective scope:** 4 files need changes (gp.py, acqf.py, optim_mixed.py, prior.py).
Not 4 files as originally estimated — it's really 2 hard files + 2 easy files.

**Verdict:** Correctly prioritized as #1. The torch->MLX swap is the cleanest
migration path and has the highest per-operation impact.

### TPE Module (`samplers/_tpe/`) — Confirmed as #2

| File | Lines | NumPy Ops | Difficulty | Impact |
|------|-------|-----------|------------|--------|
| _erf.py | 142 | 23 | TRIVIAL (replace with mx.erf) | High |
| _truncnorm.py | 296 | 42 | HIGH (Newton iteration) | High |
| probability_distributions.py | 223 | 41 | MEDIUM | Critical (hot path) |
| parzen_estimator.py | 251 | 40 | MEDIUM | Medium |

**Key finding:** `_erf.py` is a massive win (142 LOC -> 1 function call).
`probability_distributions.py:log_pdf()` is THE hottest path in TPE and has
the most GPU acceleration potential.

**Verdict:** Correctly prioritized as #2. The `_erf.py` replacement alone is
worth the phase. The `log_pdf()` hot path acceleration could be the biggest
user-visible speedup in the entire project.

### Phases 3-5: No Change

The hypervolume, multi-objective, and NSGAII modules are all numpy-based with
no torch dependency. Their priority order is correct: hypervolume has the most
matrix operations, multi-objective has O(n^2) comparisons, and NSGAII operations
are too small to benefit from GPU at typical scales.

---

## Reprioritization Decision

### WITHIN Phase 1: Reorder sub-tasks

**Original order:** gp.py -> acqf.py -> optim_mixed.py -> batched_lbfgsb.py
**New order:**

1. **prior.py** (warmup, 5 torch ops) — ADDED, was missing
2. **gp.py Step 8 prototype** (`_fit_kernel_params` autograd pattern) — ELEVATED to risk gate
3. **gp.py full migration** (remaining steps)
4. **acqf.py** (depends on gp.py working)
5. **optim_mixed.py** (depends on acqf.py working)
6. **Tests** (after all code changes)

**Rationale:** The autograd pattern in `_fit_kernel_params` is the single
highest-risk item. If it doesn't work, we need to know immediately — not after
spending effort on the rest of gp.py. Prototype it as a spike.

### WITHIN Phase 2: Reorder sub-tasks

**Original order:** _erf.py -> _truncnorm.py -> probability_distributions.py -> parzen_estimator.py
**New order:**

1. **_erf.py** (trivial, massive LOC reduction) — keep as #1
2. **probability_distributions.py:log_pdf()** (THE hot path) — ELEVATED to #2
3. **_truncnorm.py** (required by #2, but can be partially deferred)
4. **parzen_estimator.py** (lowest impact in TPE)

**Rationale:** `log_pdf()` is the single function that determines TPE's
per-iteration speed. Accelerating it gives the biggest user-visible improvement.
`_truncnorm.py` is a dependency but only the functions called by `log_pdf()`
need to be migrated first (logpdf, _log_gauss_mass). Newton iteration in
`_ndtri_exp` can wait since it's used by `rvs()` (sampling), not `log_pdf()`.

### Cross-Phase: No changes

The phase order (GP -> TPE -> Hypervolume -> Multi-obj -> NSGAII) remains correct.

---

## New Phase 0 Addition: Risk Gate

**INSERT before Phase 1:**

### Phase 0.5: Autograd Risk Gate (NEW)

Before any Phase 1 migration work, dev must:

1. Create a minimal prototype demonstrating `mx.value_and_grad` with a closure
   that mutates a captured object's attributes:
```python
import mlx.core as mx

class SimpleGP:
    def __init__(self):
        self.param = mx.array(1.0)

    def loss(self):
        return mx.sum(mx.square(self.param))

    def fit(self):
        def loss_fn(raw_param):
            self.param = mx.exp(raw_param)
            return self.loss()

        val, grad = mx.value_and_grad(loss_fn)(mx.array(0.5))
        mx.eval(val, grad)
        print(f"val={val.item()}, grad={grad.item()}")

gp = SimpleGP()
gp.fit()
```

2. If this works: proceed with Phase 1 as planned.
3. If this fails: escalate to PO. Alternative approach: refactor to pass all
   params as a flat array to a pure loss function.

**This is a go/no-go gate for the entire Phase 1 approach.**

---

## Summary of Priority Changes

| Item | Original Priority | New Priority | Reason |
|------|------------------|--------------|--------|
| prior.py | Not listed | Phase 1, first | Missing from plan |
| autograd prototype | Phase 1, implicit | Phase 0.5, explicit risk gate | Highest risk item |
| gp.py Step 8 | Phase 1, step 8/10 | Phase 1, step 2 (after prototype) | Risk-first |
| log_pdf() in TPE | Phase 2, step 3/4 | Phase 2, step 2 | Hot path priority |
| _ndtri_exp Newton | Phase 2, step 2 | Phase 2, step 3 (partial defer ok) | Not on hot path |
| Everything else | Unchanged | Unchanged | Plan is solid |

---

## PO Decision: MINOR REPRIORITIZATION

The original phase order is correct. Changes are within-phase only:
- Add prior.py to Phase 1
- Add autograd risk gate before Phase 1
- Reorder Phase 1 sub-tasks for risk-first
- Reorder Phase 2 sub-tasks for impact-first

**No architectural changes. No scope changes. No new phases.**
