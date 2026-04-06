# ============================================================================
# OPTUNA-MLX QA REPORT
# Date: 2026-04-05
# QA Engineer: Claude (QA Role)
# Scope: Full project state assessment, stress testing, documentation audit
# ============================================================================

## Executive Summary

The optuna-mlx project is currently in **Phase 0 (Planning/Setup)** with zero
code changes on the `dev` branch beyond upstream `master`. All work to date is
documentation and analysis (untracked files). The existing Optuna codebase
(upstream) is functional for TPE, multi-objective, and hypervolume workloads.
The GP sampler cannot be tested because **neither PyTorch nor MLX are installed**.

**Severity counts:**
- CRITICAL: 3
- HIGH: 4
- MEDIUM: 7
- LOW: 5
- INFO: 6

---

## CRITICAL Issues

### C-1: No MLX or PyTorch installed - GP Sampler completely broken
- **Severity:** CRITICAL
- **Component:** Environment / GP Sampler
- **Description:** The GP Sampler (`GPSampler`) crashes with
  `ModuleNotFoundError: No module named 'torch'` on the first trial that
  requires relative sampling (trial 10+). MLX is also not installed. The
  project's core value proposition (MLX acceleration of the GP module) cannot
  be developed or tested.
- **Steps to reproduce:**
  ```python
  import optuna
  study = optuna.create_study(sampler=optuna.samplers.GPSampler())
  study.optimize(lambda trial: trial.suggest_float('x', -10, 10)**2, n_trials=20)
  ```
- **Expected:** Clear error at sampler creation or graceful fallback
- **Actual:** Silently creates sampler, crashes at trial 10 with a deep
  traceback through `_LazyImport` -> `ModuleNotFoundError`
- **Recommendation:** Install torch and MLX. Add a check at `GPSampler.__init__`
  that validates torch (or future MLX) is available and raises a clear
  `ImportError` with install instructions.

### C-2: dev branch has zero commits beyond master
- **Severity:** CRITICAL
- **Component:** Git/Repository
- **Description:** `git log dev --not master` returns empty. All project
  documentation (README_MLX.md, NOTES.md, DECISIONS.md, TODO.md, CHANGELOG_MLX.md)
  exists only as untracked files. These files are not version-controlled and
  could be lost.
- **Recommendation:** Commit the documentation files to the dev branch
  immediately. This is foundational project state that must be preserved.

### C-3: pyproject.toml has no MLX dependency
- **Severity:** CRITICAL (for project goals)
- **Component:** Build/Dependencies
- **Description:** The TODO correctly marks this as incomplete, but without
  the `mlx` optional dependency in pyproject.toml, there's no way for users
  or CI to install the MLX backend. The current `[project.optional-dependencies]`
  sections reference `torch` and `scipy` but not `mlx`.
- **Recommendation:** Add `mlx = ["mlx>=0.5.0"]` to optional-dependencies
  before any MLX code is written.

---

## HIGH Issues

### H-1: LOC counts in documentation are all off by 1
- **Severity:** HIGH (accuracy of planning documents)
- **Component:** Documentation (NOTES.md, TODO.md, CHANGELOG_MLX.md)
- **Description:** Every single LOC count is overstated by exactly 1 line:
  | File | Claimed | Actual |
  |------|---------|--------|
  | gp.py | 410 | 409 |
  | acqf.py | 403 | 402 |
  | optim_mixed.py | 330 | 329 |
  | batched_lbfgsb.py | 169 | 168 |
  | _erf.py | 143 | 142 |
  | _truncnorm.py | 297 | 296 |
  | probability_distributions.py | 224 | 223 |
  | parzen_estimator.py | 252 | 251 |
- **Root cause:** Likely counted with `wc -l` which includes a trailing
  newline, or 1-indexed counting error.
- **Impact:** These counts cascade into the "1300 LOC" and "916 LOC" aggregate
  claims in README_MLX.md and CHANGELOG_MLX.md. Actual GP module is ~1308 LOC,
  not 1300. Actual TPE module is ~912 LOC, not 916.
- **Recommendation:** Fix all LOC counts. Use consistent counting method.

### H-2: _ndtri_exp returns NaN for y=0.0 (valid input)
- **Severity:** HIGH
- **Component:** `optuna/samplers/_tpe/_truncnorm.py:151`
- **Description:** `_ndtri_exp(np.array([0.0]))` returns `[nan]`.
  The function is documented as computing the inverse of `log_ndtr`, and
  `log_ndtr(inf) = 0.0`, so `_ndtri_exp(0.0)` should return `inf`.
  The code has a comment on line 218-219 acknowledging this:
  `"x[y == 0.0] = np.inf ... are necessary for the accurate computation,
  but we omit them as the ppf applies clipping"`.
  This is technically correct for the current call path but is a latent bug
  that will bite during MLX migration if `_ndtri_exp` is used independently.
- **Recommendation:** Document this limitation prominently or add the inf
  handling.

### H-3: NaN values in Pareto front treated as non-dominated
- **Severity:** HIGH
- **Component:** `optuna/study/_multi_objective.py`
- **Description:** `_is_pareto_front(np.array([[1.0, 2.0], [np.nan, 1.0], [0.5, 3.0]]))` returns
  `[True, True, True]` - the NaN point is considered Pareto-optimal. This
  is because NaN comparisons return False in numpy, so a NaN solution can
  never be dominated. This could pollute Pareto fronts with invalid trials.
- **Recommendation:** Filter or handle NaN values before Pareto computation,
  or document the behavior explicitly.

### H-4: rvs() accepts negative scale without validation
- **Severity:** HIGH
- **Component:** `optuna/samplers/_tpe/_truncnorm.py:268`
- **Description:** `rvs(a, b, loc=0, scale=-1.0)` produces results without
  error. Negative scale is mathematically meaningless for a truncated normal.
  The output is `-ppf(q, a, b)` which flips the distribution, potentially
  producing values outside the expected bounds.
- **Recommendation:** Add `assert scale > 0` or validation.

---

## MEDIUM Issues

### M-1: _log_gauss_mass(a, b) when a > b produces NaN silently
- **Severity:** MEDIUM
- **Component:** `optuna/samplers/_tpe/_truncnorm.py:112`
- **Description:** Invalid interval (a > b) returns NaN without warning.
  This could mask bugs in callers.
- **Recommendation:** Add assertion or warning for a > b.

### M-2: ppf() accepts q outside [0, 1] without validation
- **Severity:** MEDIUM
- **Component:** `optuna/samplers/_tpe/_truncnorm.py:223`
- **Description:** `ppf(np.array([-0.1, 1.1, 2.0]), ...)` returns `[nan, nan, nan]`
  silently. No validation on the quantile range.
- **Recommendation:** Add bounds check or document NaN behavior.

### M-3: _log_ndtr uses Python-level loop (perf bottleneck for MLX migration)
- **Severity:** MEDIUM (perf impact for Phase 2)
- **Component:** `optuna/samplers/_tpe/_truncnorm.py:104`
- **Description:** `_log_ndtr` uses `np.frompyfunc(_log_ndtr_single, 1, 1)` which
  is a Python-level loop over scalar `_log_ndtr_single` calls. Each call also
  has `@functools.lru_cache(1000)`. For the MLX migration, this pattern is
  fundamentally incompatible with GPU vectorization.
- **Performance test:** 10,000 elements took 0.004s (fast due to caching), but
  cache misses on unique values will be much slower.
- **Recommendation:** This is already noted in NOTES.md as a "prime GPU target".
  Prioritize vectorized MLX implementation.

### M-4: Architecture diagram in README shows files that don't exist
- **Severity:** MEDIUM
- **Component:** README_MLX.md
- **Description:** The Architecture section shows `optuna/_mlx/` with 5 files
  (\_\_init\_\_.py, array_ops.py, linalg.py, random.py, grad.py). These are
  labeled "(coming soon)" but could confuse new contributors looking for them.
- **Recommendation:** Add a clear note that these are planned, not implemented.

### M-5: GP sampler error message is unhelpful
- **Severity:** MEDIUM
- **Component:** `optuna/samplers/_gp/sampler.py`
- **Description:** When torch is missing, the error comes from deep in the
  stack (`_LazyImport.__getattr__`), not from the sampler. A user seeing
  `ModuleNotFoundError` at `torch.device("cpu")` won't know they need to
  install torch as an optional dependency.
- **Recommendation:** Add try/except in `GPSampler.__init__` with:
  `raise ImportError("GPSampler requires PyTorch. Install: pip install optuna[optional]")`

### M-6: warn_and_convert_inf doesn't handle NaN despite comment
- **Severity:** MEDIUM
- **Component:** `optuna/_gp/gp.py:47`
- **Description:** The comment on line 55 says "values cannot include nan...
  Optuna anyways won't pass nan in values by design." This is an assumption
  that could break if the MLX migration changes calling patterns or if
  upstream Optuna changes.
- **Recommendation:** Add explicit NaN check or assertion.

### M-7: Inconsistent torch.float64 usage across GP module
- **Severity:** MEDIUM (MLX migration concern)
- **Component:** `optuna/_gp/gp.py`, `optuna/_gp/acqf.py`
- **Description:** Some tensors are explicitly typed `dtype=torch.float64`
  (gp.py:112, 280, 318, 364; acqf.py:59) while others rely on implicit
  dtype from `torch.from_numpy()` (which preserves numpy's float64 default).
  During MLX migration, MLX defaults to float32, so every implicit-dtype
  path will silently downgrade precision unless caught.
- **Recommendation:** Audit and document every dtype assumption. Create a
  checklist of all `torch.from_numpy()` calls that need explicit
  `dtype=mx.float64`.

---

## LOW Issues

### L-1: CHANGELOG_MLX.md references "37,726 LOC" - coincidentally exact
- **Severity:** LOW
- **Description:** The LOC count is exactly correct (verified). However, it
  will become stale as code is modified. Consider removing or dating it.

### L-2: Documentation files are not gitignored but also not committed
- **Severity:** LOW
- **Component:** Git repository
- **Description:** Five documentation files show as untracked in `git status`.
  They're neither committed (tracked) nor ignored. This ambiguous state means
  they'll show up in every `git status` command.
- **Recommendation:** Commit them to dev branch.

### L-3: Sheldon Cooper quotes are entertaining but may confuse non-fans
- **Severity:** LOW
- **Component:** All documentation files
- **Description:** Pervasive Big Bang Theory references. Professional but
  unusual. Fine for internal team docs.

### L-4: DECISIONS.md ADR-009 status is "ACCEPTED (pending float64 benchmark)"
- **Severity:** LOW
- **Component:** DECISIONS.md
- **Description:** The parenthetical creates ambiguity about whether this
  decision is truly accepted or conditional. If the benchmark shows float64
  is slow, does the decision reverse?
- **Recommendation:** Add a follow-up ADR when benchmark results are available.

### L-5: TODO.md uses non-standard markers [~] and [!] and [?]
- **Severity:** LOW
- **Component:** TODO.md
- **Description:** The legend defines `[~] In Progress`, `[!] Blocked`,
  `[?] Needs Decision` but none are currently used. The only markers used
  are `[x]` and `[ ]`. These non-standard markers won't render as checkboxes
  in GitHub's markdown.
- **Recommendation:** Either use them or remove the legend entries.

---

## INFO / Observations

### I-1: GP module has 9 in-place operations requiring refactoring
All `.square_()`, `.clamp_min_()`, `.clamp_()`, `.requires_grad_()` calls
must become functional-style for MLX. 19 `torch.from_numpy()` calls become
`mx.array()`. 3 `.backward()` calls need `mx.value_and_grad()`. Documented
in NOTES.md - verified accurate.

### I-2: _erf.py accuracy is excellent
The vectorized path (>=2000 elements) achieves 2.22e-16 max error vs
`math.erf()`. The scalar path (<2000 elements) is exact. Both paths handle
NaN, Inf, and extreme values correctly.

### I-3: Existing test suite structure is solid
Tests exist for GP module (4 files), TPE (separate directory), hypervolume,
multi-objective, and NSGAII. These provide good baseline coverage for
verifying MLX parity.

### I-4: Line number references in NOTES.md are 100% accurate
All 22 line number references checked against actual source code were correct.
Impressive attention to detail in the analysis phase.

### I-5: Hypervolume computation is fast even at scale
3D hypervolume with 200 points computed in <1ms. The WFG algorithm's
recursive structure may not benefit much from GPU acceleration at typical
scales (contradicts the effort planned for Phase 3).

### I-6: Multi-objective + NSGAII work correctly
All multi-objective tests passed. NSGAII with mixed parameters works. Pareto
front computation for 1000x3 took only 1ms. Phase 4-5 GPU acceleration may
have limited ROI at typical scales.

---

## Recommendations for Dev Team

### Immediate (Before Writing Any MLX Code)
1. **Commit documentation to dev branch** (C-2)
2. **Install MLX and PyTorch** in development environment (C-1)
3. **Add MLX to pyproject.toml** optional dependencies (C-3)
4. **Fix LOC counts** in all documentation (H-1)
5. **Run existing test suite** with torch installed to establish baseline

### Before Phase 1 (GP Migration)
6. Create a comprehensive dtype audit of all 19 `torch.from_numpy()` calls (M-7)
7. Write baseline benchmark numbers with torch for comparison
8. Add graceful import error for GPSampler when backend missing (M-5)

### Architecture Feedback
9. Consider whether Phases 3-5 are worth the effort - hypervolume and Pareto
   are already fast at typical scales (I-5, I-6). ROI may be low unless users
   commonly run with 10K+ trials.
10. The `_ndtri_exp` Newton iteration (up to 100 iterations with per-element
    convergence) is the hardest GPU porting challenge in Phase 2. Consider
    fixed iteration count with masked updates as noted in TODO.

---

## Test Environment
- **Platform:** macOS Darwin 25.3.0, Apple Silicon
- **Python:** 3.13.2 (Anaconda)
- **NumPy:** 2.4.3
- **SciPy:** 1.17.1
- **Optuna:** 4.9.0.dev (editable install from dev branch)
- **PyTorch:** NOT INSTALLED
- **MLX:** NOT INSTALLED

---

*QA Report generated 2026-04-05. All findings verified by automated test scripts.*
