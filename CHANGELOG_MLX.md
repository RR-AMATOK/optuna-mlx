# ============================================================================
# OPTUNA-MLX CHANGELOG
# "Everything is better with a changelog. Except for a lack of a changelog."
#  - Leonard Hofstadter, on documentation
# ============================================================================
# Created: 2026-04-05
# Last Updated: 2026-04-05
# Project: Optuna-MLX (Apple Silicon GPU Acceleration)
# ============================================================================

## [Unreleased] - The Season Premiere

### Phase 1 Complete (2026-04-05/06) — GP Module Migrated to MLX
- All PyTorch operations in `optuna/_gp/` replaced with MLX equivalents
- Custom VJP for Matern52 kernel (handles sqrt(0) singularity)
- Custom VJP for marginal log likelihood (Rasmussen & Williams eq. 5.9 analytical gradient)
- `_solve_triangular_right()` helper for missing MLX `left=False` parameter
- `_erfcx()` with asymptotic expansion (threshold lowered from 4.0 to 3.0 for better
  coverage of `_log_ndtr` transition zone)
- `_log_ndtr()` with tail-safe branch for x < -5 using erfcx-based computation
  (fixes log(0) for large negative inputs, accurate to 1e-5 at x=-5)
- Sobol QMC via `scipy.stats.qmc.Sobol` (replaces `torch.quasirandom.SobolEngine`)
- Mixed precision: GPU float32 for element-wise ops, CPU float64 for linalg (ADR-009)
- `mx.stream(mx.cpu)` context managers for precision-critical paths
- Batched diagonal clamping via eye broadcasting (fixes `mx.diag` 3D+ limitation)
- 61 tests pass, 0 regressions
- PO review: CONDITIONAL PASS → fixes applied (MF-1, MF-2, SF-1)
- PO_PHASE1_REVIEW.md: detailed code review and sign-off assessment

### PO Review Fixes (2026-04-06)
- MF-1: GPSampler import check now shows "(Apple Silicon only)" in error message
- MF-2: ADR-011 resolved — Option B (hard fork, MLX-only). Non-macOS users use upstream.
- SF-1: `_log_ndtr` tail accuracy fixed (was returning -inf for x < -6, now 1e-5 to 1e-8)

### QA Bug Fixes (2026-04-06)
- B-1: `fit_kernel_params` now catches `np.linalg.LinAlgError` (was only `RuntimeError`).
  Prevents study crash from non-positive-definite kernel matrices.
- B-2: `standard_logei` erfcx-stable branch extended from z < -25 to z < -4.5.
  Both branches clamped to prevent NaN gradient propagation through `mx.where`.
- B-3: `_erfcx` asymptotic threshold set to x > 3.5 (rel_err < 1e-5 at boundary).
  Direct branch clamped to x <= 3.5 to prevent NaN from `exp(x^2) * (1-erf(x))`
  when MLX erf float32 precision makes `1-erf(x)` collapse to zero.

### QA Phase 1 Verification (2026-04-06) — CONDITIONAL FAIL
- Q-1 PASS: 270/270 parametrized GP tests pass
- Q-9 PASS: Numerical parity confirmed (kernel, posterior, MLL)
- Q-6 PASS: Greenlet + MLX interaction works correctly
- B-1 CRITICAL: LinAlgError escapes exception handler -> study crash at ~38-70 trials
- B-2 CRITICAL: LogEI gradient catastrophic cancellation for z in (-25, -5)
- B-3 CRITICAL: erfcx 2.66% error at transition x=3.01 (threshold too low)
- B-4 HIGH: _log_ndtr 3.97% error for positive x (cancellation in 1+erf)
- B-5 HIGH: Memory growth 6.6 MB/trial (464 MB in 70 trials)
- S-1 STRATEGIC: MLX 3.21x slower than torch E2E (GPU not used due to float64)
- PO decision: Fix B-1/B-2/B-3 (3 one-line changes), re-verify, then sign off
- ADR-013 needed: GP performance regression accepted, project viability rests on Phase 2

### Phase 0 Complete (2026-04-05)
- MLX 0.31.1 installed, float64 NOT GPU-accelerated (ADR-009 refined)
- `optuna/_mlx/__init__.py` created with backend detection
- `pyproject.toml` updated with MLX optional dependency
- Torch baseline benchmarks captured in `benchmarks/results_torch_baseline.json`

### Added
- Project documentation structure (NOTES.md, DECISIONS.md, TODO.md, CHANGELOG_MLX.md)
- README_MLX.md with project overview and roadmap
- Dev branch for feature development
- DEV_BRIEF.md: Precise dev instructions for Phase 0 remaining + Phase 1
- QA_ACCEPTANCE_CRITERIA.md: Testable acceptance criteria for all 6 phases
- PO_ROADMAP_REVIEW.md: Audit of plan vs codebase reality (7 gaps, 3 risks found)
- PO_REPRIORITIZATION.md: Within-phase priority adjustments
- ADR-011: Backend fallback strategy — ACCEPTED, Option B (hard fork, MLX-only)
- ADR-012: Autograd pattern for _fit_kernel_params (OPEN - needs prototype)
- Phase 0.5: Autograd risk gate added to TODO.md (blocking Phase 1)
- prior.py added to Phase 1 scope (was missing from original plan)
- PO_QA_RESPONSE.md: PO triage of 25 QA findings with dispositions
- Pre-Phase "Fix Now" section added to TODO.md (8 items from QA report)
- dtype audit added as Phase 1 gate (from QA finding M-7)
- Deferred QA fixes added to Phase 2 (H-2, H-4, M-1, M-2)
- Phases 3-5 marked CONDITIONAL based on QA perf findings (I-5, I-6)

### Analysis
- Completed full Optuna codebase analysis (37,726 LOC)
- Identified 5 major GPU acceleration targets with line-level detail
- Documented 10 architectural decisions (ADR-001 through ADR-010)
- Mapped every torch/numpy function to MLX equivalent across all target modules
- Created torch -> MLX and numpy -> MLX translation maps
- Identified MLX API gaps: erfc, erfcx, log_ndtr, SobolEngine, np.unique
- Identified critical risk areas: float64 performance, in-place ops, autograd restructuring

### Key Technical Findings
- GP module (`_gp/`): ~1308 LOC, 4 files, full PyTorch dependency -> direct MLX swap
- TPE module (`samplers/_tpe/`): ~912 LOC, 4 files, heavy NumPy -> selective MLX replacement
- `_erf.py` (142 LOC): Entirely replaceable by `mx.erf()` (one function call!)
- `_log_ndtr` uses `np.frompyfunc` (Python loop) - huge speedup opportunity with MLX vectorization
- Matern52Kernel uses `torch.autograd.Function` -> must restructure to pure function for `mx.grad()`
- scipy L-BFGS-B stays (ADR-008) - feed MLX gradients via numpy conversion

### Infrastructure
- Set up fork at github.com/RR-AMATOK/optuna-mlx
- Created dev branch from master
