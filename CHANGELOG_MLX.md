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

### Added
- Project documentation structure (NOTES.md, DECISIONS.md, TODO.md, CHANGELOG_MLX.md)
- README_MLX.md with project overview and roadmap
- Dev branch for feature development
- DEV_BRIEF.md: Precise dev instructions for Phase 0 remaining + Phase 1
- QA_ACCEPTANCE_CRITERIA.md: Testable acceptance criteria for all 6 phases
- PO_ROADMAP_REVIEW.md: Audit of plan vs codebase reality (7 gaps, 3 risks found)
- PO_REPRIORITIZATION.md: Within-phase priority adjustments
- ADR-011: Backend fallback strategy (OPEN - needs team decision)
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
