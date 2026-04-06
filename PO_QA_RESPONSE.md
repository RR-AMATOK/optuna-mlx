# ============================================================================
# OPTUNA-MLX PO RESPONSE TO QA REPORT
# Product Owner Triage & Disposition
# ============================================================================
# Created: 2026-04-05
# Author: PO (Claude)
# In response to: QA_REPORT.md (2026-04-05)
# ============================================================================

# PO Triage of QA Findings

## Triage Legend

| Disposition | Meaning |
|-------------|---------|
| **FIX NOW** | Must fix before Phase 1 dev starts |
| **FIX IN PHASE** | Fix as part of the relevant phase |
| **ACCEPTED** | Valid finding, added to backlog |
| **DEFERRED** | Valid but low priority, revisit later |
| **ACKNOWLEDGED** | Noted, no action needed |
| **WONTFIX** | Not fixing (with rationale) |

---

## CRITICAL Issues

### C-1: No MLX or PyTorch installed — GP Sampler broken
**Disposition: FIX NOW**

QA is correct. This is the #1 blocker. Dev cannot work without the runtime.

**Actions:**
1. Dev: Run Work Item 0A from DEV_BRIEF.md (`pip install mlx torch`)
2. Dev: Verify `GPSampler` works end-to-end with torch (baseline)
3. Dev: Add early-fail check to `GPSampler.__init__` (QA's recommendation is good)

**Note on QA's recommendation to add a check at GPSampler.__init__:** Agree. However,
this is **upstream Optuna behavior**, not something we introduced. Since we're a fork,
we should fix it in our fork but not conflate it with the MLX migration work. Separate
commit.

**Assigned to:** Dev, before Phase 0.5

---

### C-2: dev branch has zero commits beyond master
**Disposition: FIX NOW**

QA is absolutely right. All documentation is untracked and vulnerable to loss.

**Actions:**
1. Dev: Stage and commit all documentation files to dev branch:
   - README_MLX.md, NOTES.md, DECISIONS.md, TODO.md, CHANGELOG_MLX.md
   - DEV_BRIEF.md, QA_ACCEPTANCE_CRITERIA.md, QA_REPORT.md
   - PO_ROADMAP_REVIEW.md, PO_REPRIORITIZATION.md, PO_QA_RESPONSE.md
2. Commit message: `docs: add Optuna-MLX project documentation and planning`

**Assigned to:** Dev (or Ramos), immediately

---

### C-3: pyproject.toml has no MLX dependency
**Disposition: FIX NOW**

Already in DEV_BRIEF.md as Work Item 0C. QA confirms urgency.

**Actions:** Follow Work Item 0C instructions exactly.

**Assigned to:** Dev, as part of Phase 0

---

## HIGH Issues

### H-1: LOC counts in documentation are all off by 1
**Disposition: ACCEPTED — fix in bulk**

QA is right. Every LOC count is +1. This is a systematic error (likely `wc -l`
counting trailing newline). Affects trust in documentation accuracy.

**Actions:**
1. Dev: Fix all LOC counts in NOTES.md, TODO.md, README_MLX.md, CHANGELOG_MLX.md
2. Use actual line counts: gp.py=409, acqf.py=402, optim_mixed.py=329,
   batched_lbfgsb.py=168, _erf.py=142, _truncnorm.py=296,
   probability_distributions.py=223, parzen_estimator.py=251
3. Fix aggregates: GP module ~1308 LOC (not 1300), TPE module ~912 LOC (not 916)

**Priority:** Do this in the same commit as C-2 (documentation commit).

---

### H-2: _ndtri_exp returns NaN for y=0.0
**Disposition: DEFERRED to Phase 2**

QA correctly identifies this as a latent bug. The current code explicitly acknowledges
it (lines 218-219) and relies on `ppf()` clipping to avoid it. During Phase 2 MLX
migration, if we use `_ndtri_exp` independently, this will bite.

**Actions:**
1. Add to TODO.md Phase 2B: "Handle y=0.0 edge case in _ndtri_exp (return inf)"
2. No fix now — upstream behavior, and fixing it prematurely could change TPE results
   before we have parity tests in place

**Risk if deferred:** Low. The function is only called via `ppf()` which clips.

---

### H-3: NaN values in Pareto front treated as non-dominated
**Disposition: DEFERRED to Phase 4**

This is upstream Optuna behavior, not something we introduced. QA's finding is valid
but fixing it would change multi-objective semantics before we have MLX parity.

**Actions:**
1. Add to TODO.md Phase 4: "Document or handle NaN in Pareto front computation"
2. For MLX migration: ensure MLX NaN comparison semantics match numpy's
   (NaN < x = False in both, so behavior should be preserved automatically)

**Risk if deferred:** Medium. Could produce incorrect Pareto fronts if trials
have NaN objectives. But this is an upstream issue, not our regression.

---

### H-4: rvs() accepts negative scale without validation
**Disposition: DEFERRED to Phase 2**

Upstream behavior. Valid bug but not our regression.

**Actions:**
1. Add to TODO.md Phase 2B: "Add scale > 0 validation to rvs()"
2. During Phase 2 MLX migration, add the validation as part of the rewrite

**Risk if deferred:** Low. Callers in Optuna always pass positive scale.

---

## MEDIUM Issues

### M-1: _log_gauss_mass(a, b) when a > b produces NaN silently
**Disposition: DEFERRED to Phase 2**

Same pattern as H-2/H-4 — upstream behavior, valid but not our regression.
Add to Phase 2 backlog.

---

### M-2: ppf() accepts q outside [0, 1] without validation
**Disposition: DEFERRED to Phase 2**

Same pattern. Add to Phase 2 backlog.

---

### M-3: _log_ndtr uses Python-level loop (perf bottleneck)
**Disposition: ACCEPTED — already planned**

QA confirms what NOTES.md already identified. This is the #1 TPE acceleration target.
Already in TODO.md Phase 2B. No new action needed — just confirmation that our
priority is correct.

---

### M-4: Architecture diagram shows files that don't exist
**Disposition: FIX NOW**

Misleading docs are worse than no docs. The README shows `array_ops.py`, `linalg.py`,
`random.py`, `grad.py` under `_mlx/` with "(coming soon)" — but a developer skimming
the tree will look for them and be confused.

**Actions:**
1. Dev: Update README_MLX.md architecture diagram to clearly mark planned vs existing
2. Change `(coming soon)` to `<-- PLANNED (not yet created)` or similar

**Priority:** Include in C-2 documentation commit.

---

### M-5: GP sampler error message is unhelpful
**Disposition: FIX NOW**

QA's recommendation is exactly right. A `ModuleNotFoundError` from deep in
`_LazyImport.__getattr__` is terrible UX.

**Actions:**
1. Dev: Add early import check in `GPSampler.__init__` or `_sample_relative`:
```python
try:
    import torch  # or mlx.core after migration
except ImportError:
    raise ImportError(
        "GPSampler requires PyTorch (or MLX on Apple Silicon). "
        "Install with: pip install optuna[optional]"
    ) from None
```
2. This aligns with C-1 and should be in the same commit.

**Note:** After Phase 1, this changes to check for MLX first, then torch fallback
(per ADR-011 decision). For now, just make the torch error message better.

---

### M-6: warn_and_convert_inf doesn't handle NaN
**Disposition: ACKNOWLEDGED — no action**

The code comment explicitly states Optuna won't pass NaN. This is a design contract,
not a bug. The upstream maintainers made this decision deliberately (see the
`NOTE(nabenabe)` comment at gp.py:54).

Adding a NaN check would be a behavior change that could mask other bugs upstream.
Leave it as-is. If the MLX migration somehow introduces NaN paths, we'll catch it
in parity testing (AC-1.11).

---

### M-7: Inconsistent torch.float64 usage across GP module
**Disposition: FIX IN PHASE 1 — critical migration concern**

This is the most actionable medium finding. QA correctly identifies that MLX defaults
to float32, so every implicit-dtype `torch.from_numpy()` -> `mx.array()` conversion
risks a silent precision downgrade.

**Actions:**
1. Dev: Before Phase 1 migration, create a dtype audit:
   - List every `torch.from_numpy()` call in _gp/
   - Note which ones have explicit `dtype=torch.float64` vs implicit
   - In the MLX migration, add explicit `dtype=mx.float64` to ALL array creations
2. Add to DEV_BRIEF.md Work Item 1B Step 0 (pre-step): dtype audit
3. QA: Add to acceptance criteria — `grep -r "mx.array(" optuna/_gp/` should show
   that every call includes `dtype=mx.float64` (or has a comment justifying float32)

**This is now a gate for Phase 1 PO sign-off.**

---

## LOW Issues

### L-1: "37,726 LOC" will become stale
**Disposition: WONTFIX**

It's a snapshot from analysis day. It's dated in context. Removing it loses
useful baseline info. Leave it.

---

### L-2: Documentation files not gitignored but not committed
**Disposition: FIX NOW (same as C-2)**

Already addressed by C-2 fix. Commit them.

---

### L-3: Sheldon Cooper quotes may confuse non-fans
**Disposition: ACKNOWLEDGED — no action**

Ramos established the tone intentionally. These are internal team docs, not
user-facing API docs. The quotes add personality without harming clarity.
Keep them.

---

### L-4: ADR-009 status ambiguity
**Disposition: ACCEPTED — clarify**

QA is right that "ACCEPTED (pending float64 benchmark)" is ambiguous.

**Actions:**
1. Dev: Change ADR-009 status to: `ACCEPTED — implementation approach will be
   refined by float64 benchmark results (Phase 0, Work Item 0A)`
2. When benchmark results come in, update with a follow-up note

---

### L-5: Non-standard TODO markers unused
**Disposition: ACCEPTED — clean up**

QA is right. `[~]`, `[!]`, `[?]` are defined but never used. They also won't
render as GitHub checkboxes.

**Actions:**
1. Dev: Remove the unused markers from the legend, OR
2. Start using them where appropriate (e.g., Phase 0.5 risk gate = `[?]`)

**Priority:** Low. Include in C-2 commit if convenient.

---

## INFO Items — PO Notes

### I-1: 9 in-place ops, 19 torch.from_numpy, 3 .backward()
**Acknowledged.** Numbers match our analysis. Good QA verification.

### I-2: _erf.py accuracy is excellent
**Acknowledged.** Confirms our plan to replace with `mx.erf()` is safe — we're
replacing accurate code with a built-in that should be equally or more accurate.

### I-3: Existing test suite is solid
**Acknowledged.** This is great news for Phase 1. We have good coverage to verify
parity against.

### I-4: Line number references 100% accurate
**Acknowledged.** Nice validation of the analysis quality.

### I-5: Hypervolume is already fast at typical scales
**IMPORTANT FINDING.** QA measured 3D hypervolume at <1ms for 200 points.

**PO Decision:** Phase 3 (Hypervolume) is now **conditional**. We will only proceed
with Phase 3 if:
- Phase 1 and 2 demonstrate significant speedups, AND
- Users report hypervolume as a bottleneck at scale (10K+ points)

This does NOT change the phase order but makes Phase 3 a "proceed if justified"
rather than "definitely do."

### I-6: Multi-objective + NSGAII already fast
**IMPORTANT FINDING.** Pareto computation for 1000x3 took 1ms.

**PO Decision:** Same as I-5. Phases 4-5 are now **conditional**. The ROI at
typical scales doesn't justify the engineering effort unless users need extreme
scale (10K+ trials with many objectives).

**Updated priority:**
- Phase 1 (GP): MUST DO — torch replacement, clear GPU acceleration win
- Phase 2 (TPE): MUST DO — numpy hot paths, biggest user-visible speedup
- Phase 3 (Hypervolume): DO IF JUSTIFIED — need evidence of bottleneck
- Phase 4 (Multi-obj): DO IF JUSTIFIED — already fast
- Phase 5 (NSGAII): DO IF JUSTIFIED — already fast
- Phase 6 (Optimization): MUST DO for Phases 1-2, conditional for 3-5

---

## Action Summary

### FIX NOW (before Phase 1 dev starts)

| # | Action | Owner | Source |
|---|--------|-------|--------|
| 1 | Install MLX + PyTorch | Dev | C-1 |
| 2 | Commit all docs to dev branch | Dev/Ramos | C-2, L-2 |
| 3 | Add MLX to pyproject.toml | Dev | C-3 |
| 4 | Fix all LOC counts (-1 each) | Dev | H-1 |
| 5 | Fix README architecture diagram | Dev | M-4 |
| 6 | Add GPSampler import check | Dev | M-5, C-1 |
| 7 | Clarify ADR-009 status wording | Dev | L-4 |
| 8 | Clean up TODO.md legend | Dev | L-5 |

### FIX IN PHASE 1

| # | Action | Owner | Source |
|---|--------|-------|--------|
| 9 | dtype audit of all torch.from_numpy calls | Dev | M-7 |
| 10 | Explicit mx.float64 on all array creations | Dev | M-7 |

### DEFERRED TO PHASE 2

| # | Action | Source |
|---|--------|--------|
| 11 | Handle y=0.0 in _ndtri_exp | H-2 |
| 12 | Add scale>0 validation to rvs() | H-4 |
| 13 | Add a>b validation to _log_gauss_mass | M-1 |
| 14 | Add q bounds check to ppf() | M-2 |

### DEFERRED TO PHASE 4

| # | Action | Source |
|---|--------|--------|
| 15 | NaN handling in Pareto front | H-3 |

### STRATEGIC DECISIONS

| # | Decision | Source |
|---|----------|--------|
| 16 | Phases 3-5 now conditional on demonstrated need | I-5, I-6 |

---

## Updated Project Velocity Gate

The project now has this gate sequence:

```
FIX NOW items (actions 1-8)
    │
    ▼
Phase 0 remaining (Work Items 0A-0D)
    │
    ▼
Phase 0.5 Autograd Risk Gate
    │
    ▼
Phase 1 (GP Migration) — includes dtype audit (action 9-10)
    │
    ▼
Phase 1 PO Sign-off
    │
    ▼
Phase 1 QA Sign-off
    │
    ▼
Phase 2 (TPE Migration) — includes deferred fixes (actions 11-14)
    │
    ▼
Phase 2 PO + QA Sign-off
    │
    ▼
Decision gate: Do Phases 3-5 proceed? (based on Phase 1-2 speedup data)
    │
    ▼
Phase 6 (Optimization) for completed phases
```

---

## Response to QA

QA: Excellent work. The report was thorough, well-structured, and surfaced
real issues. Specific highlights:

- **C-2 (uncommitted docs)** — Good catch. Obvious in hindsight but easy to miss.
- **H-1 (LOC counts)** — Systematic error detection. Shows rigor.
- **M-7 (dtype audit)** — The most impactful medium finding. This could have
  caused subtle precision bugs in Phase 1 if not caught.
- **I-5/I-6 (perf at scale)** — These findings changed the project strategy.
  Phases 3-5 are now conditional. That's significant ROI from QA effort.

One note: Several HIGH/MEDIUM findings (H-2, H-3, H-4, M-1, M-2) are upstream
Optuna behaviors, not issues we introduced. They're valid findings for the MLX
migration context (we need to preserve these behaviors during migration, or
deliberately fix them). But they shouldn't be counted as our bugs — they're
inherited technical debt that we may choose to fix while we're in the code.

**QA is cleared to proceed with Phase 0 acceptance testing once dev completes
the FIX NOW items.**
