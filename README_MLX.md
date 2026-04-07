# ============================================================================
# OPTUNA-MLX: Apple Silicon GPU-Accelerated Hyperparameter Optimization
# ============================================================================
# "One does not simply optimize hyperparameters on CPU when you have a GPU."
#  - Sheldon Cooper, if he were a machine learning engineer
# ============================================================================
# Created: 2026-04-05
# Last Updated: 2026-04-05
# ============================================================================

# Optuna-MLX

**Bringing Apple Silicon GPU power to hyperparameter optimization.**

> "People say I'm hard to work with. Those people are wrong.
> I'm impossible to work with. But this framework is not."
> -- Sheldon Cooper

---

## What Is This?

Optuna-MLX is a fork of [Optuna](https://github.com/optuna/optuna) that
replaces CPU-bound numerical computations with GPU-accelerated operations
using Apple's [MLX](https://github.com/ml-explore/mlx) framework.

If you have a Mac with Apple Silicon (M1/M2/M3/M4), your hyperparameter
optimization just got a whole lot faster. Bazinga!

## Why MLX?

| Feature | NumPy (CPU) | PyTorch (MPS) | MLX (GPU) |
|---------|------------|---------------|-----------|
| Apple Silicon Native | No | Partial | Yes |
| Unified Memory | No | Partial | Yes |
| Lazy Evaluation | No | No | Yes |
| NumPy-like API | Yes | Similar | Yes |
| Zero-copy GPU | No | No | Yes |

MLX was designed from the ground up for Apple Silicon. It uses unified
memory (no CPU-GPU transfer overhead), supports lazy evaluation (compute
only what you need), and has a familiar NumPy-like API.

As Howard would say: "It's not rocket science... actually, it kind of is."

## Architecture

```
optuna/
  _mlx/              <-- PLANNED (not yet created)
    __init__.py       Backend detection and fallback logic  [PLANNED]
    (additional modules will be added as needed during Phase 1-2)
  _gp/                <-- PHASE 1: First module to be accelerated
    gp.py             Kernel matrices, Cholesky (torch -> mlx)
    acqf.py           Acquisition functions (torch -> mlx)
    optim_mixed.py    Optimization (torch -> mlx)
  samplers/
    _tpe/             <-- PHASE 2: TPE Sampler acceleration
    nsgaii/           <-- PHASE 5: NSGAII acceleration
  _hypervolume/       <-- PHASE 3: Hypervolume acceleration
  study/
    _multi_objective.py  <-- PHASE 4: Pareto front acceleration
```

## Acceleration Targets

### Phase 1: Gaussian Process Module (Highest Impact)
The GP module already uses PyTorch tensors, making it the most natural
MLX migration target. **~1308 LOC across 4 files.**

| File | LOC | Key Operations | MLX Strategy |
|------|-----|----------------|--------------|
| `gp.py` | 409 | Matern 5/2 kernel, Cholesky, posterior, MLL | Replace torch tensors with mx.array |
| `acqf.py` | 402 | LogEI, UCB, LCB, EHVI, constraint handling | Replace torch.special + autograd |
| `optim_mixed.py` | 329 | Batched gradient ascent, discrete search | MLX grads -> scipy L-BFGS-B |
| `batched_lbfgsb.py` | 168 | Greenlet-batched L-BFGS-B | Keep as-is (scipy wrapper) |

Key challenge: PyTorch's OOP autograd (`loss.backward()`) must become
MLX's functional API (`mx.value_and_grad(loss_fn)(params)`).

### Phase 2: TPE Sampler (Most Frequently Used)
The default sampler in Optuna. **~912 LOC across 4 files.**

| File | LOC | Key Operations | MLX Strategy |
|------|-----|----------------|--------------|
| `_erf.py` | 142 | Error function (polynomial approx) | Replace entirely with `mx.erf()` |
| `_truncnorm.py` | 296 | Truncated normal CDF/PDF/sampling | Vectorize with MLX ops |
| `probability_distributions.py` | 223 | Mixture sampling, log-PDF (hottest path) | MLX for core math |
| `parzen_estimator.py` | 251 | Bandwidth computation, distribution fitting | Selective MLX replacement |

Key win: `_erf.py` (142 LOC of polynomial wizardry) replaced by one `mx.erf()` call.
Key win: `_log_ndtr` uses Python-level loop via `np.frompyfunc` -> massive GPU speedup.

### Phase 3-5: Hypervolume, Multi-Objective, NSGAII
Supporting modules with significant vectorized computations. Recursive
algorithms (WFG hypervolume) stay on CPU; inner operations (dot products,
Pareto comparisons) move to GPU for large problem sizes.

## Development Approach

We follow a **harness approach** with defined roles:
- **Product Owner (PO):** Reviews feature completeness and priorities
- **QA:** Validates numerical accuracy and performance
- **Dev (Claude + Ramos):** Implementation and testing

> "Our development process is like the roommate agreement:
> thorough, well-documented, and occasionally fun." - Sheldon

### Priority: Functional Parity First
1. Match existing results exactly with MLX backend
2. Verify numerical accuracy with comprehensive tests
3. Then optimize for GPU performance

### Branch Strategy
- `master` - Stable reference (upstream Optuna)
- `dev` - Active development branch
- Feature branches as needed for large changes

## Getting Started (Coming Soon)

```bash
# Clone the fork
git clone https://github.com/RR-AMATOK/optuna-mlx.git
cd optuna-mlx
git checkout dev

# Install with MLX support
pip install -e ".[mlx]"

# Verify MLX GPU access
python -c "import mlx.core as mx; print(mx.default_device())"
```

## Project Files

| File | Purpose |
|------|---------|
| `README_MLX.md` | This file - project overview and roadmap |
| `NOTES.md` | Technical notes, analysis, session logs |
| `DECISIONS.md` | Architectural Decision Records (ADR-001 through ADR-012) |
| `TODO.md` | Task tracking across all phases |
| `CHANGELOG_MLX.md` | What changed and when |
| `DEV_BRIEF.md` | Dev instructions: what to build, exact file/line targets |
| `QA_ACCEPTANCE_CRITERIA.md` | QA test plan: tolerances, gates, sign-off criteria |
| `PO_ROADMAP_REVIEW.md` | PO audit: gaps, risks, plan alignment |
| `PO_REPRIORITIZATION.md` | PO priority adjustments based on code reality |
| `QA_REPORT.md` | QA findings: 3C, 4H, 7M, 5L, 6I issues found |
| `PO_QA_RESPONSE.md` | PO triage: dispositions for all 25 QA findings |
| `PO_PHASE1_REVIEW.md` | PO review of Phase 1 GP migration (conditional pass) |

## Requirements

- macOS with Apple Silicon (M1, M2, M3, M4, or later)
- Python >= 3.9
- MLX >= 0.5.0
- All standard Optuna dependencies

## Team

- **Ramos** - Project Lead
- **Claude (Dev)** - Development Partner (AI-powered, GPU-curious)
- **Claude (PO)** - Product Owner (briefs dev, gates QA, maintains roadmap)
- **QA** - Quality Assurance (TBD)

> "I'm not insane. My mother had me tested."
> And we'll test this code too. Extensively. - Sheldon

---

*This project is not affiliated with Caltech, but we do share their
commitment to scientific rigor. And like Raj, we promise this code
will work even when talking to women... or GPUs.*
