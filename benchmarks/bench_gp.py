# ============================================================================
# Optuna-MLX GP Module Benchmark Suite
# "Benchmarking: because 'it feels faster' is not a metric." - Sheldon
# ============================================================================
# Usage: python benchmarks/bench_gp.py [--backend torch|mlx] [--output FILE]
# ============================================================================
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np


def _make_data(
    n_trials: int, n_params: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.rand(n_trials, n_params)
    Y = rng.randn(n_trials)
    is_categorical = np.zeros(n_params, dtype=bool)
    return X, Y, is_categorical


def _timeit(fn, n_warmup: int = 3, n_runs: int = 30) -> dict:
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "median_ms": round(np.median(times) * 1000, 3),
        "mean_ms": round(np.mean(times) * 1000, 3),
        "std_ms": round(np.std(times) * 1000, 3),
        "min_ms": round(np.min(times) * 1000, 3),
        "n_runs": n_runs,
    }


def bench_torch() -> dict:
    import torch

    from optuna._gp.gp import GPRegressor, fit_kernel_params
    from optuna._gp.prior import default_log_prior
    from optuna._gp.acqf import LogEI
    from optuna._gp.search_space import SearchSpace
    from optuna._gp.optim_mixed import optimize_acqf_mixed

    results: dict = {"backend": "torch", "torch_version": torch.__version__}

    # --- 1. Kernel matrix computation ---
    results["kernel"] = {}
    for n_trials, n_params in [(10, 2), (50, 10), (100, 10), (500, 10)]:
        X, Y, is_cat = _make_data(n_trials, n_params)
        gpr = GPRegressor(
            is_categorical=torch.from_numpy(is_cat),
            X_train=torch.from_numpy(X),
            y_train=torch.from_numpy(Y),
            inverse_squared_lengthscales=torch.ones(n_params, dtype=torch.float64),
            kernel_scale=torch.tensor(1.0, dtype=torch.float64),
            noise_var=torch.tensor(0.1, dtype=torch.float64),
        )

        def run_kernel():
            with torch.no_grad():
                K = gpr.kernel()
            return K

        results["kernel"][f"{n_trials}x{n_params}"] = _timeit(run_kernel)

    # --- 2. Cache matrix (Cholesky + solve) ---
    results["cache_matrix"] = {}
    for n_trials in [10, 50, 100]:
        X, Y, is_cat = _make_data(n_trials, 10)

        def run_cache():
            g = GPRegressor(
                is_categorical=torch.from_numpy(is_cat),
                X_train=torch.from_numpy(X),
                y_train=torch.from_numpy(Y),
                inverse_squared_lengthscales=torch.ones(10, dtype=torch.float64),
                kernel_scale=torch.tensor(1.0, dtype=torch.float64),
                noise_var=torch.tensor(0.1, dtype=torch.float64),
            )
            g._cache_matrix()

        results["cache_matrix"][f"{n_trials}"] = _timeit(run_cache, n_warmup=2, n_runs=20)

    # --- 3. Marginal log likelihood ---
    results["mll"] = {}
    for n_trials in [10, 50, 100]:
        X, Y, is_cat = _make_data(n_trials, 10)
        gpr = GPRegressor(
            is_categorical=torch.from_numpy(is_cat),
            X_train=torch.from_numpy(X),
            y_train=torch.from_numpy(Y),
            inverse_squared_lengthscales=torch.ones(10, dtype=torch.float64),
            kernel_scale=torch.tensor(1.0, dtype=torch.float64),
            noise_var=torch.tensor(0.1, dtype=torch.float64),
        )

        def run_mll():
            return gpr.marginal_log_likelihood()

        results["mll"][f"{n_trials}"] = _timeit(run_mll)

    # --- 4. fit_kernel_params (full optimization) ---
    results["fit_kernel_params"] = {}
    for n_trials in [10, 50]:
        X, Y, is_cat = _make_data(n_trials, 5)

        def run_fit():
            fit_kernel_params(
                X, Y, is_cat,
                log_prior=default_log_prior,
                minimum_noise=1e-6,
                deterministic_objective=False,
            )

        results["fit_kernel_params"][f"{n_trials}"] = _timeit(run_fit, n_warmup=1, n_runs=5)

    # --- 5. End-to-end GPSampler optimize ---
    results["e2e_optimize"] = {}
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def run_e2e():
        study = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=42))
        study.optimize(
            lambda trial: (trial.suggest_float("x", -5, 5) - 2) ** 2,
            n_trials=30,
        )
        return study.best_value

    results["e2e_optimize"]["30_trials"] = _timeit(run_e2e, n_warmup=1, n_runs=3)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna GP Module Benchmarks")
    parser.add_argument("--backend", default="torch", choices=["torch", "mlx"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Running benchmarks with backend: {args.backend}")

    if args.backend == "torch":
        results = bench_torch()
    else:
        raise NotImplementedError("MLX benchmarks will be added after Phase 1")

    results["hardware"] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }

    output_file = args.output or f"benchmarks/results_{args.backend}_baseline.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
