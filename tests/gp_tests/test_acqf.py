from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from optuna._gp import acqf as acqf_module
from optuna._gp.gp import GPRegressor
from optuna._gp.search_space import SearchSpace
from optuna.distributions import FloatDistribution


def verify_eval_acqf(x: np.ndarray, acqf: acqf_module.BaseAcquisitionFunc) -> None:
    with mx.stream(mx.cpu):
        x_mx = mx.array(x, dtype=mx.float64)

        def acqf_sum(x_in: mx.array) -> mx.array:
            return mx.sum(acqf.eval_acqf(x_in))

        acqf_value_sum, acqf_grad = mx.value_and_grad(acqf_sum)(x_mx)
        acqf_value = acqf.eval_acqf(x_mx)
        mx.eval(acqf_value, acqf_grad)

        assert acqf_value.shape == x.shape[:-1]
        assert bool(mx.all(mx.isfinite(acqf_value)))
        assert bool(mx.all(mx.isfinite(acqf_grad)))


def get_gpr(y_train: np.ndarray) -> GPRegressor:
    with mx.stream(mx.cpu):
        gpr = GPRegressor(
            is_categorical=mx.array([False, False]),
            X_train=mx.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.1]], dtype=mx.float64),
            y_train=mx.array(y_train, dtype=mx.float64),
            inverse_squared_lengthscales=mx.array([2.0, 3.0], dtype=mx.float64),
            kernel_scale=mx.array(4.0, dtype=mx.float64),
            noise_var=mx.array(0.1, dtype=mx.float64),
        )
        gpr._cache_matrix()
        return gpr


@pytest.fixture
def search_space() -> SearchSpace:
    n_dims = 2
    return SearchSpace({chr(ord("a") + i): FloatDistribution(0.0, 1.0) for i in range(n_dims)})


parametrized_x = pytest.mark.parametrize(
    "x",
    [np.array([0.15, 0.12]), np.array([[0.15, 0.12], [0.0, 1.0]])],  # unbatched  # batched
)

parametrized_additional_values = pytest.mark.parametrize(
    "additional_values",
    [
        np.array([[0.2], [0.3], [-0.4]]),
        np.array([[0.2, 0.3, 0.4], [0.2, 0.3, 0.4], [0.2, 0.3, 0.4]]),
        np.array([[0.2, 0.3, 0.4], [0.2, 0.3, -0.4], [-0.2, -0.3, -0.4]]),
        np.array([[-0.2, -0.3, -0.4], [-0.2, -0.3, -0.4], [-0.2, -0.3, -0.4]]),
    ],
)


@pytest.mark.parametrize(
    "acqf_cls", [acqf_module.LogEI, acqf_module.LCB, acqf_module.UCB, acqf_module.LogPI]
)
@parametrized_x
def test_eval_acqf(
    acqf_cls: type[acqf_module.BaseAcquisitionFunc],
    x: np.ndarray,
    search_space: SearchSpace,
) -> None:
    Y = np.array([1.0, 2.0, 3.0])
    kwargs = dict(gpr=get_gpr(Y), search_space=search_space)
    if acqf_cls in [acqf_module.LCB, acqf_module.UCB]:
        kwargs.update(beta=2.0)
    else:
        kwargs.update(threshold=np.max(Y))

    verify_eval_acqf(x, acqf_cls(**kwargs))  # type: ignore[arg-type]


@parametrized_x
@parametrized_additional_values
def test_eval_acqf_with_constraints(
    x: np.ndarray,
    additional_values: np.ndarray,
    search_space: SearchSpace,
) -> None:
    c = additional_values.copy()
    Y = np.array([1.0, 2.0, 3.0])
    is_feasible = np.all(c <= 0, axis=1)
    is_all_infeasible = not np.any(is_feasible)
    acqf = acqf_module.ConstrainedLogEI(
        gpr=get_gpr(Y),
        search_space=search_space,
        threshold=-np.inf if is_all_infeasible else np.max(Y[is_feasible]),
        stabilizing_noise=0.0,
        constraints_gpr_list=[get_gpr(vals) for vals in c.T],
        constraints_threshold_list=[0.0] * len(c.T),
    )
    verify_eval_acqf(x, acqf)


@parametrized_x
@parametrized_additional_values
def test_eval_multi_objective_acqf(
    x: np.ndarray,
    additional_values: np.ndarray,
    search_space: SearchSpace,
) -> None:
    with mx.stream(mx.cpu):
        Y = np.hstack([np.array([1.0, 2.0, 3.0])[:, np.newaxis], additional_values])
        n_objectives = Y.shape[-1]
        acqf = acqf_module.LogEHVI(
            gpr_list=[get_gpr(Y[:, i]) for i in range(n_objectives)],
            search_space=search_space,
            Y_train=mx.array(Y, dtype=mx.float64),
            n_qmc_samples=32,
            qmc_seed=42,
            stabilizing_noise=0.0,
        )
        verify_eval_acqf(x, acqf)


@parametrized_x
@parametrized_additional_values
def test_eval_multi_objective_acqf_with_constraints(
    x: np.ndarray,
    additional_values: np.ndarray,
    search_space: SearchSpace,
) -> None:
    with mx.stream(mx.cpu):
        c = additional_values.copy()
        Y = np.hstack([np.array([1.0, 2.0, 3.0])[:, np.newaxis], additional_values])
        n_objectives = Y.shape[-1]
        is_feasible = np.all(c <= 0, axis=1)
        is_all_infeasible = not np.any(is_feasible)
        acqf = acqf_module.ConstrainedLogEHVI(
            gpr_list=[get_gpr(Y[:, i]) for i in range(n_objectives)],
            search_space=search_space,
            Y_feasible=None if is_all_infeasible else mx.array(Y[is_feasible], dtype=mx.float64),
            n_qmc_samples=32,
            qmc_seed=42,
            constraints_gpr_list=[get_gpr(vals) for vals in c.T],
            constraints_threshold_list=[0.0] * len(c.T),
            stabilizing_noise=0.0,
        )
        verify_eval_acqf(x, acqf)
