from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from optuna._gp.gp import GPRegressor
from optuna._gp.gp import warn_and_convert_inf
import optuna._gp.prior as prior


@pytest.mark.parametrize(
    "values,ans",
    [
        (np.array([-1, 0, 1])[:, np.newaxis], np.array([-1, 0, 1])[:, np.newaxis]),
        (
            np.array([-1, -np.inf, 0, np.inf, 1])[:, np.newaxis],
            np.array([-1, -1, 0, 1, 1])[:, np.newaxis],
        ),
        (np.array([[-1, 2], [0, -2], [1, 0]]), np.array([[-1, 2], [0, -2], [1, 0]])),
        (
            np.array([[-1, 2], [-np.inf, np.inf], [0, -np.inf], [np.inf, -2], [1, 0]]),
            np.array([[-1, 2], [-1, 2], [0, -2], [1, -2], [1, 0]]),
        ),
        (
            np.array(
                [
                    [-100, np.inf, 10],
                    [-np.inf, np.inf, 100],
                    [-10, -np.inf, np.inf],
                    [np.inf, np.inf, -np.inf],
                ]
            ),
            np.array([[-100, 0, 10], [-100, 0, 100], [-10, 0, 100], [-10, 0, 10]]),
        ),
        (np.array([-np.inf, np.inf])[:, np.newaxis], np.array([0, 0])[:, np.newaxis]),
        (np.array([])[:, np.newaxis], np.array([])[:, np.newaxis]),
    ],
)
def test_warn_and_convert_inf_for_2d_array(values: np.ndarray, ans: np.ndarray) -> None:
    assert np.allclose(warn_and_convert_inf(values), ans)


@pytest.mark.parametrize(
    "values,ans",
    [
        (np.array([-1, 0, 1]), np.array([-1, 0, 1])),
        (np.array([-1, -np.inf, 0, np.inf, 1]), np.array([-1, -1, 0, 1, 1])),
        (np.array([-np.inf, np.inf]), np.array([0, 0])),
        (np.array([]), np.array([])),
    ],
)
def test_warn_and_convert_inf_for_1d_array(values: np.ndarray, ans: np.ndarray) -> None:
    assert np.allclose(warn_and_convert_inf(values), ans)


@pytest.mark.parametrize(
    "X, Y, is_categorical",
    [
        (
            np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.1]]),
            np.array([1.0, 2.0, 3.0]),
            np.array([False, False]),
        ),
        (
            np.array([[0.1, 0.2, 0.0], [0.2, 0.3, 1.0]]),
            np.array([1.0, 2.0]),
            np.array([False, False, True]),
        ),
        (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 2.0]), np.array([True, True])),
        (np.array([[0.0]]), np.array([0.0]), np.array([True])),
        (np.array([[0.0]]), np.array([0.0]), np.array([False])),
    ],
)
@pytest.mark.parametrize("deterministic_objective", [True, False])
def test_fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    deterministic_objective: bool,
) -> None:
    with mx.stream(mx.cpu):
        log_prior = prior.default_log_prior
        minimum_noise = prior.DEFAULT_MINIMUM_NOISE_VAR
        gtol: float = 1e-2
        gpr = GPRegressor(
            X_train=mx.array(X, dtype=mx.float64),
            y_train=mx.array(Y, dtype=mx.float64),
            is_categorical=mx.array(is_categorical),
            inverse_squared_lengthscales=mx.ones((X.shape[1],), dtype=mx.float64),
            kernel_scale=mx.array(1.0, dtype=mx.float64),
            noise_var=mx.array(1.0, dtype=mx.float64),
        )._fit_kernel_params(
            log_prior=log_prior,
            minimum_noise=minimum_noise,
            deterministic_objective=deterministic_objective,
            gtol=gtol,
        )

        assert int(
            (gpr.inverse_squared_lengthscales != 1.0).sum()
            + (gpr.kernel_scale != 1.0).sum()
            + (gpr.noise_var != 1.0).sum()
        )


@pytest.mark.parametrize(
    "x_shape", [(1, 3), (2, 3), (1, 2, 3), (2, 1, 3), (2, 2, 3), (2, 2, 2, 3)]
)
def test_posterior(x_shape: tuple[int, ...]) -> None:
    with mx.stream(mx.cpu):
        rng = np.random.RandomState(0)
        X = rng.random(size=(10, x_shape[-1]))
        Y = rng.randn(10)
        Y = (Y - Y.mean()) / Y.std()
        log_prior = prior.default_log_prior
        minimum_noise = prior.DEFAULT_MINIMUM_NOISE_VAR
        gtol: float = 1e-2
        gpr = GPRegressor(
            X_train=mx.array(X, dtype=mx.float64),
            y_train=mx.array(Y, dtype=mx.float64),
            is_categorical=mx.array(np.zeros(X.shape[-1], dtype=bool)),
            inverse_squared_lengthscales=mx.ones((X.shape[1],), dtype=mx.float64),
            kernel_scale=mx.array(1.0, dtype=mx.float64),
            noise_var=mx.array(1.0, dtype=mx.float64),
        )._fit_kernel_params(
            log_prior=log_prior,
            minimum_noise=minimum_noise,
            deterministic_objective=False,
            gtol=gtol,
        )
        x = rng.random(size=x_shape)
        mean_joint, covar = gpr.posterior(mx.array(x, dtype=mx.float64), joint=True)
        mean, var_ = gpr.posterior(mx.array(x, dtype=mx.float64), joint=False)

        mean_np = np.array(mean)
        mean_joint_np = np.array(mean_joint)
        covar_np = np.array(covar)
        var_np = np.array(var_)

        assert mean_joint_np.shape == mean_np.shape
        np.testing.assert_allclose(mean_np, mean_joint_np, rtol=1e-5)
        assert covar_np.shape == (*x_shape[:-1], x_shape[-2])
        diag = np.diagonal(covar_np, axis1=-2, axis2=-1)
        np.testing.assert_allclose(diag, var_np, rtol=1e-5, err_msg="Diagonal Check.")
        np.testing.assert_allclose(
            covar_np, np.swapaxes(covar_np, -2, -1), rtol=1e-5, err_msg="Symmetric Check."
        )
        assert np.all(np.linalg.det(covar_np) >= 0.0), "Positive Semi-definite Check."


@pytest.mark.parametrize("n_running", [1, 5])
def test_append_running_data(n_running: int) -> None:
    with mx.stream(mx.cpu):
        dim = 3
        rng = np.random.RandomState(0)
        X = mx.array(rng.random(size=(10, dim)), dtype=mx.float64)
        Y = mx.array(rng.randn(10), dtype=mx.float64)
        Y = (Y - mx.mean(Y)) / mx.std(Y)
        log_prior = prior.default_log_prior
        minimum_noise = prior.DEFAULT_MINIMUM_NOISE_VAR
        gtol: float = 1e-2
        gpr = GPRegressor(
            X_train=X,
            y_train=Y,
            is_categorical=mx.array(np.zeros(X.shape[-1], dtype=bool)),
            inverse_squared_lengthscales=mx.ones((X.shape[1],), dtype=mx.float64),
            kernel_scale=mx.array(1.0, dtype=mx.float64),
            noise_var=mx.array(1.0, dtype=mx.float64),
        )._fit_kernel_params(
            log_prior=log_prior,
            minimum_noise=minimum_noise,
            deterministic_objective=False,
            gtol=gtol,
        )

        X_running = mx.array(rng.random(size=(n_running, dim)), dtype=mx.float64)
        y_running = mx.array(rng.randn(n_running), dtype=mx.float64)

        reference_gpr = GPRegressor(
            X_train=mx.concatenate([X, X_running], axis=0),
            y_train=mx.concatenate([Y, y_running], axis=0),
            is_categorical=mx.array(np.zeros(X.shape[-1] + n_running, dtype=bool)),
            inverse_squared_lengthscales=mx.array(gpr.inverse_squared_lengthscales),
            kernel_scale=mx.array(gpr.kernel_scale),
            noise_var=mx.array(gpr.noise_var),
        )
        reference_gpr._cache_matrix()

        gpr.append_running_data(X_running, y_running)

        assert reference_gpr._cov_Y_Y_chol is not None
        assert gpr._cov_Y_Y_chol is not None
        assert reference_gpr._cov_Y_Y_inv_Y is not None
        assert gpr._cov_Y_Y_inv_Y is not None
        np.testing.assert_allclose(
            np.array(reference_gpr._cov_Y_Y_chol), np.array(gpr._cov_Y_Y_chol), rtol=1e-5
        )
        np.testing.assert_allclose(
            np.array(reference_gpr._cov_Y_Y_inv_Y), np.array(gpr._cov_Y_Y_inv_Y), rtol=1e-5
        )

        x = mx.array(rng.random(size=(1, dim)), dtype=mx.float64)
        mean, var = gpr.posterior(x)
        reference_mean, reference_var = reference_gpr.posterior(x)
        np.testing.assert_allclose(np.array(mean), np.array(reference_mean), rtol=1e-5)
        np.testing.assert_allclose(np.array(var), np.array(reference_var), rtol=1e-5)
