"""Notations in this Gaussian process implementation

X_train: Observed parameter values with the shape of (len(trials), len(params)).
y_train: Observed objective values with the shape of (len(trials), ).
x: (Possibly batched) parameter value(s) to evaluate with the shape of (..., len(params)).
cov_fX_fX: Kernel matrix X = V[f(X)] with the shape of (len(trials), len(trials)).
cov_fx_fX: Kernel matrix Cov[f(x), f(X)] with the shape of (..., len(trials)).
cov_fx_fx: Kernel scalar value x = V[f(x)]. This value is constant for the Matern 5/2 kernel.
cov_Y_Y_inv:
    The inverse of the covariance matrix (V[f(X) + noise_var])^-1 with the shape of
    (len(trials), len(trials)).
cov_Y_Y_inv_Y: `cov_Y_Y_inv @ y` with the shape of (len(trials), ).
max_Y: The maximum of Y (Note that we transform the objective values such that it is maximized.)
sqd: The squared differences of each dimension between two points.
is_categorical:
    A boolean array with the shape of (len(params), ). If is_categorical[i] is True, the i-th
    parameter is categorical.
"""

from __future__ import annotations

import math
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna._gp.scipy_blas_thread_patch import single_blas_thread_if_scipy_v1_15_or_newer
from optuna._warnings import optuna_warn
from optuna.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    import mlx.core as mx
    import scipy
else:
    from optuna._imports import _LazyImport

    mx = _LazyImport("mlx.core")
    scipy = _LazyImport("scipy")

logger = get_logger(__name__)


def warn_and_convert_inf(values: np.ndarray) -> np.ndarray:
    is_values_finite = np.isfinite(values)
    if np.all(is_values_finite):
        return values

    optuna_warn("Clip non-finite values to the min/max finite values for GP fittings.")
    is_any_finite = np.any(is_values_finite, axis=0)
    # NOTE(nabenabe): values cannot include nan to apply np.clip properly, but Optuna anyways won't
    # pass nan in values by design.
    return np.clip(
        values,
        np.where(is_any_finite, np.min(np.where(is_values_finite, values, np.inf), axis=0), 0.0),
        np.where(is_any_finite, np.max(np.where(is_values_finite, values, -np.inf), axis=0), 0.0),
    )


@mx.custom_function
def matern52_kernel(squared_distance: mx.array) -> mx.array:
    """Matern 5/2 kernel: exp(-sqrt5d) * (1 + sqrt5d + 5/3 * d^2).

    Uses a custom VJP because the automatic derivative through mx.sqrt(0) is
    infinite, while the true derivative (-5/6)(1+sqrt5d)exp(-sqrt5d) is finite.
    Same fix as the original torch Matern52Kernel class, different framework.
    "Like Sheldon with a different apartment — same genius, different address."
    """
    sqrt5d = mx.sqrt(5.0 * mx.maximum(squared_distance, 0.0))
    return mx.exp(-sqrt5d) * (1.0 + sqrt5d + (5.0 / 3.0) * squared_distance)


@matern52_kernel.vjp
def _matern52_vjp(
    primals: mx.array, cotangent: mx.array, output: mx.array
) -> mx.array:
    """df/d(d^2) = (-5/6)(1 + sqrt(5*d^2)) * exp(-sqrt(5*d^2)), finite everywhere."""
    squared_distance = primals
    sqrt5d = mx.sqrt(5.0 * mx.maximum(squared_distance, 0.0))
    deriv = (-5.0 / 6.0) * (sqrt5d + 1.0) * mx.exp(-sqrt5d)
    return cotangent * deriv


# MLX does not implement VJPs for linalg.cholesky or linalg.solve_triangular,
# so we provide an analytical gradient for the marginal log likelihood using
# the well-known formula from Rasmussen & Williams (2006), eq. 5.9.
# "When the framework won't differentiate for you, you differentiate yourself.
# That's what separates physicists from mere mortals." - Sheldon
@mx.custom_function
def _differentiable_mll(
    kernel_matrix: mx.array, y_train: mx.array, noise_var: mx.array
) -> mx.array:
    """Marginal log likelihood with custom VJP for Cholesky-dependent gradients."""
    n = kernel_matrix.shape[0]
    C = kernel_matrix + noise_var * mx.eye(n, dtype=kernel_matrix.dtype)
    L = mx.linalg.cholesky(C)
    logdet_part = -mx.sum(mx.log(mx.diag(L)))
    inv_L_y = mx.linalg.solve_triangular(L, y_train[:, None], upper=False)[:, 0]
    quad_part = -0.5 * (inv_L_y @ inv_L_y)
    const = -0.5 * n * math.log(2 * math.pi)
    return logdet_part + quad_part + const


@_differentiable_mll.vjp
def _mll_vjp(
    primals: tuple[mx.array, mx.array, mx.array],
    cotangent: mx.array,
    output: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Analytical VJP for MLL (Rasmussen & Williams eq. 5.9).

    ∂MLL/∂C = 0.5 * (α α^T - C^{-1}) where α = C^{-1} y.
    Since C = K + σ²I, the gradient w.r.t. K equals ∂MLL/∂C
    and the gradient w.r.t. σ² equals trace(∂MLL/∂C).
    """
    kernel_matrix, y_train, noise_var = primals
    n = kernel_matrix.shape[0]
    dtype = kernel_matrix.dtype

    C = kernel_matrix + noise_var * mx.eye(n, dtype=dtype)
    L = mx.linalg.cholesky(C)

    # α = C^{-1} y via forward-backward substitution
    inv_L_y = mx.linalg.solve_triangular(L, y_train[:, None], upper=False)[:, 0]
    Lt = mx.swapaxes(L, -2, -1)
    alpha = mx.linalg.solve_triangular(Lt, inv_L_y[:, None], upper=True)[:, 0]

    # W = C^{-1} = (L^{-1})^T @ L^{-1}
    L_inv = mx.linalg.solve_triangular(L, mx.eye(n, dtype=dtype), upper=False)
    W = mx.swapaxes(L_inv, -2, -1) @ L_inv

    # ∂MLL/∂K = 0.5 * (α α^T - W), ∂MLL/∂σ² = 0.5 * (α·α - trace(W))
    aat = alpha[:, None] * alpha[None, :]
    grad_K = cotangent * 0.5 * (aat - W)
    grad_noise = cotangent * 0.5 * (alpha @ alpha - mx.sum(mx.diag(W)))
    grad_y = mx.zeros_like(y_train)

    return (grad_K, grad_y, grad_noise)


def _solve_triangular_right(
    A: mx.array, B: mx.array, upper: bool
) -> mx.array:
    """Solve X @ A = B for X, where A is triangular.

    Equivalent to torch.linalg.solve_triangular(A, B, upper=upper, left=False).
    Uses the identity: X @ A = B  ->  A.T @ X.T = B.T
    """
    return mx.swapaxes(
        mx.linalg.solve_triangular(
            mx.swapaxes(A, -2, -1),
            mx.swapaxes(B, -2, -1),
            upper=not upper,
        ),
        -2, -1,
    )


class GPRegressor:
    def __init__(
        self,
        is_categorical: mx.array,
        X_train: mx.array,
        y_train: mx.array,
        inverse_squared_lengthscales: mx.array,  # (len(params), )
        kernel_scale: mx.array,  # Scalar
        noise_var: mx.array,  # Scalar
    ) -> None:
        self._is_categorical = is_categorical
        self._X_train = X_train
        self._y_train = y_train
        self._X_all = X_train
        self._y_all = y_train
        self._squared_X_diff = mx.square(
            mx.expand_dims(X_train, -2) - mx.expand_dims(X_train, -3)
        )
        if mx.any(self._is_categorical):
            self._squared_X_diff = mx.where(
                mx.expand_dims(self._is_categorical, tuple(range(self._squared_X_diff.ndim - 1))),
                (self._squared_X_diff > 0.0).astype(mx.float64),
                self._squared_X_diff,
            )
        self._cov_Y_Y_chol: mx.array | None = None
        self._cov_Y_Y_inv_Y: mx.array | None = None
        # TODO(nabenabe): Rename the attributes to private with `_`.
        self.inverse_squared_lengthscales = inverse_squared_lengthscales
        self.kernel_scale = kernel_scale
        self.noise_var = noise_var

    @property
    def length_scales(self) -> np.ndarray:
        return 1.0 / np.sqrt(np.array(self.inverse_squared_lengthscales))

    def _cache_matrix(self) -> None:
        assert self._cov_Y_Y_chol is None and self._cov_Y_Y_inv_Y is None, (
            "Cannot call cache_matrix more than once."
        )
        cov_Y_Y = np.array(self.kernel())

        cov_Y_Y[np.diag_indices(self._X_train.shape[0])] += self.noise_var.item()
        cov_Y_Y_chol = np.linalg.cholesky(cov_Y_Y)
        # cov_Y_Y_inv @ y = v --> y = cov_Y_Y @ v --> y = cov_Y_Y_chol @ cov_Y_Y_chol.T @ v
        # NOTE(nabenabe): Don't use np.linalg.inv because it is too slow und unstable.
        # cf. https://github.com/optuna/optuna/issues/6230
        cov_Y_Y_inv_Y = scipy.linalg.solve_triangular(
            cov_Y_Y_chol.T,
            scipy.linalg.solve_triangular(cov_Y_Y_chol, np.array(self._y_train), lower=True),
            lower=False,
        )
        # NOTE(nabenabe): Here we use NumPy to guarantee the reproducibility from the past.
        self._cov_Y_Y_chol = mx.array(cov_Y_Y_chol)
        self._cov_Y_Y_inv_Y = mx.array(cov_Y_Y_inv_Y)
        self.inverse_squared_lengthscales = mx.stop_gradient(
            self.inverse_squared_lengthscales
        )
        self.kernel_scale = mx.stop_gradient(self.kernel_scale)
        self.noise_var = mx.stop_gradient(self.noise_var)

    def append_running_data(self, X_running: mx.array, y_running: mx.array) -> None:
        assert self._cov_Y_Y_chol is not None and self._cov_Y_Y_inv_Y is not None, (
            "Call _cache_matrix before append_running_data"
        )
        n_train = self._X_train.shape[0]
        n_running = X_running.shape[0]
        n_total = n_train + n_running

        cov_Y_Y_chol = np.zeros((n_total, n_total), dtype=np.float64)
        cov_Y_Y_chol[:n_train, :n_train] = np.array(self._cov_Y_Y_chol)
        kernel_running_train = np.array(self.kernel(X_running))
        kernel_running_running = np.array(self.kernel(X_running, X_running))
        kernel_running_running[np.diag_indices(n_running)] += self.noise_var.item()

        cov_Y_Y_chol[n_train:, :n_train] = scipy.linalg.solve_triangular(
            np.array(self._cov_Y_Y_chol), kernel_running_train.T, lower=True
        ).T
        cov_Y_Y_chol[n_train:, n_train:] = np.linalg.cholesky(
            kernel_running_running
            - cov_Y_Y_chol[n_train:, :n_train] @ cov_Y_Y_chol[n_train:, :n_train].T
        )
        self._y_all = mx.concatenate([self._y_train, y_running], axis=0)
        cov_Y_Y_inv_Y = scipy.linalg.solve_triangular(
            cov_Y_Y_chol.T,
            scipy.linalg.solve_triangular(cov_Y_Y_chol, np.array(self._y_all), lower=True),
            lower=False,
        )

        # NOTE(nabenabe): Here we use NumPy to guarantee the reproducibility from the past.
        self._cov_Y_Y_chol = mx.array(cov_Y_Y_chol)
        self._cov_Y_Y_inv_Y = mx.array(cov_Y_Y_inv_Y)
        self._X_all = mx.concatenate([self._X_train, X_running], axis=0)

    def kernel(
        self, X1: mx.array | None = None, X2: mx.array | None = None
    ) -> mx.array:
        """
        Return the kernel matrix with the shape of (..., n_A, n_B) given X1 and X2 each with the
        shapes of (..., n_A, len(params)) and (..., n_B, len(params)).

        If x1 and x2 have the shape of (len(params), ), kernel(x1, x2) is computed as:
            kernel_scale * matern52_kernel(
                sqd(x1, x2) @ inverse_squared_lengthscales
            )
        where if x1[i] is continuous, sqd(x1, x2)[i] = (x1[i] - x2[i]) ** 2 and if x1[i] is
        categorical, sqd(x1, x2)[i] = int(x1[i] != x2[i]).
        Note that the distance for categorical parameters is the Hamming distance.
        """
        if X1 is None:
            assert X2 is None
            sqd = self._squared_X_diff
        else:
            if X2 is None:
                X2 = self._X_train

            sqd = mx.square(
                X1 - X2 if X1.ndim == 1
                else mx.expand_dims(X1, -2) - mx.expand_dims(X2, -3)
            )
            if mx.any(self._is_categorical):
                sqd = mx.where(
                    mx.expand_dims(
                        self._is_categorical,
                        tuple(range(sqd.ndim - 1)),
                    ),
                    (sqd > 0.0).astype(mx.float64),
                    sqd,
                )
        sqdist = sqd @ self.inverse_squared_lengthscales
        return matern52_kernel(sqdist) * self.kernel_scale

    def posterior(self, x: mx.array, joint: bool = False) -> tuple[mx.array, mx.array]:
        """
        This method computes the posterior mean and variance given the points `x` where both mean
        and variance tensors will have the shape of x.shape[:-1].
        If ``joint=True``, the joint posterior will be computed.

        The posterior mean and variance are computed as:
            mean = cov_fx_fX @ inv(cov_fX_fX + noise_var * I) @ y, and
            var = cov_fx_fx - cov_fx_fX @ inv(cov_fX_fX + noise_var * I) @ cov_fx_fX.T.

        Please note that we clamp the variance to avoid negative values due to numerical errors.
        """
        assert self._cov_Y_Y_chol is not None and self._cov_Y_Y_inv_Y is not None, (
            "Call cache_matrix before calling posterior."
        )
        is_single_point = x.ndim == 1
        x_ = x if not is_single_point else mx.expand_dims(x, 0)
        cov_fx_fX = self.kernel(x_, self._X_all)
        mean = mx.sum(cov_fx_fX * self._cov_Y_Y_inv_Y, axis=-1)
        # K @ inv(C) = V --> K = V @ C --> K = V @ L @ L.T
        # Solve V @ L.T = cov_fx_fX (inner), then V_final @ L = V_inner (outer)
        V = _solve_triangular_right(
            self._cov_Y_Y_chol,
            _solve_triangular_right(
                mx.swapaxes(self._cov_Y_Y_chol, -2, -1),
                cov_fx_fX,
                upper=True,
            ),
            upper=False,
        )
        if joint:
            assert not is_single_point, "Call posterior with joint=False for a single point."
            cov_fx_fx = self.kernel(x_, x_)
            # NOTE(nabenabe): Indeed, var_ here is a covariance matrix.
            var_ = cov_fx_fx - V @ mx.swapaxes(cov_fx_fX, -1, -2)
            # Clamp diagonal to non-negative (replaces in-place .clamp_min_(0))
            # Works for any number of batch dimensions by using eye broadcasting.
            n = var_.shape[-1]
            eye = mx.eye(n, dtype=var_.dtype)
            diag_vals = mx.sum(var_ * eye, axis=-1)
            correction = mx.maximum(diag_vals, 0.0) - diag_vals
            var_ = var_ + correction[..., None] * eye
        else:
            cov_fx_fx = self.kernel_scale  # kernel(x, x) = kernel_scale
            var_ = cov_fx_fx - mx.sum(cov_fx_fX * V, axis=-1)
            var_ = mx.maximum(var_, 0.0)
        return (mx.squeeze(mean, 0), mx.squeeze(var_, 0)) if is_single_point else (mean, var_)

    def marginal_log_likelihood(self) -> mx.array:  # Scalar
        """
        This method computes the marginal log-likelihood of the kernel hyperparameters given the
        training dataset (X, y).
        Assume that N = len(X) in this method.

        Mathematically, the closed form is given as:
            -0.5 * log((2*pi)**N * det(C)) - 0.5 * y.T @ inv(C) @ y
            = -0.5 * log(det(C)) - 0.5 * y.T @ inv(C) @ y + const,
        where C = cov_Y_Y = cov_fX_fX + noise_var * I and inv(...) is the inverse operator.

        We exploit the full advantages of the Cholesky decomposition (C = L @ L.T) in this method:
            1. The determinant of a lower triangular matrix is the diagonal product, which can be
               computed with N flops where log(det(C)) = log(det(L.T @ L)) = 2 * log(det(L)).
            2. Solving linear system L @ u = y, which yields u = inv(L) @ y, costs N**2 flops.
        Note that given `u = inv(L) @ y` and `inv(C) = inv(L @ L.T) = inv(L).T @ inv(L)`,
        y.T @ inv(C) @ y is calculated as (inv(L) @ y) @ (inv(L) @ y).

        In principle, we could invert the matrix C first, but in this case, it costs:
            1. 1/3*N**3 flops for the determinant of inv(C).
            2. 2*N**2-N flops to solve C @ alpha = y, which is alpha = inv(C) @ y.

        Since the Cholesky decomposition costs 1/3*N**3 flops and the matrix inversion costs
        2/3*N**3 flops, the overall cost for the former is 1/3*N**3+N**2+N flops and that for the
        latter is N**3+2*N**2-N flops.

        Uses _differentiable_mll (custom VJP) so mx.value_and_grad flows through
        Cholesky and solve_triangular, which lack built-in MLX VJPs.
        """
        return _differentiable_mll(self.kernel(), self._y_train, self.noise_var)

    def _fit_kernel_params(
        self,
        log_prior: Callable[[GPRegressor], mx.array],
        minimum_noise: float,
        deterministic_objective: bool,
        gtol: float,
    ) -> GPRegressor:
        n_params = self._X_train.shape[1]

        # We apply log transform to enforce the positivity of the kernel parameters.
        # Note that we cannot just use the constraint because of the numerical unstability
        # of the marginal log likelihood.
        # We also enforce the noise parameter to be greater than `minimum_noise` to avoid
        # pathological behavior of maximum likelihood estimation.
        initial_raw_params = np.concatenate(
            [
                np.log(np.array(self.inverse_squared_lengthscales)),
                [
                    np.log(self.kernel_scale.item()),
                    # We add 0.01 * minimum_noise to initial noise_var to avoid instability.
                    np.log(self.noise_var.item() - 0.99 * minimum_noise),
                ],
            ]
        )

        def loss_func(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
            raw_params_mx = mx.array(raw_params, dtype=mx.float64)

            def loss_fn(params: mx.array) -> mx.array:
                self.inverse_squared_lengthscales = mx.exp(params[:n_params])
                self.kernel_scale = mx.exp(params[n_params])
                self.noise_var = (
                    mx.array(minimum_noise, dtype=mx.float64)
                    if deterministic_objective
                    else mx.exp(params[n_params + 1]) + minimum_noise
                )
                return -(self.marginal_log_likelihood() + log_prior(self))

            loss_val, grad_val = mx.value_and_grad(loss_fn)(raw_params_mx)
            mx.eval(loss_val, grad_val)
            # scipy.minimize requires all the gradients to be zero for termination.
            assert not deterministic_objective or np.array(grad_val)[n_params + 1] == 0
            return float(loss_val.item()), np.array(grad_val)

        with single_blas_thread_if_scipy_v1_15_or_newer():
            # jac=True means loss_func returns the gradient for gradient descent.
            res = scipy.optimize.minimize(
                # Too small `gtol` causes instability in loss_func optimization.
                loss_func,
                initial_raw_params,
                jac=True,
                method="l-bfgs-b",
                options={"gtol": gtol},
            )
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        raw_params_opt = mx.array(res.x, dtype=mx.float64)
        self.inverse_squared_lengthscales = mx.exp(raw_params_opt[:n_params])
        self.kernel_scale = mx.exp(raw_params_opt[n_params])
        self.noise_var = (
            mx.array(minimum_noise, dtype=mx.float64)
            if deterministic_objective
            else minimum_noise + mx.exp(raw_params_opt[n_params + 1])
        )
        self._cache_matrix()
        return self


def fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    log_prior: Callable[[GPRegressor], mx.array],
    minimum_noise: float,
    deterministic_objective: bool,
    gpr_cache: GPRegressor | None = None,
    gtol: float = 1e-2,
) -> GPRegressor:
    with mx.stream(mx.cpu):
        default_kernel_params = mx.ones((X.shape[1] + 2,), dtype=mx.float64)
        # TODO: Move this function into a method of `GPRegressor`

        def _default_gpr() -> GPRegressor:
            return GPRegressor(
                is_categorical=mx.array(is_categorical),
                X_train=mx.array(X, dtype=mx.float64),
                y_train=mx.array(Y, dtype=mx.float64),
                inverse_squared_lengthscales=default_kernel_params[:-2],
                kernel_scale=default_kernel_params[-2],
                noise_var=default_kernel_params[-1],
            )

        default_gpr_cache = _default_gpr()
        if gpr_cache is None:
            gpr_cache = _default_gpr()

        error = None
        # First try optimizing the kernel params with the provided kernel parameters in gpr_cache,
        # but if it fails, rerun the optimization with the default kernel parameters above.
        # This increases the robustness of the optimization.
        for gpr_cache_to_use in [gpr_cache, default_gpr_cache]:
            try:
                return GPRegressor(
                    is_categorical=mx.array(is_categorical),
                    X_train=mx.array(X, dtype=mx.float64),
                    y_train=mx.array(Y, dtype=mx.float64),
                    inverse_squared_lengthscales=gpr_cache_to_use.inverse_squared_lengthscales,
                    kernel_scale=gpr_cache_to_use.kernel_scale,
                    noise_var=gpr_cache_to_use.noise_var,
                )._fit_kernel_params(
                    log_prior=log_prior,
                    minimum_noise=minimum_noise,
                    deterministic_objective=deterministic_objective,
                    gtol=gtol,
                )
            except RuntimeError as e:
                error = e

        logger.warning(
            f"The optimization of kernel parameters failed: \n{error}\n"
            "The default initial kernel parameters will be used instead."
        )
        default_gpr = _default_gpr()
        default_gpr._cache_matrix()
        return default_gpr
