from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import math
from typing import cast
from typing import TYPE_CHECKING

import numpy as np

from optuna._hypervolume import get_non_dominated_box_bounds
from optuna.study._multi_objective import _is_pareto_front


if TYPE_CHECKING:
    import mlx.core as mx

    from optuna._gp.gp import GPRegressor
    from optuna._gp.search_space import SearchSpace
else:
    from optuna._imports import _LazyImport

    mx = _LazyImport("mlx.core")


_SQRT_HALF = math.sqrt(0.5)
_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
_SQRT_HALF_PI = math.sqrt(0.5 * math.pi)
_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))
_LOG_2 = math.log(2.0)
_EPS = 1e-12  # NOTE(nabenabe): grad becomes nan when EPS=0.
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def _erfcx_asymptotic(x: mx.array) -> mx.array:
    """erfcx(x) = exp(x^2) * erfc(x) via asymptotic expansion for large positive x.

    Uses the series: erfcx(x) = 1/(x*sqrt(pi)) * sum (-1)^n * (2n)! / (n! * (2x)^{2n}).
    Convergent for x > ~4.  "Like Sheldon's series, it converges — eventually." - Raj
    """
    inv_2x2 = 1.0 / (2.0 * mx.square(x))
    # Horner-form of first 8 terms: 1 - t + 3t^2 - 15t^3 + 105t^4 - ...
    # where t = 1/(2x^2)
    result = mx.array(1.0, dtype=x.dtype)
    coeff = 1.0
    for n in range(1, 8):
        coeff *= -(2 * n - 1)
        result = result + coeff * inv_2x2**n
    return _INV_SQRT_PI / x * result


def _erfcx(x: mx.array) -> mx.array:
    """Scaled complementary error function: erfcx(x) = exp(x^2) * erfc(x).

    Uses asymptotic expansion for large x, direct computation for moderate x.
    """
    # For large positive x (> 3.5), use asymptotic expansion (rel_err < 1e-5 at x=3.5)
    # Direct computation degrades at x >= 3.5 due to MLX erf having float32-level precision.
    large_mask = x > 3.5
    # Clamp inputs to each branch to prevent NaN in the unused branch
    # (mx.where gradient propagates through both branches; NaN * 0 = NaN)
    x_direct = mx.minimum(x, 3.5)
    direct = mx.exp(mx.square(x_direct)) * (1.0 - mx.erf(x_direct))
    asymp = _erfcx_asymptotic(mx.maximum(x, 0.1))
    return mx.where(large_mask, asymp, direct)


def _log_ndtr(x: mx.array) -> mx.array:
    """Log of the normal CDF: log(Phi(x)) = log(0.5 * erfc(-x / sqrt(2))).

    For large negative x, uses erfcx-based computation to avoid log(0).
    """
    safe = x > -5.0
    # Clamp inputs to each branch to prevent NaN in the unused branch
    # (mx.where gradient propagates through both; NaN * 0 = NaN)
    x_std = mx.maximum(x, -5.0)
    standard = mx.log(0.5 * (1.0 + mx.erf(x_std * _SQRT_HALF)))
    # Tail: log(Phi(x)) = log(erfc(-x/sqrt2)/2) = -log(2) - x^2/2 + log(erfcx(-x/sqrt2))
    x_tail = mx.minimum(x, -5.0)
    neg_x_sqrt_half = -x_tail * _SQRT_HALF
    tail = -_LOG_2 - 0.5 * mx.square(x_tail) + mx.log(_erfcx(neg_x_sqrt_half))
    return mx.where(safe, standard, tail)


def _sample_from_normal_sobol(dim: int, n_samples: int, seed: int | None) -> mx.array:
    # NOTE(nabenabe): Normal Sobol sampling based on BoTorch.
    # Uses scipy.stats.qmc.Sobol instead of torch.quasirandom.SobolEngine (per ADR-010).
    from scipy.stats.qmc import Sobol

    sobol = Sobol(d=dim, scramble=True, seed=seed)
    sobol_samples = sobol.random(n_samples)  # numpy array in [0, 1]
    samples = 2.0 * (sobol_samples - 0.5)  # [-1, 1]
    # Inverse transform to standard normal (values too close to -1 or 1 result in infinity).
    return mx.erfinv(mx.array(samples, dtype=mx.float64)) * float(np.sqrt(2))


def logehvi(
    Y_post: mx.array,  # (..., n_qmc_samples, n_objectives)
    non_dominated_box_lower_bounds: mx.array,  # (n_boxes, n_objectives)
    non_dominated_box_intervals: mx.array,  # (n_boxes, n_objectives)
) -> mx.array:  # (..., )
    log_n_qmc_samples = float(np.log(Y_post.shape[-2]))
    # This function calculates Eq. (1) of https://arxiv.org/abs/2006.05078.
    diff = mx.expand_dims(Y_post, -2) - non_dominated_box_lower_bounds
    diff = mx.clip(diff, a_min=_EPS, a_max=non_dominated_box_intervals)
    # NOTE(nabenabe): logsumexp with axis=-1 is for the HVI calculation and that with axis=-2 is for
    # expectation of the HVIs over the fixed_samples.
    return mx.logsumexp(mx.sum(mx.log(diff), axis=-1), axis=(-2, -1)) - log_n_qmc_samples


def standard_logei(z: mx.array) -> mx.array:
    """
    Return E_{x ~ N(0, 1)}[max(0, x+z)]
    The calculation depends on the value of z for numerical stability.
    Please refer to Eq. (9) in the following paper for more details:
        https://arxiv.org/pdf/2310.20708.pdf

    NOTE: We do not use the third condition because [-10**100, 10**100] is an overly high range.
    """
    # Second condition check (z < -4.5): stable branch for numerical stability
    small = z < -4.5
    # First condition: clamp z to safe range so log never sees negative arg
    # (prevents NaN propagation through mx.where gradient for extreme z)
    z_safe = mx.maximum(z, -4.5)
    z_half = 0.5 * z_safe
    erfc_val = 1.0 + mx.erf(_SQRT_HALF * z_safe)
    out = mx.log(
        z_half * erfc_val  # z * cdf(z)
        + mx.exp(-z_half * z_safe) * _INV_SQRT_2PI  # pdf(z)
    )
    # Second condition (z < -4.5): numerically stable branch using erfcx
    if mx.any(small):
        z_small = mx.where(small, z, -5.0)  # Safe value for non-small elements
        erfcx_val = _erfcx(-_SQRT_HALF * z_small)
        stable_branch = (
            -0.5 * mx.square(z_small)
            - _LOG_SQRT_2PI
            + mx.log(1.0 + _SQRT_HALF_PI * z_small * erfcx_val)
        )
        out = mx.where(small, stable_branch, out)
    return out


def logei(mean: mx.array, var: mx.array, f0: float) -> mx.array:
    # Return E_{y ~ N(mean, var)}[max(0, y-f0)]
    sigma = mx.sqrt(var)
    return standard_logei((mean - f0) / sigma) + mx.log(sigma)


class BaseAcquisitionFunc(ABC):
    def __init__(self, length_scales: np.ndarray, search_space: SearchSpace) -> None:
        self.length_scales = length_scales
        self.search_space = search_space

    @abstractmethod
    def eval_acqf(self, x: mx.array) -> mx.array:
        raise NotImplementedError

    def eval_acqf_no_grad(self, x: np.ndarray) -> np.ndarray:
        with mx.stream(mx.cpu):
            result = self.eval_acqf(mx.array(x, dtype=mx.float64))
            mx.eval(result)
            return np.array(result)

    def eval_acqf_with_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        assert x.ndim == 1
        with mx.stream(mx.cpu):
            x_mx = mx.array(x, dtype=mx.float64)
            val, grad = mx.value_and_grad(self.eval_acqf)(x_mx)
            mx.eval(val, grad)
            return float(val.item()), np.array(grad)


class LogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        normalized_params_of_running_trials: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._gpr = gpr
        self._stabilizing_noise = stabilizing_noise
        self._threshold = threshold

        if normalized_params_of_running_trials is not None:
            with mx.stream(mx.cpu):
                normalized_params_of_running_trials_mx = mx.array(
                    normalized_params_of_running_trials, dtype=mx.float64
                )

                # NOTE(sawa3030): To handle running trials, the `best` constant liar strategy is
                # currently implemented, as it is simple and performs well in our benchmarks.
                constant_liar_value = mx.max(self._gpr._y_train)
                constant_liar_y = mx.broadcast_to(
                    constant_liar_value,
                    (normalized_params_of_running_trials_mx.shape[0],),
                )

                self._gpr.append_running_data(
                    normalized_params_of_running_trials_mx,
                    constant_liar_y,
                )

        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: mx.array) -> mx.array:
        mean, var = self._gpr.posterior(x)
        # If there are no feasible trials, max_Y is set to -np.inf.
        # If max_Y is set to -np.inf, we set logEI to zero to ignore it.
        return (
            logei(mean=mean, var=var + self._stabilizing_noise, f0=self._threshold)
            if not np.isneginf(self._threshold)
            else mx.zeros(x.shape[:-1], dtype=mx.float64)
        )


class LogPI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        normalized_params_of_running_trials: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._gpr = gpr
        self._stabilizing_noise = stabilizing_noise
        self._threshold = threshold

        if normalized_params_of_running_trials is not None:
            with mx.stream(mx.cpu):
                normalized_params_of_running_trials_mx = mx.array(
                    normalized_params_of_running_trials, dtype=mx.float64
                )

                # NOTE(sawa3030): To handle running trials, the Kriging Believer strategy is
                # currently implemented.
                self._gpr.append_running_data(
                    normalized_params_of_running_trials_mx,
                    gpr.posterior(normalized_params_of_running_trials_mx)[0],
                )
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: mx.array) -> mx.array:
        # Return the integral of N(mean, var) from f0 to inf.
        mean, var = self._gpr.posterior(x)
        sigma = mx.sqrt(var + self._stabilizing_noise)
        return _log_ndtr((mean - self._threshold) / sigma)


class UCB(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        beta: float,
    ) -> None:
        self._gpr = gpr
        self._beta = beta
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: mx.array) -> mx.array:
        mean, var = self._gpr.posterior(x)
        return mean + mx.sqrt(self._beta * var)


class LCB(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        beta: float,
    ) -> None:
        self._gpr = gpr
        self._beta = beta
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: mx.array) -> mx.array:
        mean, var = self._gpr.posterior(x)
        return mean - mx.sqrt(self._beta * var)


class ConstrainedLogEI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        threshold: float,
        constraints_gpr_list: list[GPRegressor],
        constraints_threshold_list: list[float],
        normalized_params_of_running_trials: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        assert (
            len(constraints_gpr_list) == len(constraints_threshold_list) and constraints_gpr_list
        )
        self._acqf = LogEI(
            gpr, search_space, threshold, normalized_params_of_running_trials, stabilizing_noise
        )
        self._constraints_acqf_list = [
            LogPI(
                _gpr,
                search_space,
                _threshold,
                normalized_params_of_running_trials,
                stabilizing_noise,
            )
            for _gpr, _threshold in zip(constraints_gpr_list, constraints_threshold_list)
        ]
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: mx.array) -> mx.array:
        # TODO(kAIto47802): Handle the infeasible case inside `ConstrainedLogEI`
        # instead of `LogEI`.
        return self._acqf.eval_acqf(x) + sum(
            acqf.eval_acqf(x) for acqf in self._constraints_acqf_list
        )


class LogEHVI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr_list: list[GPRegressor],
        search_space: SearchSpace,
        Y_train: mx.array,
        n_qmc_samples: int,
        qmc_seed: int | None,
        normalized_params_of_running_trials: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        def _get_non_dominated_box_bounds() -> tuple[mx.array, mx.array]:
            # NOTE(nabenabe): Y is to be maximized, loss_vals is to be minimized.
            loss_vals = -np.array(Y_train)
            pareto_sols = loss_vals[_is_pareto_front(loss_vals, assume_unique_lexsorted=False)]
            ref_point = np.max(loss_vals, axis=0)
            ref_point = np.nextafter(np.maximum(1.1 * ref_point, 0.9 * ref_point), np.inf)
            lbs, ubs = get_non_dominated_box_bounds(pareto_sols, ref_point)
            # NOTE(nabenabe): Flip back the sign to make them compatible with maximization.
            return mx.array(-ubs, dtype=mx.float64), mx.array(-lbs, dtype=mx.float64)

        self._stabilizing_noise = stabilizing_noise
        self._gpr_list = gpr_list
        if normalized_params_of_running_trials is not None:
            with mx.stream(mx.cpu):
                normalized_params_of_running_trials_mx = mx.array(
                    normalized_params_of_running_trials, dtype=mx.float64
                )

                for gpr in self._gpr_list:
                    gpr.append_running_data(
                        normalized_params_of_running_trials_mx,
                        gpr.posterior(normalized_params_of_running_trials_mx)[0],
                    )

        self._fixed_samples = _sample_from_normal_sobol(
            dim=Y_train.shape[-1], n_samples=n_qmc_samples, seed=qmc_seed
        )
        self._non_dominated_box_lower_bounds, non_dominated_box_upper_bounds = (
            _get_non_dominated_box_bounds()
        )
        self._non_dominated_box_intervals = mx.maximum(
            non_dominated_box_upper_bounds - self._non_dominated_box_lower_bounds, _EPS
        )
        # Since all the objectives are equally important, we simply use the mean of
        # inverse of squared mean lengthscales over all the objectives.
        super().__init__(np.mean([gpr.length_scales for gpr in gpr_list], axis=0), search_space)

    def eval_acqf(self, x: mx.array) -> mx.array:
        Y_post = []
        for i, gpr in enumerate(self._gpr_list):
            mean, var = gpr.posterior(x)
            stdev = mx.sqrt(var + self._stabilizing_noise)
            # NOTE(nabenabe): By using fixed samples from the Sobol sequence, EHVI becomes
            # deterministic, making it possible to optimize the acqf by l-BFGS.
            Y_post.append(mean[..., None] + stdev[..., None] * self._fixed_samples[..., i])

        return logehvi(
            Y_post=mx.stack(Y_post, axis=-1),
            non_dominated_box_lower_bounds=self._non_dominated_box_lower_bounds,
            non_dominated_box_intervals=self._non_dominated_box_intervals,
        )


class ConstrainedLogEHVI(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr_list: list[GPRegressor],
        search_space: SearchSpace,
        Y_feasible: mx.array | None,
        n_qmc_samples: int,
        qmc_seed: int | None,
        constraints_gpr_list: list[GPRegressor],
        constraints_threshold_list: list[float],
        normalized_params_of_running_trials: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        assert (
            len(constraints_gpr_list) == len(constraints_threshold_list) and constraints_gpr_list
        )
        self._acqf = (
            LogEHVI(
                gpr_list,
                search_space,
                Y_feasible,
                n_qmc_samples,
                qmc_seed,
                normalized_params_of_running_trials,
                stabilizing_noise,
            )
            if Y_feasible is not None
            else None
        )
        self._constraints_acqf_list = [
            LogPI(
                _gpr,
                search_space,
                _threshold,
                normalized_params_of_running_trials,
                stabilizing_noise,
            )
            for _gpr, _threshold in zip(constraints_gpr_list, constraints_threshold_list)
        ]
        super().__init__(np.mean([gpr.length_scales for gpr in gpr_list], axis=0), search_space)

    def eval_acqf(self, x: mx.array) -> mx.array:
        constraints_acqf_values = sum(acqf.eval_acqf(x) for acqf in self._constraints_acqf_list)
        if self._acqf is None:
            return cast("mx.array", constraints_acqf_values)
        return constraints_acqf_values + self._acqf.eval_acqf(x)
