"""Gradient step plug-and-play (PnP) reconstructor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, cast

import deepinv
import torch
from deepinv.optim import PGD
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import RED

from LION.classical_algorithms.conjugate_gradient import (
    CGProxParameter,
    conjugate_gradient,
)
from LION.experiments.config.other_config import NormParameter
from LION.operators.Operator import Operator
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm_tools.pcm_prox import (
    PCMProxParameter,
    closed_form_pcm_prox,
    closed_form_pcm_prox_with_precomputed_terms,
    precompute_pcm_prox_terms,
)
from LION.reconstructors.BasePnP import BasePnP, DenoiserParameter
from LION.utils.parameter import LIONParameter


@dataclass(kw_only=True)
class GSPnPParameter(LIONParameter):
    """Configuration for GS-PnP reconstruction.

    Reference: ``GRADIENT STEP DENOISER FOR CONVERGENT PLUGAND-PLAY'' (Hurault et al., 2022) https://arxiv.org/abs/2110.03220

    Attributes:
        denoiser: The configuration for the denoiser.
        iters: The number of iterations.
        lambd: Value of regularization strength lambda, should be greater than 0.
        stepsize: Initial value of tau, should be greater than 0. If None, defaults to 1/lambda.
        eps: Early stopping threshold for relative change in objective function, should be greater than 0.
        use_backtracking: Whether to apply backtracking as the paper suggests.
        gamma_backtracking: Should be within (0, 0.5).
        eta_backtracking: Should be within (0, 1).
        data_fidelity: Configuration for the data fidelity term.
        data_fidelity_prox: Configuration for the proximal step algorithm.
        uses_deepinv: Whether to use DeepInv backend.
        early_stop: Whether the configured convergence criterion may stop GS-PnP early.
        continue_after_early_stop: If True, record the first convergence point but continue
            to ``iters`` accepted iterations.
        max_backtracking_attempts: Maximum number of rejected backtracking candidates
            allowed before stopping with an error.
    """

    denoiser: DenoiserParameter
    iters: int
    lambd: float
    stepsize: float | None
    eps: float
    use_backtracking: bool = True
    gamma_backtracking: float
    eta_backtracking: float
    sigma: float | None = None
    data_fidelity: Any
    data_fidelity_prox: Any
    uses_deepinv: bool = False
    early_stop: bool = True
    continue_after_early_stop: bool = False
    max_backtracking_attempts: int = 100


class GSDenoiser(torch.nn.Module):
    """Gradient step denoiser."""

    def __init__(
        self,
        network: torch.nn.Module,
        adds_batch_dim: bool = False,
        adds_channel_dim: bool = False,
    ):
        """Gradient step denoiser.

        Args:
            network: The backend network.
            adds_batch_dim: Whether to add a batch dimension to the input of the network.
            adds_channel_dim: Whether to add a channel dimension to the input of the network.
        """
        super().__init__()
        self.network = network
        self.adds_batch_dim = adds_batch_dim
        self.adds_channel_dim = adds_channel_dim

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor before passing it to the network."""
        return x  # Default implementation does nothing. Override if needed.

    def postprocess(self, nabla_g: torch.Tensor) -> torch.Tensor:
        """Postprocess the output tensor from the network."""
        return nabla_g  # Default implementation does nothing. Override if needed.

    def forward(
        self,
        x: torch.Tensor,
        sigma: float | None = None,
        *,
        nabla_g: torch.Tensor | None = None,
        weight: float = 1.0,
        set_reset_grad: bool = False,
    ) -> torch.Tensor:
        r"""Denoise the input tensor.

        The denoised output is $x - \w \nabla g(x)$ where $g(x) = \frac{1}{2} ||x - N(x)||^2$, $N$ is the network,
        and $w$ is an optional weight.

        Reference: ``GRADIENT STEP DENOISER FOR CONVERGENT PLUGAND-PLAY'' (Hurault et al., 2022) https://arxiv.org/abs/2110.03220

        Args:
            x: The input tensor to be denoised.
            sigma: The noise level.
            nabla_g: Optional pre-computed gradient of the regularization term.
            weight: Weight for the gradient step.
            set_reset_grad: Whether to set and reset gradients (should be True for inference).

        Returns:
            The denoised tensor.
        """
        if nabla_g is None:
            nabla_g, _ = self.grad(x, sigma, set_reset_grad=set_reset_grad)
        return x - weight * nabla_g

    def grad(
        self, x: torch.Tensor, sigma: float | None, set_reset_grad: bool
    ) -> tuple[torch.Tensor, float]:
        r"""Calculate the gradient of the regularization term.

        The regularization term is $g(x) = \frac{1}{2} ||x - N(x)||^2$, where $N$ is the network.
        The gradient is computed using autograd.

        Reference: ``GRADIENT STEP DENOISER FOR CONVERGENT PLUGAND-PLAY'' (Hurault et al., 2022) https://arxiv.org/abs/2110.03220

        Args:
            x: The input tensor.
            sigma: The noise level.
            set_reset_grad: Whether to set and reset gradients (should be True for inference).

        Returns:
            nabla_g, g: Gradient of the regularization term and value of the regularization term.
        """
        x = self.preprocess(x)
        with torch.enable_grad():
            x_grad = (
                x.detach().requires_grad_(True)
                if set_reset_grad
                else x.requires_grad_()
            )
            g = self.potential(x_grad, sigma)  # Regularization term `g`
            # Compute gradient of `g` with autograd
            nabla_g = torch.autograd.grad(
                g, x_grad, torch.ones_like(g), create_graph=not set_reset_grad
            )[0]
        if set_reset_grad:  # Disable autograd if required
            nabla_g = nabla_g.detach()
        nabla_g = self.postprocess(nabla_g)
        return nabla_g, g.item()

    def potential(self, x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
        """Compute the GS potential."""
        if self.adds_batch_dim:
            x = x.unsqueeze(0)  # Add batch dimension if required
        if self.adds_channel_dim:
            x = x.unsqueeze(0)  # Add channel dimension if required
        network_output = self.network(x, sigma)
        if self.adds_batch_dim:
            x = x.squeeze(0)  # Remove batch dimension if added
            network_output = network_output.squeeze(
                0
            )  # Remove batch dimension if added
        if self.adds_channel_dim:
            x = x.squeeze(0)  # Remove channel dimension if added
            network_output = network_output.squeeze(
                0
            )  # Remove channel dimension if added
        # Regularization term `g` is defined as 0.5 * ||x - N(x)||^2, where N is the network
        return 0.5 * torch.sum(
            torch.square(x - network_output)
        )  # 1/2 * squared_L2_norm


class GSPnP(BasePnP[GSDenoiser]):
    """Gradient step plug-and-play reconstructor."""

    def __init__(
        self,
        physics: Operator,
        denoiser: GSDenoiser,
        measurement_dims: tuple[int, ...] = (-1,),
    ):
        """Initialize the GS-PnP reconstructor.

        Args:
            physics: A CT Geometry or an Operator. If Geometry, CT operator will be inferred.
            denoiser: The gradient step denoiser prior used in the GS-PnP algorithm.
            measurement_dims: Dimensions belonging to one measurement sample. Useful for normalizing.
        """
        super().__init__(physics, denoiser)
        self.measurement_dims = measurement_dims

    @torch.no_grad()
    def reconstruct_sample(self, measurement: torch.Tensor, **kwargs) -> torch.Tensor:
        """``GRADIENT STEP DENOISER FOR CONVERGENT PLUGAND-PLAY'' (Hurault et al., 2022)  https://arxiv.org/abs/2110.03220.

        Note: Even if number of iterations <= 0, the algorithm will still perform gradient step.

        Args:
            measurement: Measurement tensor.
            **kwargs: Additional arguments consisting of:
                - `config`: Configuration parameters for the GS-PnP algorithm.
                - `init_recon`: Initialization of the algorithm.

        Returns:
            Reconstructed image tensor.
        """
        config: GSPnPParameter = cast(GSPnPParameter, kwargs["config"])
        init_recon: torch.Tensor | None = kwargs.get("init_recon")
        f: Callable[[torch.Tensor, torch.Tensor], float] | None = kwargs.get("f")
        prox_tau_f: Callable[[torch.Tensor, float], torch.Tensor] | None = kwargs.get(
            "prox_tau_f"
        )

        def denoiser_grad(x: torch.Tensor) -> tuple[torch.Tensor, float]:
            return self.denoiser.grad(
                x, config.sigma, set_reset_grad=True
            )  # Assume for inference only

        if f is None:
            f = self.prep_data_fidelity(config.data_fidelity, self.measurement_dims)
        if prox_tau_f is None:
            prox_tau_f = self.prep_data_fidelity_prox(
                config.data_fidelity_prox, measurement, self.measurement_dims
            )

        lambd = config.lambd
        # if config.data_fidelity.normalize:  # TODO: Check this
        #     lambd = lambd / measurement.numel()  # Normalize lambda by number of measurements

        # Note: `tau` will continually change due to the backtracking process
        tau = config.stepsize if config.stepsize is not None else 1.0 / lambd

        if init_recon is None:
            init_recon = torch.zeros_like(
                self.op.adjoint(measurement)
            )  # Default initialization with zeros
        recon = init_recon.clone()
        x = prox_tau_f(recon, tau)  # Initial proximal step
        nabla_g, g = denoiser_grad(x)  # Regularization term `g` and its gradient
        F0 = f(x, measurement) + lambd * g
        F = F0
        delta = (
            config.eps + 1.0
        )  # Init delta greater than eps so that the first iteration runs
        i = 0
        while i < config.iters:
            if delta <= config.eps:
                break  # Stop if there is no significant relative change in objective function
            recon = x - (lambd * tau * nabla_g)  # Gradient step
            x_new = prox_tau_f(recon, tau)  # Proximal step
            nabla_g_new, g_new = denoiser_grad(
                x_new
            )  # Regularization term `g` and its gradient
            F_new = f(x_new, measurement) + lambd * g_new  # Variational objective
            if config.use_backtracking:  # Backtracking
                diff_x = torch.sum(torch.square(x_new - x)).item()  # ||x_new - x||^2
                if F - F_new < (config.gamma_backtracking / tau) * diff_x:
                    tau = config.eta_backtracking * tau
                    continue
            delta = (F - F_new) / F0
            x = x_new
            nabla_g = nabla_g_new
            g = g_new
            F = F_new
            i += 1
        return x - (
            lambd * tau * nabla_g
        )  # Final gradient step after the last iteration

    def prep_data_fidelity(
        self, config, measurement_dims: tuple[int, ...]
    ) -> Callable[[torch.Tensor, torch.Tensor], float]:
        """Create the data fidelity function `f`.

        Assume `f` to be proper, lower semicontinous and convex.

        Args:
            config: Configuration parameters for the data fidelity term.
            measurement_dims: Dimensions belonging to one measurement sample. Used for normalizing.

        Returns:
            The data fidelity function. Currently supports 1-norm and squared 2-norm.
        """
        if isinstance(config, NormParameter):
            factor = config.factor if config.factor is not None else 1.0 / config.ord
            if config.ord == 2:  # 2-norm squared

                if config.normalize:

                    def data_fidelity(
                        recon: torch.Tensor, measurement: torch.Tensor
                    ) -> float:
                        measurement_size = math.prod(
                            measurement.shape[d] for d in measurement_dims
                        )
                        return (
                            factor
                            / measurement_size
                            * torch.sum(
                                torch.square(self.op(recon) - measurement)
                            ).item()
                        )

                else:

                    def data_fidelity(
                        recon: torch.Tensor, measurement: torch.Tensor
                    ) -> float:
                        return (
                            factor
                            * torch.sum(
                                torch.square(self.op(recon) - measurement)
                            ).item()
                        )

            elif config.ord == 1:  # 1-norm. Note: Should not be used with CG.

                if config.normalize:

                    def data_fidelity(
                        recon: torch.Tensor, measurement: torch.Tensor
                    ) -> float:
                        measurement_size = math.prod(
                            measurement.shape[d] for d in measurement_dims
                        )
                        return (
                            factor
                            / measurement_size
                            * torch.sum(torch.abs(self.op(recon) - measurement)).item()
                        )

                else:

                    def data_fidelity(
                        recon: torch.Tensor, measurement: torch.Tensor
                    ) -> float:
                        return (
                            factor
                            * torch.sum(torch.abs(self.op(recon) - measurement)).item()
                        )

            else:
                raise ValueError(f"Unsupported norm order: {config.ord}.")
            return data_fidelity

        raise ValueError(f"Unsupported data fidelity configuration: {type(config)}")

    def prep_data_fidelity_prox(
        self, config, measurement: torch.Tensor, measurement_dims: tuple[int, ...]
    ) -> Callable[[torch.Tensor, float], torch.Tensor]:
        """Create the proximal step for the data fidelity term.

        Args:
            config: Configuration parameters for the proximal step algorithm.
            measurement: The measurement tensor. Useful if some precomputation can be done.
            measurement_dims: Dimensions belonging to one measurement sample. Used for normalizing.

        Returns:
            The proximal step algorithm. It takes current reconstruction and tau, and returns updated reconstruction.
        """
        op = self.op
        if isinstance(config, CGProxParameter):
            # Normalize data fidelity term by number of measurements so that eta has consistent meaning
            # regardless of measurement size
            measurement_size = math.prod(measurement.shape[d] for d in measurement_dims)
            factor = 1.0 / measurement_size if config.normalize else 1.0
            AT_y = op.adjoint(measurement) * factor

            def prox(x: torch.Tensor, tau: float) -> torch.Tensor:
                eta = 1.0 / tau
                rhs = AT_y + eta * x

                def matmul_closure(q: torch.Tensor) -> torch.Tensor:
                    return op.adjoint(op(q)) * factor + eta * q

                return conjugate_gradient(
                    matmul_closure,
                    rhs,
                    x0=x,
                    max_iter=config.iters,
                    eps=config.eps,
                    rel_tol=config.rel_tol,
                )

        elif isinstance(config, PCMProxParameter):
            assert isinstance(op, PhotocurrentMapOp), f"Got {type(op)}."

            STy, m = precompute_pcm_prox_terms(op, y=measurement)

            def prox(x: torch.Tensor, tau: float) -> torch.Tensor:
                rho = 1.0 / tau
                return closed_form_pcm_prox_with_precomputed_terms(
                    op, rho=rho, z=x, ST_y=STy, m_plus_rho=m + rho
                )

        else:
            raise ValueError(f"Unsupported proximal step configuration: {type(config)}")
        return prox


OperatorT = TypeVar("OperatorT", bound=Operator)


class DeepInvLinearPhysics(deepinv.physics.LinearPhysics, Generic[OperatorT]):
    """Adapter from a LION linear operator to DeepInv physics."""

    def __init__(self, op: OperatorT):
        super().__init__()
        self.op = op

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.op(x)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.op.adjoint(y)


class PCMDeepInvPhysics(DeepInvLinearPhysics[PhotocurrentMapOp]):
    """DeepInv wrapper around PhotocurrentMapOp."""

    def prox_l2(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        gamma: float,
        solver="CG",
        max_iter=None,
        tol=None,
        verbose=False,
        **kwargs,
    ) -> torch.Tensor:
        return closed_form_pcm_prox(self.op, rho=1.0 / gamma, y=y, z=z)


class GSPnPDeepInv(BasePnP[GSDenoiser]):
    class GSPnPPrior(RED):
        """Gradient-step denoiser prior."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.explicit_prior = True

        def forward(self, x, *args, **kwargs):
            x = self.denoiser.preprocess(x)
            return self.denoiser.potential(x, *args, **kwargs)

    def __init__(self, physics: DeepInvLinearPhysics, denoiser: GSDenoiser):
        super().__init__(physics.op, denoiser=denoiser)
        self.physics = physics
        self.pgd: PGD | None = None

    def _build_pgd(self, config: GSPnPParameter) -> PGD:
        """Build DeepInv PGD from the LION GS-PnP configuration."""
        stepsize = (
            config.stepsize if config.stepsize is not None else 1.0 / config.lambd
        )
        backtracking = (
            deepinv.optim.BacktrackingConfig(
                gamma=config.gamma_backtracking,
                eta=config.eta_backtracking,
                max_iter=config.max_backtracking_attempts,
            )
            if config.use_backtracking
            else False
        )
        return PGD(
            prior=self.GSPnPPrior(denoiser=self.denoiser),
            g_first=True,
            data_fidelity=L2(),
            sigma_denoiser=config.sigma,
            lambda_reg=config.lambd,
            stepsize=stepsize,
            max_iter=config.iters,
            crit_conv="cost",
            thres_conv=config.eps,
            # LION owns the outer loop below so that failed backtracking attempts
            # do not consume the requested number of accepted GS-PnP iterations.
            early_stop=False,
            backtracking=backtracking,
        )

    @torch.no_grad()
    def reconstruct_sample(self, measurement: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run DeepInv GS-PnP while counting accepted iterations explicitly.

        DeepInv's fixed-point loop counts failed backtracking attempts in ``max_iter``.
        For iteration analysis this can make a nominal 50-iteration run expose fewer
        than 50 accepted reconstructions. Here LION retries rejected backtracking
        attempts and increments the iteration number only after an accepted update.

        When ``continue_after_early_stop`` is enabled, the first iterate satisfying
        the configured convergence threshold is reported through
        ``early_stop_callback`` but reconstruction continues to ``config.iters``.

        If backtracking cannot accept a new update after
        ``max_backtracking_attempts`` reductions, the method returns the last accepted
        reconstruction and reports the stop through ``backtracking_stop_callback``.
        """
        config: GSPnPParameter = cast(GSPnPParameter, kwargs["config"])
        iteration_callback: Callable[[int, torch.Tensor], None] | None = kwargs.get(
            "iteration_callback"
        )
        early_stop_callback: Callable[[int, torch.Tensor], None] | None = kwargs.get(
            "early_stop_callback"
        )
        backtracking_stop_callback = kwargs.get("backtracking_stop_callback")

        self.pgd = self._build_pgd(config)
        pgd = self.pgd
        pgd.has_converged = False

        X = pgd.init_iterate_fn(
            measurement,
            self.physics,
            init=None,
            cost_fn=pgd.fixed_point.iterator.cost_fn,
        )

        first_converged_iteration: int | None = None
        for accepted_iteration in range(1, config.iters + 1):
            failed_backtracking = 0
            backtracking_exhausted = False

            while True:
                X_prev = X
                X_candidate = pgd.fixed_point.single_iteration(
                    X_prev,
                    accepted_iteration - 1,
                    measurement,
                    self.physics,
                )

                if (
                    pgd.fixed_point.backtracking_config is None
                    or pgd.fixed_point.backtracking_check
                ):
                    X = X_candidate
                    break

                failed_backtracking += 1
                if failed_backtracking >= config.max_backtracking_attempts:
                    backtracking_exhausted = True
                    break

            if backtracking_exhausted:
                reconstruction = pgd.get_output(X)
                last_accepted_iteration = accepted_iteration - 1
                current_stepsize = float(pgd.params_algo["stepsize"][0])
                if backtracking_stop_callback is not None:
                    backtracking_stop_callback(
                        last_accepted_iteration,
                        reconstruction,
                        accepted_iteration,
                        failed_backtracking,
                        current_stepsize,
                    )
                break

            reconstruction = pgd.get_output(X)
            if iteration_callback is not None:
                iteration_callback(accepted_iteration, reconstruction)

            converged = (
                config.early_stop
                and accepted_iteration > 2
                and pgd.check_conv_fn(accepted_iteration - 1, X_prev, X)
            )
            if converged and first_converged_iteration is None:
                first_converged_iteration = accepted_iteration
                if early_stop_callback is not None:
                    early_stop_callback(accepted_iteration, reconstruction)
                if not config.continue_after_early_stop:
                    break

        return pgd.get_output(X)
