"""TV-regularized PDHG solvers implemented in PyTorch.

This module provides:

- ``TvPdhgSolver``: the low-level primal-dual iteration loop.
- ``TvPdhgTorch``: a higher-level wrapper that manages defaults, lambda projection
  thresholds, and device placement.

The iterative scheme follows the TV variant of PDHG used in:
"Learning Regularization Parameter-Maps for Variational Image Reconstruction using
Deep Neural Networks and Algorithm Unrolling" (Kofler et al., 2023).
"""

from __future__ import annotations

from typing import Callable

import torch

from LION.classical_algorithms.pdhg_algo.gradops_torch import GradOpsTorch
from LION.classical_algorithms.pdhg_algo.pdhg_common import PdhgTorch, prepare_lambda
from LION.operators.Operator import Operator


class TvPdhgSolver:
    """Core TV-PDHG iteration engine.

    The solver updates:

    - a primal variable ``x`` in image space,
    - a dual TV variable ``v`` in gradient space,
    - a dual data-fidelity residual ``r`` in measurement space.

    The supplied ``forward_op`` must provide ``forward`` and ``adjoint`` methods.
    The supplied ``project_lambda_v`` must project/clamp the TV dual variable to the
    regularization-constrained feasible set.
    """

    def __init__(
        self,
        *,
        sigma: float | torch.Tensor,
        tau: float | torch.Tensor,
        theta: float | torch.Tensor,
        project_lambda_v: Callable,
    ):
        """Initialize a TV-PDHG solver with fixed algorithm parameters.

        Parameters
        ----------
        sigma : float | torch.Tensor
            Dual step size.
        tau : float | torch.Tensor
            Primal step size.
        theta : float | torch.Tensor
            Extrapolation/over-relaxation parameter.
        project_lambda_v : Callable
            Projection callable applied to the updated TV dual variable ``v``.
        """
        self.sigma = sigma
        self.tau = tau
        self.theta = theta
        self.project_lambda_v = project_lambda_v

    def solve(
        self,
        *,
        measurement: torch.Tensor,  # measured data (e.g. noisy image or only chosen pixels from the noisy images)
        x: torch.Tensor,
        v: torch.Tensor,
        x_bar: torch.Tensor | None = None,
        num_iters: int,
        forward_op: Operator,
        func: Callable | None = None,
        tqdm: Callable | None = None,
    ):
        """Run TV-PDHG iterations and return the reconstructed primal variable.

        Adapted from Algorithm 2 from
        'Learning Regularization Parameter-Maps for Variational
        Image Reconstruction using Deep Neural Networks and
        Algorithm Unrolling', 2023,
        by Andreas Kofler et al.

        Parameters
        ----------
        measurement : torch.Tensor
            Measured data in the forward-operator codomain. The residual dual
            variable ``r`` is initialized with the same shape.
        x : torch.Tensor
            Initial primal variable (image-domain estimate), typically shape
            ``(..., H, W)``.
        v : torch.Tensor
            Initial TV dual variable in gradient space, typically shape
            ``(..., H, W, 2)``.
        x_bar : torch.Tensor | None, optional
            Initial extrapolated primal variable. If ``None``, a clone of ``x`` is
            used.
        num_iters : int
            Number of PDHG iterations.
        forward_op : Operator
            Forward model with ``forward`` and ``adjoint`` methods.
        func : Callable | None, optional
            Placeholder for API compatibility with other solver variants.
            Currently unused in this implementation.
        tqdm : Callable | None, optional
            Progress-wrapper factory. If provided, used as ``tqdm(range(num_iters))``.

        Returns:
        -------
        torch.Tensor
            Final primal variable ``x`` after ``num_iters`` updates.

        Notes:
        -----
        The iteration uses solver attributes ``sigma``, ``tau``, and ``theta``:

        1. Dual TV ascent: ``v <- v + sigma * nabla_h(x_bar)`` then projection.
        2. Data dual update + proximal step:
           ``r <- (r + sigma * (A x_bar - measurement)) / (1 + sigma)``.
        3. Primal update:
           ``x_next <- x + tau * (div_h(v) - A^*(r))``.
        4. Extrapolation: ``x_bar <- x_next + theta * (x_next - x)``.

        The sign convention in the primal update intentionally matches the current
        implementation and is known to be opposite to the cited paper notation.
        """
        sigma = self.sigma
        tau = self.tau
        theta = self.theta

        if x_bar is None:
            x_bar = x.clone()

        r = torch.zeros_like(measurement)  # match the shape of the measured data

        grad_ops = GradOpsTorch()
        if tqdm is not None:
            iterator = tqdm(range(num_iters))
        else:
            iterator = range(num_iters)
        for _ in iterator:
            # Dual TV ascent in gradient space, then projection onto |v| <= lambda.
            v = v + sigma * grad_ops.nabla_h(x_bar)
            v = self.project_lambda_v(v)

            # Data-fidelity dual update followed by the proximal map of the
            # quadratic fidelity term.
            r = r + sigma * (forward_op.forward(x_bar) - measurement)
            r = r / (1.0 + sigma)  # proximal operator ?

            # NOTE: The plus and minus signs are opposite to the paper.
            # If we switch any of the signs, the algorithm will not work.
            # Primal update using TV divergence and data-term adjoint contribution.
            x_next = x + tau * (grad_ops.div_h_v(v) - forward_op.adjoint(r))
            # Over-relaxation/extrapolation for the next dual step.
            x_bar = x_next + theta * (x_next - x)
            x = x_next

        return x


class TvPdhgTorch(PdhgTorch):
    """User-facing TV-PDHG module with defaults and device management."""

    def __init__(
        self,
        *,
        device,
        sigma: float | torch.Tensor | None = None,
        tau: float | torch.Tensor | None = None,
        theta: float | torch.Tensor | None = None,
        lambda_reg: float | torch.Tensor = 0.1,  # default value to avoid None
    ):
        """Construct a TV-PDHG wrapper and initialize projection thresholds.

        Parameters
        ----------
        device
            Device used internally for solver state and computations.
        sigma : float | torch.Tensor | None, optional
            Dual step size. If ``None``, defaults to ``sigmoid(10) / sqrt(13)``.
        tau : float | torch.Tensor | None, optional
            Primal step size. If ``None``, defaults to ``sigmoid(10) / sqrt(13)``.
        theta : float | torch.Tensor | None, optional
            Extrapolation parameter. If ``None``, defaults to ``sigmoid(10)``.
        lambda_reg : float | torch.Tensor, optional
            TV regularization threshold(s) used by the dual projection
            ``project_lambda_v_clip``.

        Notes:
        -----
        ``L = sqrt(13)`` and ``sigmoid(10)`` follow the same initialization
        convention used in this codebase for stable PDHG defaults.
        """
        super().__init__()  # Ensure proper initialisation
        self.device = device

        # See "Learning Regularisation Parameter-Map..."
        self.L = torch.sqrt(torch.tensor(13.0, device=device))
        self.sigmoid_10 = torch.sigmoid(torch.tensor(10.0, device=device))

        # constants depending on the operators
        if sigma is None:
            sigma = self.sigmoid_10 / self.L
        if tau is None:
            tau = self.sigmoid_10 / self.L
        if theta is None:
            theta = self.sigmoid_10

        self.pdhg_solver = TvPdhgSolver(
            sigma=sigma,
            tau=tau,
            theta=theta,
            project_lambda_v=self.project_lambda_v_clip,
        )
        self.set_lambda_v(lambda_reg)

    @property
    def solver(self):
        """Expose the underlying low-level solver instance.

        Returns:
        -------
        TvPdhgSolver
            Configured solver used by ``forward`` and ``solve``.
        """
        return self.pdhg_solver

    def set_lambda_v(self, lambda_v):
        """Set and broadcast TV projection thresholds for the dual variable ``v``.

        Parameters
        ----------
        lambda_v : float | torch.Tensor
            Scalar or map-like regularization threshold(s). The helper
            ``prepare_lambda(..., dim=(-1,))`` unsqueezes to make the tensor
            broadcastable with the last dual-channel dimension of ``v``.
        """
        self.lambda_v = prepare_lambda(lambda_v, dim=(-1,), device=self.device)
        self.negative_lambda_v = -self.lambda_v

    def solve(
        self,
        *,
        num_iters: int,
        measurement: torch.Tensor,
        initial_image: torch.Tensor,
        forward_op: Operator,
        func: Callable | None = None,
        tqdm: Callable | None = None,
    ) -> torch.Tensor:
        """Run reconstruction with automatic device transfer in/out.

        Wrapper for the forward method. Only adding the device management.

        Args:
            num_iters : Number of PDHG iterations.
            measurement : Measured data passed to the solver.
            initial_image : Initial primal image estimate.
            forward_op : Forward model with ``forward`` and ``adjoint`` methods.
            func : Placeholder for API compatibility with other solver variants.
                Currently unused by the TV solver.
            tqdm : Progress-wrapper factory.

        Returns:
            Reconstructed image moved back to ``measurement.device``.
        """
        old_device = measurement.device
        return self.forward(
            measurement=measurement.to(self.device),
            x=initial_image.to(self.device),
            num_iters=num_iters,
            forward_op=forward_op,
            func=func,
            tqdm=tqdm,
        ).to(old_device)

    def forward(
        self,
        *,
        num_iters: int,
        measurement: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor | None = None,
        forward_op: Operator,
        func: Callable | None = None,
        tqdm: Callable | None = None,
    ) -> torch.Tensor:
        """Run TV-PDHG from an initial state and return the final primal image.

        Args:
            num_iters : Number of PDHG iterations.
            measurement : Measured data in the forward-operator codomain.
            x : Initial primal estimate in image space, e.g. initial guess for the denoised image,
                typically ``(..., H, W)`` (expect 2D tensor of shape ``(H, W)`` for now).
            v : Initial TV dual variable, typically ``(..., H, W, 2)``. If ``None``,
                initialized to zeros on ``self.device``.
            forward_op : Forward model with ``forward`` and ``adjoint`` methods.
            func : Placeholder for API compatibility with other solver variants.
                Currently unused by the TV solver.
            tqdm : Progress-wrapper factory.

        Returns:
            Final primal image estimate.

        Note:
            This method delegates the update loop to ``self.pdhg_solver.solve`` after
            preparing default dual state ``v`` when required.
        """
        device = self.device

        if v is None:
            # Add extra dimension with 2 channels for v since v \in V
            v = torch.zeros((*x.shape, 2), device=device)

        return self.pdhg_solver.solve(
            num_iters=num_iters,
            measurement=measurement,
            x=x,
            v=v,
            forward_op=forward_op,
            func=func,
            tqdm=tqdm,
        )

    def project_lambda_v_clip(self, v: torch.Tensor) -> torch.Tensor:
        """Project TV dual variable ``v`` using elementwise clipping.

        Args:
            v: Dual variable in gradient space.

        Returns:
            Clipped variable in ``[-lambda_v, lambda_v]`` with broadcasting from ``self.lambda_v``.
        """
        return torch.clamp(v, self.negative_lambda_v, self.lambda_v)
