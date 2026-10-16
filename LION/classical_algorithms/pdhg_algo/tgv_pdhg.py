"""TGV-regularized PDHG solvers implemented in PyTorch.

This module provides:

- ``TgvPdhgSolver``: the low-level primal-dual update loop for second-order TGV.
- ``TgvPdhgTorch``: a wrapper that configures defaults, projection thresholds, and device handling.

Adapted page 17 in "Recovering piecewise smooth multichannel images..." https://unipub.uni-graz.at/obvugroa/content/titleinfo/125370

The implementation follows the TGV-PDHG formulation presented in: "Second Order Total Generalized Variation (TGV)
for MRI" and related derivations from "Recovering piecewise smooth multichannel images by minimising convex functionals
with total generalized variation penalty".
"""

from __future__ import annotations

from typing import Callable

import torch
from zsu.gradops.gradops_torch import GradOpsTorch
from zsu.operators.Operator import Operator
from zsu.pdhg.pdhg_common import PdhgTorch, prepare_lambda


class TgvPdhgSolver:
    """Core TGV-PDHG iteration engine.

    Adapted page 17 in "Recovering piecewise smooth multichannel images..."
    https://unipub.uni-graz.at/obvugroa/content/titleinfo/125370

    The solver updates:

    - primal image variable ``x``,
    - first-order dual/auxiliary variable ``v``,
    - second-order dual variable ``w``,
    - additional primal-like variable ``p``,
    - data-fidelity dual residual ``r``.

    The supplied ``forward_op`` must expose ``forward`` and ``adjoint`` methods.
    Projection callables constrain ``v`` and ``w`` according to TGV regularization
    thresholds.
    """

    def __init__(
        self,
        *,
        sigma: float | torch.Tensor,
        tau: float | torch.Tensor,
        theta: float | torch.Tensor,
        project_lambda_1_v: Callable,
        project_lambda_0_w: Callable,
        convergence_limit: float | torch.Tensor,
    ):
        """Initialize a TGV-PDHG solver with fixed parameters and projections.

        Parameters
        ----------
        sigma : float | torch.Tensor
            Dual step size.
        tau : float | torch.Tensor
            Primal step size.
        theta : float | torch.Tensor
            Extrapolation/over-relaxation parameter.
        project_lambda_1_v : Callable
            Projection callable for first-order dual variable ``v``.
        project_lambda_0_w : Callable
            Projection callable for second-order dual variable ``w``.
        convergence_limit : float | torch.Tensor
            Reference bound derived from operator norms, stored for diagnostics and
            documentation of stability constraints.
        """
        # print(f"convergence_limit: {convergence_limit}")

        # if theta is None:
        #     theta = torch.sigmoid(torch.tensor(10.0))
        # sigma = prepare_param(sigma)
        # tau = prepare_param(tau)
        # theta = prepare_param(theta)

        self.sigma = sigma
        self.tau = tau
        self.theta = theta
        self.project_lambda_1_v = project_lambda_1_v
        self.project_lambda_0_w = project_lambda_0_w
        self.convergence_limit = convergence_limit

        # # while self.sigma * self.tau < convergence_limit:
        # #     self.sigma /= 2
        # #     self.tau /= 2
        # # print(f"sigma: {self.sigma}, tau: {self.tau}")
        # # Tried:
        # [
        #     (0.29, 0.29), #31.130604748354536
        # ]
        # self.sigma = 0.29
        # self.tau = 0.29
        # # self.sigma = 0.1
        # # self.tau = 0.87

    def solve(
        self,
        *,
        measurement: torch.Tensor,  # measured data (e.g. noisy image or only chosen pixels from the noisy images)
        x: torch.Tensor,
        v: torch.Tensor,
        x_bar: torch.Tensor | None = None,
        v_bar: torch.Tensor | None = None,
        p: torch.Tensor,
        w: torch.Tensor,
        num_iters: int,
        forward_op: Operator,
        func: Callable | None = None,
        tqdm: Callable | None = None,
    ) -> torch.Tensor:
        """Run TGV-PDHG iterations and return the reconstructed primal variable.

        Adapted Algorithm 2 in "Second Order Total Generalized Variation (TGV) for MRI"
        by Florian Knoll, Kristian Bredies, Thomas Pock, and Rudolf Stollberger.

        Args:
            measurement : torch.Tensor
                Measured data in the forward-operator codomain.
            x : torch.Tensor
                Initial primal image variable, typically ``(..., H, W)``.
            v : torch.Tensor
                Initial first-order variable, typically ``(..., H, W, 2)``.
            x_bar : torch.Tensor | None, optional
                Initial extrapolated ``x``. If ``None``, initialized as ``x.clone()``.
            v_bar : torch.Tensor | None, optional
                Initial extrapolated ``v``-like state. If ``None``, initialized as
                ``v.clone()``.
            p : torch.Tensor
                Initial auxiliary primal-like variable, typically ``(..., H, W, 2)``.
            w : torch.Tensor
                Initial second-order variable, typically ``(..., H, W, 2, 2)``.
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
            Final primal variable ``x`` after ``num_iters`` updates.

        Note:
        The iteration uses solver attributes ``sigma``, ``tau``, and ``theta``:

        1. First-order dual update: ``v <- v + sigma * (nabla_h(x_bar) - v_bar)``,
           then projection with ``project_lambda_1_v``.
        2. Second-order dual update: ``w <- w + sigma * e_h(v_bar)``,
           then projection with ``project_lambda_0_w``.
        3. Data dual/proximal update:
           ``r <- (r + sigma * (A x_bar - measurement)) / (1 + sigma)``.
        4. Primal update: ``x_next <- x + tau * (div_h_v(v) - A^*(r))`` and
           extrapolation ``x_bar <- x_next + theta * (x_next - x)``.
        5. Auxiliary primal-like update:
           ``p_next <- p + tau * (div_h_w(w) + v)`` and
           ``v_bar <- p_next + theta * (p_next - p)``.
        """
        sigma = self.sigma
        tau = self.tau
        theta = self.theta

        if x_bar is None:
            x_bar = x.clone()
        if v_bar is None:
            v_bar = v.clone()

        r = torch.zeros_like(measurement)  # match the shape of the measured data

        grad_ops = GradOpsTorch()
        if tqdm is not None:
            iterator = tqdm(range(num_iters))
        else:
            iterator = range(num_iters)
        for _ in iterator:
            # First-order dual ascent with correction term, then projection.
            v = v + sigma * (grad_ops.nabla_h(x_bar) - v_bar)
            v = self.project_lambda_1_v(v)  # projection operator

            # Second-order dual ascent in symmetrized-gradient space, then projection.
            w = w + sigma * grad_ops.e_h(v_bar)
            w = self.project_lambda_0_w(w)  # projection operator

            # Data-fidelity dual update followed by quadratic proximal step.
            r = r + sigma * (forward_op.forward(x_bar) - measurement)
            r = r / (1.0 + sigma)  # proximal operator

            # Primal image update and over-relaxation.
            x_next = x + tau * (grad_ops.div_h_v(v) - forward_op.adjoint(r))
            x_bar = x_next + theta * (x_next - x)
            x = x_next

            # Auxiliary primal-like update and over-relaxation used by v-update.
            p_next = p + tau * (grad_ops.div_h_w(w) + v)
            v_bar = p_next + theta * (p_next - p)
            p = p_next

        return x


class TgvPdhgTorch(PdhgTorch):
    """User-facing TGV-PDHG module with defaults and device management."""

    def __init__(
        self,
        *,
        device,
        sigma: float | torch.Tensor = 0.29,  # default value to satisfy condition
        tau: float | torch.Tensor = 0.29,  # default value to satisfy condition
        theta: float
        | torch.Tensor
        | None = None,  # usually sigmoid(10) which is almost 1.0
        lambda_1_v: float | torch.Tensor = 0.1,  # default value to avoid None
        lambda_0_w: float | torch.Tensor = 0.1,  # default value to avoid None
    ):
        """Construct a TGV-PDHG wrapper and initialize regularization thresholds.

        See page 17 in "Recovering piecewise smooth multichannel..." for the algorithm

        Args:
            device
                Device used internally for solver state and computations.
            sigma : float | torch.Tensor, optional
                Dual step size. Defaults to ``0.29``.
            tau : float | torch.Tensor, optional
                Primal step size. Defaults to ``0.29``.
            theta : float | torch.Tensor | None, optional
                Extrapolation parameter. If ``None``, defaults to ``sigmoid(10)``.
            lambda_1_v : float | torch.Tensor, optional
                First-order TGV regularization threshold(s) used in
                ``project_lambda_1_v_clip``.
            lambda_0_w : float | torch.Tensor, optional
                Second-order TGV regularization threshold(s) used in
                ``project_lambda_0_w_clip``.


        Note:
        ``convergence_limit`` is computed from the norm bound used in the cited TGV
        derivation and passed to ``TgvPdhgSolver`` for reference.
        """
        super().__init__()  # Ensure proper initialisation
        self.device = device

        if theta is None:
            theta = torch.sigmoid(
                torch.tensor(10.0)
            ).item()  # Default to sigmoid(10) if not provided

        # Norm of K. See page 16 "Recovering piecewise smooth multichannel..."
        convergence_limit = 1.0 / (
            0.5 * (17 + torch.sqrt(torch.tensor(33.0, device=device)))
        )

        self.pdhg_solver = TgvPdhgSolver(
            sigma=sigma,
            tau=tau,
            project_lambda_1_v=self.project_lambda_1_v_clip,
            project_lambda_0_w=self.project_lambda_0_w_clip,
            theta=theta,
            convergence_limit=convergence_limit,
        )
        self.set_lambda_1_v(lambda_1_v)
        self.set_lambda_0_w(lambda_0_w)

    @property
    def solver(self) -> TgvPdhgSolver:
        """Expose the underlying low-level solver instance.

        Returns:
            Configured solver used by ``forward`` and ``solve``.
        """
        return self.pdhg_solver

    def set_lambda_1_v(self, lambda_1: float | torch.Tensor) -> None:
        """Set and broadcast first-order TGV projection thresholds.

        Args:
            lambda_1 : float | torch.Tensor
                Scalar or map-like threshold(s) for ``v``. The helper
                ``prepare_lambda(..., dim=(-1,))`` unsqueezes for broadcast compatibility
                with the last dual-channel dimension.
        """
        self.lambda_1_v = prepare_lambda(lambda_1, dim=(-1,), device=self.device)
        self.negative_lambda_1_v = -self.lambda_1_v

    def set_lambda_0_w(self, lambda_0: float | torch.Tensor) -> None:
        """Set and broadcast second-order TGV projection thresholds.

        Args:
            lambda_0 : float | torch.Tensor
                Scalar or map-like threshold(s) for ``w``. The helper
                ``prepare_lambda(..., dim=(-1, -2))`` unsqueezes for broadcast
                compatibility with the trailing ``(..., 2, 2)`` structure.
        """
        self.lambda_0_w = prepare_lambda(lambda_0, dim=(-1, -2), device=self.device)
        self.negative_lambda_0_w = -self.lambda_0_w

    def solve(
        self,
        *,
        num_iters: int,
        measurement: torch.Tensor,
        initial_image: torch.Tensor,
        initial_aux_var: torch.Tensor | None = None,
        forward_op: Operator,
        func: Callable | None = None,
        tqdm: Callable | None = None,
    ) -> torch.Tensor:
        """Run reconstruction with automatic device transfer in/out.

        Args:
            num_iters : int
                Number of PDHG iterations.
            measurement : torch.Tensor
                Measured data passed to the solver.
            initial_image : torch.Tensor
                Initial primal image estimate.
            initial_aux_var : torch.Tensor | None, optional
                Optional initial value for ``v`` (first-order variable).
            forward_op : Operator
                Forward model with ``forward`` and ``adjoint`` methods.
            func : Callable | None, optional
                Placeholder for API compatibility with other solver variants.
                Currently unused by the TGV solver.
            tqdm : Callable | None, optional
                Progress-wrapper factory.

        Returns:
            Reconstructed image moved back to ``initial_image.device``.
        """
        old_device = initial_image.device
        initial_image = initial_image.to(self.device)
        if initial_aux_var is not None:
            initial_aux_var = initial_aux_var.to(self.device)
        recon = self.forward(
            measurement=measurement,
            x=initial_image,
            v=initial_aux_var,
            num_iters=num_iters,
            forward_op=forward_op,
            func=func,
            tqdm=tqdm,
        )
        return recon.to(old_device)

    def forward(
        self,
        *,
        num_iters: int,
        measurement: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
        w: torch.Tensor | None = None,
        forward_op: Operator,
        func: Callable | None = None,
        tqdm: Callable | None = None,
    ) -> torch.Tensor:
        """Run TGV-PDHG from an initial state and return the final primal image.

        Args:
            num_iters : int
                Number of PDHG iterations.
            measurement : torch.Tensor
                Measured data in the forward-operator codomain.
            x : torch.Tensor
                Initial primal estimate in image space, typically ``(..., H, W)``.
            v : torch.Tensor | None, optional
                Initial first-order variable, typically ``(..., H, W, 2)``. If ``None``,
                initialized to zeros on ``self.device``.
            p : torch.Tensor | None, optional
                Initial auxiliary primal-like variable, typically ``(..., H, W, 2)``.
                If ``None``, initialized to zeros.
            w : torch.Tensor | None, optional
                Initial second-order variable, typically ``(..., H, W, 2, 2)``. If
                ``None``, initialized to zeros.
            forward_op : Operator
                Forward model with ``forward`` and ``adjoint`` methods.
            func : Callable | None, optional
                Placeholder for API compatibility with other solver variants.
                Currently unused by the TGV solver.
            tqdm : Callable | None, optional
                Progress-wrapper factory.

        Returns:
            Final primal image estimate.

        Notes:
            This method prepares missing state tensors and delegates the update loop to
            ``self.pdhg_solver.solve``.
        """
        device = self.device
        if v is None:
            # Add extra dimension with 2 channels for v since v \in V
            v = torch.zeros((*x.shape, 2), device=device)
        if p is None:
            # Adjusted to match gradient size [n, n, 2]
            p = torch.zeros((*x.shape, 2), device=device)
        if w is None:
            # Adjusted to match gradient size [n, n, 2, 2]
            w = torch.zeros((*v.shape, 2), device=device)

        return self.pdhg_solver.solve(
            num_iters=num_iters,
            measurement=measurement,
            x=x,
            v=v,
            p=p,
            w=w,
            forward_op=forward_op,
            func=func,
            tqdm=tqdm,
        )

    def project_lambda_1_v_clip(self, v):
        """Project first-order variable ``v`` using elementwise clipping.

        Args:
            v : torch.Tensor
                First-order variable in gradient space.

        Returns:
            Clipped variable in ``[-lambda_1_v, lambda_1_v]``.
        """
        return torch.clamp(v, self.negative_lambda_1_v, self.lambda_1_v)

    def project_lambda_0_w_clip(self, w):
        """Project second-order variable ``w`` using elementwise clipping.

        Args:
            w : torch.Tensor
                Second-order variable in symmetrized-gradient space.

        Returns:
            Clipped variable in ``[-lambda_0_w, lambda_0_w]``.
        """
        return torch.clamp(w, self.negative_lambda_0_w, self.lambda_0_w)
