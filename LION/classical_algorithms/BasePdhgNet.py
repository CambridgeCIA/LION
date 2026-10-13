"""Base class for unrolled PDHG networks.

This module provides :class:`BasePdhgNet`, an abstract :class:`torch.nn.Module` that encapsulates
the common scaffolding required to run an unrolled Primal-Dual Hybrid Gradient (PDHG) algorithm
as a learnable network.  Concrete subclasses are expected to supply the problem-specific
proximal functionals and operators by implementing the abstract methods below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from mrpro.operators import (
    LinearOperator,
    LinearOperatorMatrix,
    ProximableFunctional,
    ProximableFunctionalSeparableSum,
)

from LION.classical_algorithms.ParameterMapNetwork import ParameterMapNetwork
from LION.classical_algorithms.pdhg_mrpro import pdhg
from LION.classical_algorithms.sapm_config import AdaptivePDHGConfig, SAPMModelConfig
from LION.operators.FiniteDifferenceOp import PadModeIn


class BasePdhgNet(torch.nn.Module, ABC):
    r"""Abstract base class for networks that solve inverse problems via unrolled PDHG.

    The PDHG algorithm solves saddle-point problems of the form

    .. math::

        \\min_x f(Kx) + g(x)

    where *f* and *g* are (proximal-friendly) convex functionals and *K* is a linear operator.
    Subclasses specialise this template for concrete regularisers (e.g. TV, TGV) and forward
    models (e.g. MRI, denoising).

    Attributes:
    ----------
    primal_stepsize : float
        Step size for the primal update (set by :meth:`set_stepsizes`).
    dual_stepsize : float
        Step size for the dual update (set by :meth:`set_stepsizes`).
    n_iterations : int
        Default number of unrolled PDHG iterations used in :meth:`forward`.
    """

    def __init__(
        self,
        config: SAPMModelConfig,
        lambda_map_network: ParameterMapNetwork | None = None,
        forward_op: LinearOperator | None = None,
        example_image: torch.Tensor | None = None,
    ):
        super().__init__()
        self.lambda_map_network = lambda_map_network
        self.n_iterations = config.n_iters
        assert isinstance(
            config.pdhg, AdaptivePDHGConfig
        ), f"Expected AdaptivePDHGConfig, got {type(config.pdhg)}"
        self.dim = config.pdhg.fd_dim
        self.init_ops(config.pdhg.fd_pad_mode)
        self.set_stepsizes(
            forward_op,
            example_image,
            primal_stepsize=config.pdhg.primal_stepsize,
            dual_stepsize=config.pdhg.dual_stepsize,
        )

    @abstractmethod
    def init_ops(self, fd_pad_mode: PadModeIn | Sequence[PadModeIn]) -> None:
        """Initialize any operators required by the PDHG algorithm.

        Subclasses should implement this method to create any linear operators or
        proximal functionals required by the chosen regulariser. For example, a TV
        network would typically create a finite-difference operator here.

        Parameters
        ----------
        fd_pad_mode
            Padding mode for finite-difference operators. Can be a single string or a
            sequence of strings with length equal to the number of dimensions.
        """
        ...

    def set_stepsizes(
        self,
        forward_op: LinearOperator | None = None,
        example_image: torch.Tensor | None = None,
        *,
        primal_stepsize: float | None = None,
        dual_stepsize: float | None = None,
    ) -> None:
        r"""Configure PDHG step sizes and store them on the module.

        When both ``primal_stepsize`` and ``dual_stepsize`` are given they are used
        directly.  Otherwise the operator norm :math:`\\|K\\|` is computed via
        :meth:`compute_stacked_operator_norm` and the missing step size(s) are chosen
        such that :math:`\\tau \\cdot \\sigma = 1 / \\|K\\|^2`, which guarantees
        convergence of the PDHG algorithm.

        Parameters
        ----------
        primal_stepsize:
            Desired primal step size :math:`\\tau`.  ``None`` triggers automatic
            computation from the operator norm.
        dual_stepsize:
            Desired dual step size :math:`\\sigma`.  ``None`` triggers automatic
            computation from the operator norm.
        forward_op:
            Forward operator required for operator-norm estimation when either step size
            is ``None``.  Must be provided together with ``example_image``.
        example_image:
            Example image tensor used to initialise the power-iteration estimate when
            either step size is ``None``.

        Raises:
        ------
        ValueError
            If a step size is ``None`` but ``forward_op`` or ``example_image`` is not provided.
        """
        if primal_stepsize is not None and dual_stepsize is not None:
            self.primal_stepsize = primal_stepsize
            self.dual_stepsize = dual_stepsize
        else:
            if forward_op is None or example_image is None:
                raise ValueError(
                    "If primal or dual stepsize is not provided, "
                    "forward operator and example image for stepsize calculation should be provided"
                )
            op_norm = self.compute_stacked_operator_norm(
                forward_op, self.make_initial_values(example_image)
            )
            print(f"Computed stacked operator norm: {op_norm}")
            target_product = 1.0 / op_norm**2
            # choose primal and dual step size such that their product is 1/|operator|**2
            # to ensure convergence
            if primal_stepsize is None and dual_stepsize is None:
                self.primal_stepsize = 1.0 / op_norm
                self.dual_stepsize = 1.0 / op_norm
            elif primal_stepsize is None and dual_stepsize is not None:
                self.primal_stepsize = target_product / dual_stepsize
                self.dual_stepsize = dual_stepsize
            elif dual_stepsize is None and primal_stepsize is not None:
                self.dual_stepsize = target_product / primal_stepsize
                self.primal_stepsize = primal_stepsize

    @property
    @abstractmethod
    def g(self) -> ProximableFunctionalSeparableSum | ProximableFunctional:
        r"""Proximal functional *g* acting directly on the primal variable.

        This functional is the regularisation term in the objective
        :math:`\\min_x f(Kx) + g(x)`.  Subclasses should return either a
        :class:`mrpro.operators.ProximableFunctional` or a
        :class:`mrpro.operators.ProximableFunctionalSeparableSum`.

        Returns:
        -------
        mrpro.operators.ProximableFunctionalSeparableSum | mrpro.operators.ProximableFunctional
            Proximal-friendly convex functional *g*.
        """
        ...

    @abstractmethod
    def make_f(
        self,
        measurement: torch.Tensor,
        *,
        lambda_reg: torch.Tensor | float | tuple[float, float],
        l2_weight: torch.Tensor | None = None,
    ) -> ProximableFunctionalSeparableSum:
        r"""Construct the data-fidelity functional *f*.

        *f* is the term evaluated as :math:`f(Kx)` in the PDHG objective.  It typically
        encodes the data-consistency constraint, optionally weighted by density-compensation
        factors for non-Cartesian acquisitions.

        Parameters
        ----------
        measurement:
            Observed measurement tensor (e.g. k-space data).
        lambda_reg:
            Scalar or spatially-varying regularisation strength :math:`\\lambda`.
        l2_weight:
            In case of MRI, this is optional density-compensation weights :math:`W^{1/2}` applied as
            :math:`\\|W^{1/2}(Ax - y)\\|_2^2`.  Pass ``None`` for Cartesian data.

        Returns:
        -------
        mrpro.operators.ProximableFunctionalSeparableSum
            Proximal-friendly data-fidelity functional *f*.
        """
        ...

    @abstractmethod
    def make_stacked_operator(self, forward_op: LinearOperator) -> LinearOperatorMatrix:
        """Build the stacked operator *K* used inside PDHG.

        The stacked operator combines the forward (measurement) operator with any
        auxiliary finite-difference or regularisation operators required by the chosen
        penalty functional.

        Parameters
        ----------
        forward_op:
            Linear operator that maps the image to the measurement domain.

        Returns:
        -------
        mrpro.operators.LinearOperatorMatrix
            Column-stacked linear operator :math:`K` whose rows correspond to the
            individual components of the dual variable.
        """
        ...

    def compute_stacked_operator_norm(
        self, forward_op: LinearOperator, initial_values: Sequence[torch.Tensor]
    ) -> float:
        r"""Estimate the spectral norm of the stacked operator.

        Uses the power-iteration method provided by :class:`mrpro.operators.LinearOperatorMatrix`
        to compute :math:`\\|K\\|`, which is needed to choose convergent PDHG step sizes.

        Parameters
        ----------
        forward_op:
            Linear operator that maps the image to the measurement domain.
        initial_values:
            Sequence of initial primal tensors used as the starting point for the
            power-iteration norm estimate.

        Returns:
        -------
        float
            Estimated operator norm :math:`\\|K\\|`.
        """
        stacked_operator = self.make_stacked_operator(forward_op)
        # stacked_op_norm = stacked_operator.operator_norm(*[torch.randn_like(v) for v in initial_values]).amax().item()
        stacked_op_norm = stacked_operator.operator_norm(*initial_values).item()
        return stacked_op_norm

    @abstractmethod
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def postprocess(self, output: torch.Tensor) -> torch.Tensor:
        ...

    def estimate_lambda_map(self, image: torch.Tensor, key: int) -> torch.Tensor:
        r"""Predict a spatially-varying regularisation parameter map from an image.

        Parameters
        ----------
        image:
            Current image estimate tensor used as input to the lambda network.
        key:
            Key for the lambda map network.

        Returns:
        -------
        torch.Tensor
            Estimated regularisation parameter map :math:`\\lambda(x)`, broadcastable
            to the spatial dimensions of ``image``.
        """
        assert (
            self.lambda_map_network is not None
        ), "lambda_map_network must be provided to estimate regularization parameter map"
        return self.postprocess(self.lambda_map_network(self.preprocess(image), key))

    @abstractmethod
    def split_lambda(
        self, lambda_reg: torch.Tensor | float | tuple[float, float]
    ) -> tuple[torch.Tensor | float, torch.Tensor | float | None]:
        ...

    @abstractmethod
    def make_initial_values(
        self, initial_image: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Create initial primal variables for the PDHG solver.

        Typically returns a tuple whose first element is ``initial_image`` together with
        any auxiliary primal variables required by the chosen regulariser (e.g. the
        symmetric-gradient variable in TGV).

        Parameters
        ----------
        initial_image:
            Tensor with the initial image estimate, usually a zero-filled or
            adjoint reconstruction.

        Returns:
        -------
        tuple[torch.Tensor, ...]
            Tuple of initial primal tensors passed to :func:`~zsu.algorithms.pdhg.pdhg`.
        """
        ...

    def forward(
        self,
        *,
        input_image: torch.Tensor | None,
        initial_image: torch.Tensor,
        measurement: torch.Tensor,
        key: int,
        forward_op: LinearOperator,
        lambda_reg: torch.Tensor | float | tuple[float, float] | None = None,
        l2_weight: torch.Tensor | None = None,
        max_iterations: int | None = None,
        primal_stepsize: float | None = None,
        dual_stepsize: float | None = None,
        use_checkpoint: bool = False,
        checkpoint_chunk_size: int | None = None,
        checkpoint_use_reentrant: bool = False,
        use_tqdm: bool = False,
    ) -> torch.Tensor:
        """Reconstruct image using an unrolled PDHG algorithm.

        Parameters
        ----------
        input_image
            Input image tensor to estimate the lambda map
        initial_image
            initial guess of the solution.
        measurement
            measurement tensor of the considered problem.
        key
            key for the lambda map network.
        forward_op
            forward operator that maps the image to the measurement.
        lambda_reg
            regularization parameter to be used. If set to None,
            it is estimated by the lambda_map_network. (can also be a single scalar)
        stacked_operator_norm
            operator norm of the stacked operator. If None, the operator norm
            is calculated internally (might be slow).
        l2_weight
            For MRI, this corresponds to the dcf weights for non-Cartesian data. This can be used to change the problem
            formulation to include the dcf weights in the data fidelity term.
            The weight tensor is understood to be the one that is often referred to as
            W^(1/2) in the literature, such that one has || W^(1/2) (Ax - y) ||_2^2 in
            the data fidelity term.

        Returns:
        -------
            Image reconstructed by the PDGH algorithm to solve the weighted TV problem.
        """
        # if no regularization parameter map is provided, compute it with the network
        if lambda_reg is None:
            assert (
                input_image is not None
            ), "input_image must be provided to estimate lambda map"
            lambda_reg = self.estimate_lambda_map(input_image, key)

        stacked_operator = self.make_stacked_operator(forward_op)

        f = self.make_f(measurement, lambda_reg=lambda_reg, l2_weight=l2_weight)
        initial_values = self.make_initial_values(initial_image)

        return pdhg(
            f=f,
            g=self.g,
            operator=stacked_operator,
            initial_values=initial_values,
            max_iterations=max_iterations
            if max_iterations is not None
            else self.n_iterations,
            primal_stepsize=primal_stepsize
            if primal_stepsize is not None
            else self.primal_stepsize,
            dual_stepsize=dual_stepsize
            if dual_stepsize is not None
            else self.dual_stepsize,
            use_checkpoint=use_checkpoint,
            checkpoint_chunk_size=checkpoint_chunk_size,
            checkpoint_use_reentrant=checkpoint_use_reentrant,
            use_tqdm=use_tqdm,
        )[0]
