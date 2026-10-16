"""Class for Finite Difference Operator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.utils.filters import filter_separable

PadModeIn = Literal["zeros", "circular"]
PadModeTorch = Literal["constant", "circular"]


class FiniteDifferenceOp(LinearOperator):
    """Finite Difference Operator."""

    @staticmethod
    def finite_difference_kernel(
        mode: Literal["central", "forward", "backward"]
    ) -> torch.Tensor:
        """Finite difference kernel.

        Parameters
        ----------
        mode
            String specifying kernel type

        Returns:
        -------
            Finite difference kernel

        Raises:
        ------
        `ValueError`
            If mode is not central, forward, backward or doublecentral
        """
        if mode == "central":
            kernel = torch.tensor((-1, 0, 1)) / 2
        elif mode == "forward":
            kernel = torch.tensor((0, -1, 1))
        elif mode == "backward":
            kernel = torch.tensor((-1, 1, 0))
        else:
            raise ValueError(
                f"mode should be one of (central, forward, backward), not {mode}"
            )
        return kernel

    @staticmethod
    def _to_torch_pad_mode(pad_mode: PadModeIn) -> PadModeTorch:
        """Map public pad_mode values to torch pad modes used by filter_separable."""
        return "constant" if pad_mode == "zeros" else pad_mode

    def __init__(
        self,
        dim: Sequence[int],
        mode: Literal["central", "forward", "backward"] = "central",
        pad_mode: PadModeIn | Sequence[PadModeIn] = "zeros",
    ) -> None:
        """Finite difference operator.

        Parameters
        ----------
        dim
            Dimensions along which finite differences are calculated.
        mode
            Type of finite difference operator.
        pad_mode
            Padding to ensure output has the same size as the input.

            - If a single string is provided, the same mode is used for all `dim`.
            - If a sequence is provided, it must have the same length as `dim`,
              and each entry is used for the corresponding dimension.
        """
        super().__init__()
        self.dim = tuple(dim)
        self.kernel = self.finite_difference_kernel(mode)

        if isinstance(pad_mode, str):
            pad_mode_seq = (pad_mode,) * len(self.dim)
        else:
            pad_mode_seq = tuple(pad_mode)
            if len(pad_mode_seq) != len(self.dim):
                raise ValueError(
                    "pad_mode must be a single string or have the same length as dim. "
                    f"Got len(pad_mode)={len(pad_mode_seq)} and len(dim)={len(self.dim)}."
                )

        self.pad_modes: tuple[PadModeTorch, ...] = tuple(
            self._to_torch_pad_mode(m) for m in pad_mode_seq
        )

    def __call__(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor,]:
        """Apply forward finite difference operation.

        Calculates finite differences of the input tensor `x` along the dimensions
        specified during initialization. The results for each dimension are stacked
        along the first dimension of the output tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns:
        -------
            Tensor containing the finite differences of `x` along the specified
            dimensions, stacked along the first dimension.
        """
        return super().__call__(x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor,]:
        """Apply forward of FiniteDifferenceOp.

        .. note::
            Prefer calling the instance of the FiniteDifferenceOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return (
            torch.stack(
                [
                    filter_separable(
                        x,
                        (self.kernel,),
                        dim=(dim_i,),
                        pad_mode=pad_mode_i,
                        pad_value=0.0,
                    )
                    for dim_i, pad_mode_i in zip(self.dim, self.pad_modes, strict=True)
                ]
            ),
        )

    def adjoint(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor,]:
        """Apply adjoint finite difference operation.

        This operation is the adjoint of the forward finite difference calculation.
        It takes a tensor `y` (which is assumed to be the output of the forward pass,
        i.e., finite differences stacked along the first dimension) and computes
        the sum of the adjoints of the individual directional finite difference operations.

        Parameters
        ----------
        y
            Input tensor, representing finite differences stacked along the first dimension.
            The size of the first dimension must match the number of dimensions
            specified for the operator.

        Returns:
        -------
            Result of the adjoint finite difference operation.

        Raises:
        ------
        ValueError
            If the first dimension of `y` does not match the number of
            dimensions along which finite differences were calculated.
        """
        if y.shape[0] != len(self.dim):
            raise ValueError(
                "First dimension of input tensor has to match the number of finite difference directions."
            )
        return (
            torch.sum(
                torch.stack(
                    [
                        filter_separable(
                            yi,
                            (torch.flip(self.kernel, dims=(-1,)),),
                            dim=(dim_i,),
                            pad_mode=pad_mode_i,
                            pad_value=0.0,
                        )
                        for dim_i, pad_mode_i, yi in zip(
                            self.dim, self.pad_modes, y, strict=True
                        )
                    ]
                ),
                dim=0,
            ),
        )
