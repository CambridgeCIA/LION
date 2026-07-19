"""Thin wrapper around tomosipo's Operator class for linear tomographic projection."""

from __future__ import annotations

import tomosipo as ts
import torch

from LION.operators.Operator import Operator


class CTProjectionOp(Operator):
    """Tomographic projection operator and its adjoint using tomosipo.

    Parameters
    ----------
    ts_operator : tomosipo.Operator.Operator
        Tomosipo operator implementing the tomographic projection and its
        adjoint.
    device : torch.device | str | None, optional
        Device for computations. If None, computations are done on the
        default device.
    """

    @staticmethod
    def cite(cite_format: str = "MLA") -> None:
        """Print citations for the tomosipo and ASTRA projection backends."""
        if cite_format == "MLA":
            print(
                'Hendriksen, Allard A., et al. "Tomosipo: Fast, Flexible, and '
                "Convenient 3D Tomography for Complex Scanning Geometries in "
                'Python." Optics Express, vol. 29, no. 24, article 40494, 2021. '
                "doi:10.1364/OE.439909."
            )
            print()
            print(
                'Van Aarle, Wim, et al. "Fast and Flexible X-Ray Tomography '
                'Using the ASTRA Toolbox." Optics Express, vol. 24, no. 22, '
                "article 25129, 2016. doi:10.1364/OE.24.025129."
            )
        elif cite_format == "bib":
            print(
                r"""@article{hendriksen_tomosipo_2021,
  title = {Tomosipo: Fast, Flexible, and Convenient 3D Tomography for Complex Scanning Geometries in Python},
  author = {Hendriksen, Allard A. and Schut, Dirk and Palenstijn, Willem Jan and
    Vigan{\'o}, Nicola and Kim, Jisoo and Pelt, Dani{\"e}l M. and
    Van Leeuwen, Tristan and Batenburg, K. Joost},
  year = {2021},
  journal = {Optics Express},
  volume = {29},
  number = {24},
  pages = {40494},
  doi = {10.1364/OE.439909}
}

@article{van_aarle_fast_2016,
  title = {Fast and Flexible X-Ray Tomography Using the ASTRA Toolbox},
  author = {Van Aarle, Wim and Palenstijn, Willem Jan and Cant, Jeroen and
    Janssens, Eline and Bleichrodt, Folkert and Dabravolski, Andrei and
    De Beenhouwer, Jan and Batenburg, K. Joost and Sijbers, Jan},
  year = {2016},
  journal = {Optics Express},
  volume = {24},
  number = {22},
  pages = {25129},
  doi = {10.1364/OE.24.025129}
}"""
            )
        else:
            raise ValueError(
                f'`cite_format` "{cite_format}" is not understood, only "MLA" '
                'and "bib" are supported'
            )

    def __init__(
        self,
        ts_operator: ts.Operator.Operator,
        device: torch.device | str | None = None,
    ):
        """Initialize the CTProjectionOp."""
        super().__init__(device=device)
        self._ts = ts_operator

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the forward projection.

        Parameters
        ----------
        x : torch.Tensor
            The input volume dataset to which the forward projection is applied.
        out : torch.Tensor | None, optional
            Optional output holder to store the output projections.
            It is needed for the ``to_autograd`` functionality.

        Returns
        -------
        torch.Tensor
            The projection dataset on which the volume has been forward
            projected.
        """
        return self._ts._fp(x, out=out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward of CTProjectionOp.

        .. note::
            Prefer calling the instance of the CTProjectionOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return self._ts._fp(x)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the backprojection.

        Parameters
        ----------
        y : torch.Tensor
            The input projection dataset to which the backprojection is applied.

        Returns
        -------
        torch.Tensor
            The volume dataset on which the projection dataset has been
            backprojected.
        """
        return self._ts._bp(y)

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying tomosipo operator."""
        return getattr(self._ts, name)

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Return the shape of the volume domain."""
        return self._ts.domain_shape

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of the projection range."""
        return self._ts.range_shape
