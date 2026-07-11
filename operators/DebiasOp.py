"""Debiasing least squares operator on the support of w."""

from __future__ import annotations

from tabnanny import verbose

import torch
from tqdm import tqdm

from LION.operators.Operator import Operator
from LION.utils.math import power_method


class DebiasOp(Operator):
    """Debiasing least squares operator on the support of w.

    Parameters
    ----------
    operator : Operator
        Forward operator A.
    y : torch.Tensor
        Measurements, shape (M,).
    w : torch.Tensor
        l1 minimiser, shape (Nw,).
    support_tol : float
        Threshold defining nonzero support.

    References
    ----------
    .. [Koutsourakis2021] G. Koutsourakis, A. Thompson, and J. C. Blakesley,
        "Toward Megapixel Resolution Compressed Sensing Current Mapping of
        Photovoltaic Devices Using Digital Light Processing", Solar RRL,
        5(11):2100467, 2021. doi:10.1002/solr.202100467
    """

    def __init__(
        self,
        op: Operator,
        y: torch.Tensor,
        w: torch.Tensor,
        support_tol: float = 1e-3,
    ):
        self.op = op
        self.y = y
        self.w = w
        self.support_tol = support_tol

        self.support = torch.nonzero(
            torch.abs(w) > support_tol, as_tuple=False
        ).squeeze(1)
        self.device = op.device

    def __call__(self, v: torch.Tensor, out=None) -> torch.Tensor:
        """Apply the forward projection on the support.

        Parameters
        ----------
        v : torch.Tensor
            Input vector on the support, shape (Ns,).
        out : None
            Legacy for tomosipo ``to_autograd``. Just ignore.

        Returns
        -------
        torch.Tensor
            Output measurements, shape (M,).
        """
        return self.forward(v)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Apply the forward projection on the support.

        .. note::
            Prefer calling the instance of the DebiasOp operator as ``operator(v)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        w_full = torch.zeros_like(self.w)
        w_full[self.support] = v
        return self.op.forward(w_full)

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint projection on the support.

        Parameters
        ----------
        r : torch.Tensor
            Input measurements, shape (M,).

        Returns
        -------
        torch.Tensor
            Output vector on the support, shape (Ns,).
        """
        g_full = self.op.adjoint(r)
        return g_full[self.support]

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Return the shape of the support domain."""
        return (self.support.numel(),)

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of the measurement range."""
        return self.op.range_shape


def debias_ls(
    op: Operator,
    y: torch.Tensor,
    w: torch.Tensor,
    support_tol: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-5,
    progress_bar: bool = False,
) -> torch.Tensor:
    """Debiasing least squares on the support of w.

    Parameters
    ----------
    op : Operator
        Forward operator A.
    y : torch.Tensor
        Measurements, shape (M,).
    w : torch.Tensor
        l1 minimiser, shape (Nw,).
    support_tol : float
        Threshold defining nonzero support.
    max_iter : int
        Maximum number of gradient descent iterations.
    tol : float
        Relative stopping threshold.
    progress_bar : bool
        Whether to show a progress bar. Default is False.

    Returns
    -------
    w_deb : torch.Tensor
        Debiased coefficient vector, shape (Nw,).
    """
    device = w.device

    support = torch.nonzero(torch.abs(w) > support_tol, as_tuple=False).squeeze(1)
    if support.numel() == 0:
        return w.clone()

    op_s = DebiasOp(op, y, w, support_tol=support_tol)

    # Check: Why is the squaring needed here?
    L = power_method(op_s, device=device) ** 2
    step = 1.0 / (L + 1e-12)

    v = w[support].clone()

    iterator = range(max_iter)
    if progress_bar:
        iterator = tqdm(iterator, desc="Debiasing LS")
    for _ in iterator:
        r = op_s(v) - y
        grad = op_s.adjoint(r)
        v_next = v - step * grad

        rel_change = torch.norm(v_next - v) / (torch.norm(v) + 1e-8)
        v = v_next

        if verbose:
            data_term = 0.5 * torch.norm(op_s(v) - y).pow(2).item()
            print(
                f"Debiasing LS  f = {data_term:.4e}  "
                f"rel_change = {rel_change.item():.2e}  tol = {tol:.2e}  "
                f"rel_change < tol: {rel_change.item() < tol}"
            )

        if rel_change.item() < tol:
            break

    w_deb = torch.zeros_like(w)
    w_deb[support] = v
    return w_deb
