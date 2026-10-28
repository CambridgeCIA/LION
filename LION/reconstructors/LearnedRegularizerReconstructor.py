"""Variational reconstruction with a learned regulariser (e.g. AR, ACR)."""

from __future__ import annotations

import torch
from tqdm import tqdm

from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import LIONmodel
from LION.operators import CTProjectionOp, Operator
from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.utils.parameter import LIONParameter
from LION.models.learned_regularizer.AR import AR


class LearnedRegularizerReconstructor(LIONReconstructor):
    r"""Reconstruct by minimising a variational energy with a learned regulariser,

    .. math::

        E(x) = \tfrac12 \lVert A x - y \rVert_2^2 + \alpha R(x),

    by gradient descent with a backtracking (Armijo) step size, starting from FDK
    for CT operators and from zeros otherwise. Each measurement is reconstructed
    on its own (a batch is reconstructed sample by sample).

    Works with any regulariser R (``model``), however it was trained:
    adversarial (AR), convex (ACR), ... A LIONmodel regulariser is called on a
    batch of one image, normalised as set by its ``set_normalisation()``, and
    may return a (1, 1) value; any other callable is called on the image as it
    is (the operator's domain shape). Its output is summed to a scalar.

    ``LearnedRegularizerReconstructor(operator, model, parameters)``, with
    ``parameters`` from ``default_parameters()``:

    - ``alpha``: regularisation weight (see ``estimate_alpha``);
    - ``n_steps``: number of gradient descent iterations;
    - ``step_size``: initial gradient step;
    - ``backtracking``: factor (< 1) by which the step is reduced until the
      Armijo condition holds. After an accepted step it grows again by
      ``1 / backtracking``.
    """

    def __init__(self, operator, model=None, parameters=None):
        super().__init__(operator, model, parameters)
        backtracking = self.parameters.backtracking
        if not 0 < backtracking < 1:
            raise ValueError(f"backtracking must be in (0, 1), got {backtracking}")

    @staticmethod
    def default_parameters() -> LIONParameter:
        return LIONParameter(alpha=1.0, n_steps=150, step_size=1e-6, backtracking=0.95)

    def regularizer_value(self, x: torch.Tensor) -> torch.Tensor:
        """R(x) as a scalar, for one image ``x`` in the operator's domain shape."""
        if isinstance(self.model, LIONmodel):
            # with the model's normalisation, as in training (identity if unset)
            x_in = self.model.normalise(x.unsqueeze(0))
            return self.model(x_in).sum()
        return self.model(x).sum()

    def energy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """E(x) = 0.5 ||A x - y||^2 + alpha R(x), for one sample."""
        with torch.no_grad():
            data_term = 0.5 * torch.sum((self.op(x) - y) ** 2)
            return data_term + self.parameters.alpha * self.regularizer_value(x)

    def gradient(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """grad E(x) = A^T (A x - y) + alpha grad R(x), for one sample.

        The data term's gradient is computed with the operator's own forward and
        adjoint; only the regulariser (a network) is differentiated by autograd.
        """
        x = x.detach()
        with torch.no_grad():
            data_grad = self.op.adjoint(self.op(x) - y)
        # Autograd is needed even when called under torch.no_grad() (e.g. from a
        # solver's validation loop).
        with torch.enable_grad():
            x.requires_grad_(True)
            # gradient for the image only: nothing accumulates in R's weights
            (reg_grad,) = torch.autograd.grad(self.regularizer_value(x), x)
        return data_grad + self.parameters.alpha * reg_grad

    def initial_guess(self, y: torch.Tensor) -> torch.Tensor:
        """FDK for CT operators, zeros otherwise."""
        if isinstance(self.op, CTProjectionOp):
            return fdk(y, self.op)
        return torch.zeros(self.op.domain_shape, device=y.device, dtype=y.dtype)

    def reconstruct_sample(
        self,
        sino: torch.Tensor,
        x0: torch.Tensor | None = None,
        prog_bar: bool = False,
    ) -> torch.Tensor:
        """Reconstruct one measurement ``sino`` (e.g. (C, angles, detectors)).

        :param x0: Initial image; ``initial_guess(sino)`` if None.
        :param prog_bar: Show a progress bar over the iterations.
        :return: The reconstruction, in the operator's domain shape.
        """
        x = (self.initial_guess(sino) if x0 is None else x0).detach().clone()
        step = self.parameters.step_size
        iterations = range(self.parameters.n_steps)
        if prog_bar:
            iterations = tqdm(iterations, desc="Learned regulariser reconstruction")
        for _ in iterations:
            energy = self.energy(x, sino)
            grad = self.gradient(x, sino)
            grad_norm2 = torch.sum(grad**2)
            if grad_norm2 == 0:
                break
            # Armijo backtracking for the gradient step x - step * grad
            while True:
                candidate = x - step * grad
                if self.energy(candidate, sino) <= energy - 0.5 * step * grad_norm2:
                    break
                step *= self.parameters.backtracking
                if step < torch.finfo(x.dtype).tiny:
                    return x  # no decrease possible: x is (numerically) stationary
            x = candidate
            step /= self.parameters.backtracking  # let the step grow again
        return x

    @staticmethod
    def cite(cite_format="bib"):
        """The variational reconstruction with a learned regulariser of the adversarial regulariser paper; the regulariser has its own."""
        return AR.cite(cite_format)


def estimate_alpha(
    op: Operator, loader, device: torch.device | str | None = None
) -> float:
    r"""Regularisation weight heuristic from the adversarial regulariser paper:
    the mean, over all samples in ``loader``, of :math:`\lVert A^T (A x - y) \rVert`.

    :param op: The forward operator A.
    :param loader: Iterable of (measurement, image) batches.
    :param device: Where to compute; the data's device if None.
    """
    total, count = 0.0, 0
    with torch.no_grad():
        for sinos, targets in loader:
            if device is not None:
                sinos, targets = sinos.to(device), targets.to(device)
            for sino, target in zip(sinos, targets):
                total += torch.linalg.vector_norm(op.adjoint(op(target) - sino)).item()
                count += 1
    if count == 0:
        raise ValueError("estimate_alpha needs at least one sample")
    return total / count
