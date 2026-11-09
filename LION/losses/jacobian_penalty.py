# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Penalties on a denoiser's Jacobian, for Plug-and-Play methods that converge.

PnP convergence proofs ask something of the denoiser D: that it be firmly
nonexpansive (2D - I nonexpansive; Pesquet et al. 2021), or that its residual
I - D, the gradient of the potential of a gradient-step denoiser, be
L-Lipschitz with L < 1 (Hurault et al. 2022). A network isn't, so it is
trained with a penalty on the spectral norm of the Jacobian of that map,
estimated by power iterations. They are model regularisations: a solver
adds one to its loss when it is its ``model_regularization`` (off by
default), called as ``penalty(model, step)`` with the mini-batch's ``step``
(``LIONsolver``).
"""

import torch
import torch.nn as nn

from LION.utils.citation import citation


def jacobian_spectral_norm(f, x: torch.Tensor, power_iterations: int = 50):
    """The spectral norm of the Jacobian J of ``f`` at each image of the batch
    ``x`` (``f`` maps each image on its own), estimated with
    ``power_iterations`` iterations of the power method on JᵀJ.
    Differentiable with respect to ``f``'s parameters (as ``||Jᵀv||``, v the
    last left singular vector: second derivatives only, which e.g. batch
    normalisation supports)."""
    x = x.detach().requires_grad_(True)
    y = f(x)
    # Jᵀw is linear in w: differentiating <Jᵀw, u> in w gives Ju
    w = torch.zeros_like(y, requires_grad=True)
    (jtw,) = torch.autograd.grad(y, x, w, create_graph=True)

    def unit(v):
        norm = v.flatten(1).norm(dim=1).clamp_min(1e-12)
        return v / norm.reshape((-1,) + (1,) * (v.dim() - 1))

    u = unit(torch.randn_like(x))
    for _ in range(power_iterations):
        (ju,) = torch.autograd.grad(jtw, w, u, retain_graph=True)
        (jtju,) = torch.autograd.grad(y, x, ju, retain_graph=True)
        u = unit(jtju.detach())
    (ju,) = torch.autograd.grad(jtw, w, u, retain_graph=True)
    v = unit(ju.detach())
    (jtv,) = torch.autograd.grad(y, x, v, create_graph=True)
    return jtv.flatten(1).norm(dim=1)


class FirmlyNonexpansivePenalty(nn.Module):
    """Pesquet et al. (2021): ``weight * max(||J_Q(x̃)||², 1 - epsilon)``, with
    Q = 2D - I and x̃ a random point between each clean image and its
    denoised image (``x̃ = r x + (1 - r) D(y)``, r uniform in [0, 1]).

    Their settings: ``weight`` 1e-5, ``epsilon`` 0.05, after training
    without it (pretraining). They estimate the norm with the power method
    (their Appendix C); 50 iterations is Hurault et al.'s number.
    """

    def __init__(
        self, weight: float = 1e-5, epsilon: float = 0.05, power_iterations: int = 50
    ):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.power_iterations = power_iterations

    def forward(self, model, step) -> torch.Tensor:
        """The penalty for a mini-batch ``step``: the denoiser
        ``step.forward``, its clean images ``step.target``, its outputs
        ``step.output``."""
        if step.target is None:
            raise ValueError(
                "FirmlyNonexpansivePenalty needs the clean images: a solver "
                "with targets (e.g. GaussianDenoiserSolver, SupervisedSolver)"
            )
        denoise, clean, denoised = step.forward, step.target, step.output
        r = torch.rand(clean.shape[0], device=clean.device)
        r = r.reshape((-1,) + (1,) * (clean.dim() - 1))
        point = r * clean + (1 - r) * denoised.detach()
        norm = jacobian_spectral_norm(
            lambda x: 2 * denoise(x) - x, point, self.power_iterations
        )
        return self.weight * torch.clamp(norm**2, min=1 - self.epsilon).mean()

    @staticmethod
    def cite(cite_format="bib"):
        """Where the penalty comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{pesquet2021learning,
            title={Learning maximally monotone operators for image recovery},
            author={Pesquet, Jean-Christophe and Repetti, Audrey and Terris, Matthieu and Wiaux, Yves},
            journal={SIAM Journal on Imaging Sciences},
            volume={14},
            number={3},
            pages={1206--1237},
            year={2021},
            doi={10.1137/20M1387961}
            }
            """,
            mla="""
            Pesquet, Jean-Christophe, et al. "Learning maximally monotone
            operators for image recovery." SIAM Journal on Imaging Sciences
            14.3 (2021): 1206-1237.
            """,
        )


class ProximalDenoiserPenalty(nn.Module):
    """Hurault et al. (2022): ``weight * max(||J_{I-D}(y)||, 1 - epsilon)`` at
    the noisy images y: the Hessian of a gradient-step denoiser's potential
    (I - D = ∇g), so that D is a proximal operator.

    Their settings: ``weight`` (μ) 1e-3 or 1e-2, ``epsilon`` 0.1, 50 power
    iterations, fine-tuning a trained gradient-step denoiser for 10 epochs.
    """

    def __init__(
        self, weight: float = 1e-3, epsilon: float = 0.1, power_iterations: int = 50
    ):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.power_iterations = power_iterations

    def forward(self, model, step) -> torch.Tensor:
        """The penalty for a mini-batch ``step``: the denoiser
        ``step.forward`` at its inputs ``step.input``."""
        denoise = step.forward
        norm = jacobian_spectral_norm(
            lambda x: x - denoise(x), step.input, self.power_iterations
        )
        return self.weight * torch.clamp(norm, min=1 - self.epsilon).mean()

    @staticmethod
    def cite(cite_format="bib"):
        """Where the penalty comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @inproceedings{hurault2022proximal,
            title={Proximal Denoiser for Convergent Plug-and-Play Optimization with Nonconvex Regularization},
            author={Hurault, Samuel and Leclaire, Arthur and Papadakis, Nicolas},
            booktitle={Proceedings of the 39th International Conference on Machine Learning},
            series={Proceedings of Machine Learning Research},
            volume={162},
            pages={9483--9505},
            year={2022}
            }
            """,
            mla="""
            Hurault, Samuel, Arthur Leclaire, and Nicolas Papadakis. "Proximal
            denoiser for convergent plug-and-play optimization with nonconvex
            regularization." Proceedings of the 39th International Conference
            on Machine Learning, PMLR 162 (2022): 9483-9505.
            """,
        )
