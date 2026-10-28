# This file is part of LION library
# License : GPL-3
#
# Author: Ferdia Sherry
# Modifications: -
# =============================================================================

from ..CNNs.drunet import DRUNet
from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.utils.parameter import LIONParameter

from typing import Optional
import torch
from LION.utils.citation import citation


class GSDRUNet(LIONmodel):
    def __init__(self, model_parameters: LIONParameter = None):
        if model_parameters is None:
            model_parameters = GSDRUNet.default_parameters()
        super().__init__(model_parameters)
        self._backbone_model = DRUNet(model_parameters)

    @staticmethod
    def default_parameters():
        params = LIONModelParameter()
        params.model_input_type = ModelInputType.IMAGE
        params.in_channels = 1
        params.out_channels = 1
        params.int_channels = 32
        params.kernel_size = (3, 3)
        params.n_blocks = 2
        params.use_noise_level = False
        params.bias_free = False
        params.act = "leaky_relu"
        params.enforce_positivity = False
        return params

    @staticmethod
    def cite(cite_format="bib"):
        """Where GSDRUNet comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @inproceedings{hurault2022gradient,
            title = {Gradient Step Denoiser for Convergent Plug-and-Play},
            booktitle = {International {{Conference}} on {{Learning Representations}}},
            author = {Hurault, Samuel and Leclaire, Arthur and Papadakis, Nicolas},
            year = {2022}
            }
            """,
            mla="""
            Hurault, Samuel, Arthur, Leclaire, Nicolas, Papadakis.
            "Gradient Step Denoiser for Convergent Plug-and-Play."
            International Conference on Learning Representations 2022.
            """,
        )

    def obj_grad(self, x, noise_level: Optional[float] = None):
        """g(x) = 0.5 ||x - N(x)||^2 and its gradient with respect to x."""
        # The denoiser is defined through a gradient, so autograd is needed even
        # when called under torch.no_grad() (validation, testing, PnP).
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            obj = 0.5 * torch.sum((self._backbone_model(x, noise_level) - x) ** 2)
            grad = torch.autograd.grad(
                obj,
                x,
                torch.ones_like(obj),
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
        return obj, grad

    def forward(self, x, noise_level: Optional[float] = None):
        _, grad = self.obj_grad(x, noise_level)
        return x - grad
