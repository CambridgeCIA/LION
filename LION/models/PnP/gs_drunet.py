# This file is part of LION library
# License : GPL-3
#
# Author: Ferdia Sherry
# Modifications: -
# =============================================================================

from .drunet import DRUNet
from LION.models.LIONmodel import LIONmodel, LIONParameter

from typing import Optional
import torch


class GSDRUNet(LIONmodel):
    def __init__(self, model_parameters: LIONParameter = None):
        if model_parameters is None:
            model_parameters = GSDRUNet.default_parameters()
        super().__init__(model_parameters)
        self._backbone_model = DRUNet(model_parameters)

    @staticmethod
    def default_parameters():
        return LIONParameter(
            in_channels=1,
            out_channels=1,
            int_channels=32,
            kernel_size=(3, 3),
            n_blocks=2,
            use_noise_level=False,
            bias_free=False,
            act="elu",
            enforce_positivity=False,
        )

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Hurault, Samuel, Arthur, Leclaire, Nicolas, Papadakis.")
            print('"Gradient Step Denoiser for Convergent Plug-and-Play."')
            print(
                "\x1b[3mInternational Conference on Learning Representations 2022\x1b[0m."
            )

        elif cite_format == "bib":
            print("@inproceedings{hurault2022gradient,")
            print("title = {Gradient Step Denoiser for Convergent Plug-and-Play},")
            print(
                "booktitle = {International {{Conference}} on {{Learning Representations}}},"
            )
            print(
                "author = {Hurault, Samuel and Leclaire, Arthur and Papadakis, Nicolas},"
            )
            print("year = {2022}")
            print("}")
        else:
            raise ValueError(
                f'`cite_format` "{cite_format}" is not understood, only "MLA" and "bib" are supported'
            )

    def obj_grad(self, x, noise_level: Optional[float] = None):
        x = x.detach()
        x.requires_grad = True
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
