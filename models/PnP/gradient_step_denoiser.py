# This file is part of LION library
# License : GPL-3
#
# Author: Ferdia Sherry
# Modifications: -
# =============================================================================

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.utils.parameter import LIONParameter

import torch
import importlib


class GSD(LIONmodel):
    def __init__(self, model_parameters: LIONParameter = None):
        super().__init__(model_parameters)

        denoiser = getattr(
            importlib.import_module("LION.models.CNNs"),
            self.model_parameters.backbone_model,
        )

        self._backbone_model = denoiser(self.model_parameters.backbone_params)

    @staticmethod
    def default_parameters():
        backbone_params = LIONModelParameter()
        backbone_params.model_input_type = ModelInputType.IMAGE
        backbone_params.in_channels = 1
        backbone_params.out_channels = 1
        backbone_params.int_channels = 32
        backbone_params.kernel_size = (3, 3)
        backbone_params.n_blocks = 2
        backbone_params.use_noise_level = False
        backbone_params.bias_free = False
        backbone_params.act = "leaky_relu"
        backbone_params.enforce_positivity = False

        params = LIONModelParameter()
        params.backbone_model = "DRUNet"
        params.backbone_params = backbone_params
        params.model_input_type = ModelInputType.IMAGE

        return params

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

    def grad(self, x: torch.Tensor, noise_level: float = None):
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

    def forward(self, x, noise_level: float = None):
        _, grad = self.grad(x, noise_level)
        return x - grad
