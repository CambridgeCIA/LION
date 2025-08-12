# This file is part of LION library
# License : GPL-3
#
# Author: Ferdia Sherry
# Modifications: -
# =============================================================================

from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.utils.parameter import LIONParameter


from inspect import getmembers, isfunction
import torch


class DnCNN(LIONmodel):
    def __init__(self, model_parameters: LIONParameter = None):
        super().__init__(model_parameters)

        if model_parameters.act.lower() in dict(
            getmembers(torch.nn.functional, isfunction)
        ):
            self._act = torch.nn.functional.__dict__[model_parameters.act]
        else:
            raise ValueError(
                f"`torch.nn.functional` does not export a function '{model_parameters.act}'."
            )
        self.lift = torch.nn.Conv2d(
            model_parameters.in_channels,
            model_parameters.int_channels,
            model_parameters.kernel_size,
            padding=tuple(k // 2 for k in model_parameters.kernel_size),
            bias=not model_parameters.bias_free,
        )
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    model_parameters.int_channels,
                    model_parameters.int_channels,
                    model_parameters.kernel_size,
                    padding=tuple(k // 2 for k in model_parameters.kernel_size),
                    bias=not model_parameters.bias_free,
                )
                for _ in range(model_parameters.blocks - 2)
            ]
        )
        if model_parameters.batch_normalisation:
            self.bns = torch.nn.ModuleList(
                [
                    torch.nn.BatchNorm2d(
                        model_parameters.int_channels,
                        affine=not model_parameters.bias_free,
                    )
                    for _ in range(model_parameters.blocks - 2)
                ]
            )
        self.project = torch.nn.Conv2d(
            model_parameters.int_channels,
            model_parameters.in_channels,
            model_parameters.kernel_size,
            padding=tuple(k // 2 for k in model_parameters.kernel_size),
            bias=not model_parameters.bias_free,
        )

    @staticmethod
    def default_parameters():
        params = LIONModelParameter()
        params.model_input_type = ModelInputType.IMAGE
        params.in_channels = 1
        params.int_channels = 64
        params.kernel_size = (3, 3)
        params.blocks = 20
        params.residual = True
        params.bias_free = False
        params.act = "leaky_relu"
        params.enforce_positivity = False
        params.batch_normalisation = True
        return params

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Zhang, Kai, Wangmeng, Zuo, Yunjin, Chen, Deyu, Meng, Lei, Zhang.")
            print(
                '"Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising".'
            )
            print("\x1b[3mIEEE Transactions on Image Processing\x1b[0m")
            print("26. 7(2017): 3142–3155.")
        elif cite_format == "bib":
            print("@article{zhang2017beyond,")
            print(
                "title = {Beyond a {{Gaussian Denoiser}}: {{Residual Learning}} of {{Deep CNN}} for {{Image Denoising}}},"
            )
            print(
                "author = {Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},"
            )
            print("year = {2017},")
            print("journal = {IEEE Transactions on Image Processing},")
            print("volume = {26},")
            print("number = {7},")
            print("pages = {3142--3155},")
            print("issn = {1057-7149},")
            print("doi = {10.1109/TIP.2017.2662206}")
            print("}")
        else:
            raise ValueError(
                f'`cite_format` "{cite_format}" is not understood, only "MLA" and "bib" are supported'
            )

    def forward(self, x):
        z = self._act(self.lift(x))
        if self.model_parameters.batch_normalisation:
            for conv, bn in zip(self.convs, self.bns):
                z = self._act(bn(conv(z)))
        else:
            for conv in self.convs:
                z = self._act(conv(z))

        if self.model_parameters.residual:
            z = x - self.project(z)
        else:
            z = self.project(z)
        if self.model_parameters.enforce_positivity:
            return torch.nn.functional.relu(z)
        else:
            return z
