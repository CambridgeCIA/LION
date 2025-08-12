# This file is part of LION library
# License : GPL-3
#
# Author: Ferdia Sherry
# Modifications: -
# =============================================================================

from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.utils.parameter import LIONParameter

from inspect import getmembers, isfunction
from typing import Optional

import torch


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        n_channels=64,
        kernel_size=(3, 3),
        bias_free=True,
        act=torch.nn.functional.leaky_relu,
    ):
        super().__init__()
        self.act = act
        self.conv1 = torch.nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size,
            padding=tuple(k // 2 for k in kernel_size),
            bias=not bias_free,
        )

        self.conv2 = torch.nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size,
            padding=tuple(k // 2 for k in kernel_size),
            bias=not bias_free,
        )

    def forward(self, x):
        res = self.conv2(self.act(self.conv1(x)))
        return x + res


def downsample_strideconv(
    in_channels=1,
    out_channels=64,
    kernel_size=2,
    stride=2,
    bias_free=True,
):
    return torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=not bias_free,
    )


def upsample_convtranspose(
    in_channels=64,
    out_channels=1,
    kernel_size=2,
    stride=2,
    bias_free=True,
):
    return torch.nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=not bias_free,
    )


class DRUNet(LIONmodel):
    def __init__(self, model_parameters: LIONParameter = None):
        if model_parameters is None:
            model_parameters = DRUNet.default_parameters()
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
            (
                model_parameters.in_channels + 1
                if model_parameters.use_noise_level
                else model_parameters.in_channels
            ),
            model_parameters.int_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=not model_parameters.bias_free,
        )

        self.down1 = torch.nn.Sequential(
            *[
                ResBlock(
                    model_parameters.int_channels,
                    kernel_size=model_parameters.kernel_size,
                    bias_free=model_parameters.bias_free,
                    act=self._act,
                )
                for _ in range(model_parameters.n_blocks)
            ],
            downsample_strideconv(
                model_parameters.int_channels,
                model_parameters.int_channels * 2,
                bias_free=model_parameters.bias_free,
            ),
        )
        self.down2 = torch.nn.Sequential(
            *[
                ResBlock(
                    model_parameters.int_channels * 2,
                    kernel_size=model_parameters.kernel_size,
                    bias_free=model_parameters.bias_free,
                    act=self._act,
                )
                for _ in range(model_parameters.n_blocks)
            ],
            downsample_strideconv(
                model_parameters.int_channels * 2,
                model_parameters.int_channels * 4,
                bias_free=model_parameters.bias_free,
            ),
        )
        self.down3 = torch.nn.Sequential(
            *[
                ResBlock(
                    model_parameters.int_channels * 4,
                    kernel_size=model_parameters.kernel_size,
                    bias_free=model_parameters.bias_free,
                    act=self._act,
                )
                for _ in range(model_parameters.n_blocks)
            ],
            downsample_strideconv(
                model_parameters.int_channels * 4,
                model_parameters.int_channels * 8,
                bias_free=model_parameters.bias_free,
            ),
        )

        self.bottleneck = torch.nn.Sequential(
            *[
                ResBlock(
                    model_parameters.int_channels * 8,
                    kernel_size=model_parameters.kernel_size,
                    bias_free=model_parameters.bias_free,
                    act=self._act,
                )
                for _ in range(model_parameters.n_blocks)
            ]
        )

        self.up3 = torch.nn.Sequential(
            upsample_convtranspose(
                model_parameters.int_channels * 8,
                model_parameters.int_channels * 4,
                bias_free=model_parameters.bias_free,
            ),
            *[
                ResBlock(
                    model_parameters.int_channels * 4,
                    kernel_size=model_parameters.kernel_size,
                    bias_free=model_parameters.bias_free,
                    act=self._act,
                )
                for _ in range(model_parameters.n_blocks)
            ],
        )
        self.up2 = torch.nn.Sequential(
            upsample_convtranspose(
                model_parameters.int_channels * 4,
                model_parameters.int_channels * 2,
                bias_free=model_parameters.bias_free,
            ),
            *[
                ResBlock(
                    model_parameters.int_channels * 2,
                    kernel_size=model_parameters.kernel_size,
                    bias_free=model_parameters.bias_free,
                    act=self._act,
                )
                for _ in range(model_parameters.n_blocks)
            ],
        )
        self.up1 = torch.nn.Sequential(
            upsample_convtranspose(
                model_parameters.int_channels * 2,
                model_parameters.int_channels,
                bias_free=model_parameters.bias_free,
            ),
            *[
                ResBlock(
                    model_parameters.int_channels,
                    kernel_size=model_parameters.kernel_size,
                    bias_free=model_parameters.bias_free,
                    act=self._act,
                )
                for _ in range(model_parameters.n_blocks)
            ],
        )

        self.project = torch.nn.Conv2d(
            model_parameters.int_channels,
            model_parameters.out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=not model_parameters.bias_free,
        )

    @staticmethod
    def default_parameters():
        params = LIONModelParameter()
        params.model_input_type = ModelInputType.IMAGE
        params.in_channels = 1
        params.out_channels = 1
        params.int_channels = 64
        params.kernel_size = (3, 3)
        params.n_blocks = 4
        params.use_noise_level = False
        params.bias_free = True
        params.act = "leaky_relu"
        params.enforce_positivity = False
        return params

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print(
                "Zhang, Kai, Yawei, Li, Wangmeng, Zuo, Lei, Zhang, Luc, Van Gool, Radu, Timofte."
            )
            print('"Plug-and-Play Image Restoration With Deep Denoiser Prior".')
            print(
                "\x1b[3mIEEE Transactions on Pattern Analysis and Machine Intelligence\x1b[0m"
            )
            print("44. 10(2022): 6360-6376.")
        elif cite_format == "bib":
            print("@article{zhang2022plug,")
            print(
                "author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},"
            )
            print(
                "journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},"
            )
            print("title={Plug-and-Play Image Restoration With Deep Denoiser Prior},")
            print("year={2022},")
            print("volume={44},")
            print("number={10},")
            print("pages={6360-6376},")
            print("doi={10.1109/TPAMI.2021.3088914}")
            print("}")
        else:
            raise ValueError(
                f'`cite_format` "{cite_format}" is not understood, only "MLA" and "bib" are supported'
            )

    def forward(self, x0, noise_level: Optional[float] = None):
        if self.model_parameters.use_noise_level:
            assert (
                isinstance(noise_level, float) and noise_level >= 0.0
            ), "`noise_level` must be a non-negative float, instead got: " + str(
                type(noise_level)
            )
            x0 = torch.cat((x0, noise_level * torch.ones_like(x0)), dim=1)
        x1 = self.lift(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.bottleneck(x4)
        x = self.up3(x + x4)
        x = self.up2(x + x3)
        x = self.up1(x + x2)
        x = self.project(x + x1)

        if self.model_parameters.enforce_positivity:
            return torch.nn.functional.relu(x)
        else:
            return x
