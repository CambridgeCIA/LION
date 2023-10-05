# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Subhadip Mukherjee
# Modifications: Ander Biguri
# =============================================================================


import torch
import torch.nn as nn
from AItomotools.models import AItomomodel
from AItomotools.utils.parameter import Parameter
import torch.nn.utils.parametrize as P


class Positive(nn.Module):
    def forward(self, X):
        return torch.clip(X, min=0.0)


class ICNN_layer(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, relu_type="LeakyReLU"):
        super().__init__()

        # The paper diagram is in color, channels are described by "blue" and "orange"
        self.blue = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            bias=False,
        )
        P.register_parametrization(self.blue, "weight", Positive())

        self.orange = nn.Conv2d(
            in_channels=1,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            bias=True,
        )
        if relu_type == "LeakyReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(
                "Only Leaky ReLU supported (needs to be a convex and monotonically nondecreasin fun)"
            )

    def forward(self, z, x0):

        res = self.blue(z) + self.orange(x0)
        res = self.activation(res)
        return res


###An L2 tern with learnable weight
# define a network for training the l2 term
class L2net(nn.Module):
    def __init__(self):
        super(L2net, self).__init__()

        self.l2_penalty = nn.Parameter((-9.0) * torch.ones(1))

    def forward(self, x):
        l2_term = torch.sum(x.view(x.size(0), -1) ** 2, dim=1)
        out = ((torch.nn.functional.softplus(self.l2_penalty)) * l2_term).view(
            x.size(0), -1
        )
        return out


# sparsifying filter-bank (SFB) module
class SFB(AItomomodel.AItomoModel):
    def __init__(self, model_parameters):
        if model_parameters is None:
            model_parameters = ACR.default_parameters()
        super().__init__(model_parameters)
        # FoE kernels
        self.penalty = nn.Parameter((-12.0) * torch.ones(1))
        self.n_kernels = model_parameters.n_kernels
        self.conv = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    model_parameters.n_filters,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    bias=False,
                )
                for i in range(self.n_kernels)
            ]
        )
        if model_parameters.L2net:
            self.L2net = L2net()

    @staticmethod
    def default_parameters():
        param = Parameter()
        param.n_kernels = 10
        param.n_filters = 32
        paran.L2net = True
        return param

    def forward(self, x):
        # compute the output of the FoE part
        total_out = 0.0
        for kernel_idx in range(self.n_kernels):
            x_out = torch.abs(self.conv[kernel_idx](x))
            x_out_flat = x_out.view(x.size(0), -1)
            total_out += torch.sum(x_out_flat, dim=1)

        total_out = total_out.view(x.size(0), -1)
        out = (torch.nn.functional.softplus(self.penalty)) * total_out
        if self.model_parameters.L2net:
            out = out + self.L2net(x)
        return out


class ACR(AItomomodel.AItomoModel):
    def __init__(self, model_parameters: Parameter = None):
        if model_parameters is None:
            model_parameters = ACR.default_parameters()
        super().__init__(model_parameters)

        # First Conv
        self.first_layer = nn.Conv2d(
            in_channels=1,
            out_channels=model_parameters.channels,
            kernel_size=model_parameters.kernel_size,
            stride=model_parameters.stride,
            padding="same",
            bias=True,
        )

        if model_parameters.relu_type == "LeakyReLU":
            self.first_activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(
                "Only Leaky ReLU supported (needs to be a convex and monotonically nondecreasin fun)"
            )

        for i in range(model_parameters.layers):
            self.add_module(
                f"ICNN_layer_{i}",
                ICNN_layer(
                    channels=model_parameters.channels,
                    kernel_size=model_parameters.kernel_size,
                    stride=model_parameters.stride,
                    relu_type=model_parameters.relu_type,
                ),
            )

        self.last_layer = nn.Conv2d(
            in_channels=model_parameters.channels,
            out_channels=1,
            kernel_size=model_parameters.kernel_size,
            stride=model_parameters.stride,
            padding="same",
            bias=False,
        )
        P.register_parametrization(self.last_layer, "weight", Positive())

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.initialize_weights()

    # a weight initialization routine for the ICNN
    def initialize_weights(self, min_val=0.0, max_val=0.001):
        device = torch.cuda.current_device()
        for i in range(self.model_parameters.layers):
            block = getattr(self, f"ICNN_layer_{i}")
            block.blue.weight.data = min_val + (max_val - min_val) * torch.rand(
                self.model_parameters.channels,
                self.model_parameters.channels,
                self.model_parameters.kernel_size,
                self.model_parameters.kernel_size,
            ).to(device)

        return self

    def forward(self, x):

        z = self.first_layer(x)
        z = self.first_activation(z)
        for i in range(self.model_parameters.layers):
            layer = primal_module = getattr(self, f"ICNN_layer_{i}")
            z = layer(z, x)
        z = self.last_layer(z)
        return self.pool(z)

    @staticmethod
    def default_parameters():
        param = Parameter()
        param.channels = 48
        param.kernel_size = 5
        param.stride = 1
        param.relu_type = "LeakyReLU"
        param.layers = 10
        return param

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Mukherjee, Subhadip, et al.")
            print('"Learned convex regularizers for inverse problems."')
            print("\x1B[3marXiv preprint \x1B[0m")
            print("arXiv:2008.02839 (2020).")
        elif cite_format == "bib":
            string = """
            @article{mukherjee2020learned,
            title={Learned convex regularizers for inverse problems},
            author={Mukherjee, Subhadip and Dittmer, S{\"o}ren and Shumaylov, Zakhar and Lunz, Sebastian and {\"O}ktem, Ozan and Sch{\"o}nlieb, Carola-Bibiane},
            journal={arXiv preprint arXiv:2008.02839},
            year={2020}
            }"""
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )
