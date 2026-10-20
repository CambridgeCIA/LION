# This file is part of LION library
# License : GPL-3
#
# Author  : Zakhar Shumaylov, Subhadip Mukherjee
# Modifications: Ander Biguri, Zakhar Shumaylov
# =============================================================================


from typing import Optional
import torch
import torch.nn as nn
from LION.models.LIONmodel import LIONmodel, ModelInputType, LIONModelParameter
import LION.CTtools.ct_geometry as ct
import torch.nn.utils.parametrize as P
from LION.utils.math import power_method


class Positive(nn.Module):
    def forward(self, X):
        return torch.clip(X, min=0.0)


class ICNN_layer(nn.Module):
    def __init__(self, channels, kernel_size=5, stride=1, relu_type="LeakyReLU"):
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
    """``softplus(w) * ||x||^2`` per sample, with a learnable weight ``w``."""

    def __init__(self, l2_penalty: float = -9.0):
        super(L2net, self).__init__()

        self.l2_penalty = nn.Parameter((l2_penalty) * torch.ones(1))

    def forward(self, x):
        l2_term = torch.sum(x.view(x.size(0), -1) ** 2, dim=1)
        out = ((torch.nn.functional.softplus(self.l2_penalty)) * l2_term).view(
            x.size(0), -1
        )
        return out


class ACR(LIONmodel):
    def __init__(
        self,
        geometry_parameters: Optional[ct.Geometry] = None,
        model_parameters: Optional[LIONModelParameter] = None,
    ):

        super().__init__(model_parameters, geometry_parameters)
        params = self.model_parameters  # defaults filled in by LIONmodel
        # First Conv
        self.first_layer = nn.Conv2d(
            in_channels=1,
            out_channels=params.channels,
            kernel_size=params.kernel_size,
            stride=params.stride,
            padding="same",
            bias=True,
        )

        if params.relu_type == "LeakyReLU":
            self.first_activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(
                "Only Leaky ReLU supported (needs to be a convex and monotonically nondecreasin fun)"
            )

        for i in range(params.layers):
            self.add_module(
                f"ICNN_layer_{i}",
                ICNN_layer(
                    channels=params.channels,
                    kernel_size=params.kernel_size,
                    stride=params.stride,
                    relu_type=params.relu_type,
                ),
            )

        self.last_layer = nn.Conv2d(
            in_channels=params.channels,
            out_channels=1,
            kernel_size=params.kernel_size,
            stride=params.stride,
            padding="same",
            bias=False,
        )
        P.register_parametrization(self.last_layer, "weight", Positive())

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.L2 = L2net(params.l2_penalty)
        self.initialize_weights()
        if self.geometry is not None:
            # Step size for gradient descent on the variational problem
            self._make_operator()
            self.op_norm = power_method(self.op)
            self.model_parameters.step_size = 1 / (self.op_norm) ** 2

    @staticmethod
    def _set_positive_weight(conv: nn.Conv2d, value: torch.Tensor):
        # The weight is parametrized (clipped to be positive): write the
        # underlying tensor, assigning to conv.weight.data has no effect.
        conv.parametrizations.weight.original.data = value

    # a weight initialization routine for the ICNN
    def initialize_weights(self, min_val=0.0, max_val=1e-3):
        for i in range(self.model_parameters.layers):
            block = getattr(self, f"ICNN_layer_{i}")
            weight = block.blue.parametrizations.weight.original
            self._set_positive_weight(
                block.blue, min_val + (max_val - min_val) * torch.rand_like(weight)
            )
        weight = self.last_layer.parametrizations.weight.original
        self._set_positive_weight(
            self.last_layer, min_val + (max_val - min_val) * torch.rand_like(weight)
        )
        return self

    def improved_initialize_weights(self, min_val=0.0, max_val=0.001):
        ###
        ### This is based on a recent paper https://openreview.net/pdf?id=pWZ97hUQtQ
        ###
        # convex_init = ConvexInitialiser()
        # w1, b1 = icnn[1].parameters()
        # convex_init(w1, b1)
        # assert torch.all(w1 >= 0) and b1.var() > 0
        return self.initialize_weights(min_val, max_val)

    def forward(self, x):
        # x = fdk(self.op, x)
        x = self.normalise(x)
        z = self.first_layer(x)
        z = self.first_activation(z)
        for i in range(self.model_parameters.layers):
            layer = getattr(self, f"ICNN_layer_{i}")
            z = layer(z, x)

        z = self.last_layer(z)
        # convex network output + learnable L2 term on the input image
        return self.pool(z).reshape(-1, 1) + self.L2(x)

    @staticmethod
    def default_parameters():
        params = LIONModelParameter()
        params.channels = 16
        params.kernel_size = 5
        params.stride = 1
        params.relu_type = "LeakyReLU"
        params.layers = 5
        # initial value of the L2 weight before softplus (softplus(-9) ~ 1e-4)
        params.l2_penalty = -9.0
        params.model_input_type = ModelInputType.IMAGE
        return params

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print(
                'MMukherjee, Subhadip and Dittmer, S{"o}ren and Shumaylov, Zakhar and Lunz, Sebastian and {"O}ktem, Ozan and Sch{"o}nlieb, Carola-Bibiane'
            )
            print('"Learned convex regularizers for inverse problems"')
            print("arXiv preprint arXiv:2008.02839 (2020)")
            print("arXiv:2008.02839 (2020).")
        elif cite_format == "bib":
            string = """
            @article{mukherjee2020learned,
            title={Learned convex regularizers for inverse problems},
            author={Mukherjee, Subhadip and Dittmer, S{\"o}ren and Shumaylov, Zakhar and Lunz, Sebastian and {\"O}ktem, Ozan and Sch{\"o}nlieb, Carola-Bibiane},
            journal={arXiv preprint arXiv:2008.02839},
            year={2020}}
            """
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )
