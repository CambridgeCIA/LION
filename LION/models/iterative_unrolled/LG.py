# This file is part of LION library
# License : GPL-3
#
# Author  : Zakhar Shumaylov, Max Kiss, Ander Biguri
# Based on: https://arxiv.org/pdf/1704.04058.pdf
# =============================================================================

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType

from LION.utils.math import power_method
from LION.utils.parameter import LIONParameter, OperatorParameters


import torch
import torch.nn as nn
from LION.utils.citation import citation


class LGblock(nn.Module):
    """A learned-gradient block: 3x3 convolutions through the ``channels``
    listed, with ReLUs between them."""

    def __init__(self, channels):
        super(LGblock, self).__init__()

        layers = len(channels) - 1

        layer_list = []
        for ii in range(layers):
            layer_list.append(nn.Conv2d(channels[ii], channels[ii + 1], 3, padding=1))
            # Have PReLUs all the way except the last
            if ii < layers - 1:
                layer_list.append(torch.nn.ReLU())
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        """The block's output for ``x``."""
        return self.block(x)


class LG(LIONmodel):
    """Learned Gradient (Adler and Öktem, 2017): ``n_iters`` updates of the
    image, each a CNN of the image, a memory and the data-fit gradient ``Aᵀ(y
    - A x)``, with learned step sizes; with ``tv``, also of a TV gradient
    (LGTV). Starts from the initial reconstruction. Takes the measurements.

    ``model_parameters`` (``default_parameters()``):

    - ``channels``: each block's channels, a list: its input (the image, the
      memory, the gradient, and the TV gradient with ``tv``) to its output
      (the memory, then the update); [7, 32, 32, 6];
    - ``n_iters``: the iterations (5);
    - ``tv``: also give each block a TV gradient, with a learned step size
      (LGTV; False, as the paper). ``LG.lgtv_parameters()`` are LGTV's."""

    def __init__(
        self,
        operator_params: OperatorParameters,
        model_parameters: LIONParameter = None,
    ):
        super().__init__(model_parameters, operator_params)
        self.operator_params = operator_params
        self.channels = self.model_parameters.channels
        self.M = self.channels[-1] - 1
        self.n_iters = self.model_parameters.n_iters
        self.tv = self.model_parameters.tv
        # the block's input: the image, the memory, the gradient(s)
        inputs = 1 + self.M + 1 + int(self.tv)
        if self.channels[0] != inputs:
            raise ValueError(
                f"channels must start with {inputs} channels (the image, a memory "
                f"of {self.M}, the gradient{' and the TV gradient' if self.tv else ''}), "
                f"got channels={self.channels}"
            )

        self._make_operator(autograd=True)
        gd_step = 1 / (power_method(self.op)) ** 2

        for i in range(self.n_iters):
            self.add_module(f"{i}_conv", LGblock(self.model_parameters.channels))

        self.step_size = nn.ParameterList(
            [nn.Parameter(torch.ones(1) * gd_step) for i in range(self.n_iters)]
        )
        if self.tv:
            self.tv_step_size = nn.ParameterList(
                [nn.Parameter(torch.ones(1) * 1e-4) for i in range(self.n_iters)]
            )

    @staticmethod
    def default_parameters():
        """5 iterations of blocks of 32 channels, a memory of 5 channels, no
        TV gradient (see the class)."""
        LG_params = LIONModelParameter()
        LG_params.channels = [7, 32, 32, 6]
        LG_params.n_iters = 5
        LG_params.tv = False
        LG_params.model_input_type = ModelInputType.SINOGRAM
        return LG_params

    @staticmethod
    def lgtv_parameters():
        """LGTV: LG whose blocks also get a TV gradient (``tv``), as in Kiss et
        al., 2025 (see the class)."""
        LG_params = LG.default_parameters()
        LG_params.channels = [8, 32, 32, 6]
        LG_params.tv = True
        return LG_params

    @staticmethod
    def cite(cite_format="bib"):
        """Where LG comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{adler2017solving,
            title={Solving ill-posed inverse problems using iterative deep neural networks},
            author={Adler, Jonas and {\\"O}ktem, Ozan},
            journal={Inverse Problems},
            volume={33},
            number={12},
            pages={124007},
            year={2017},
            publisher={IOP Publishing}
            }
            """,
            mla="""
            Adler, Jonas, and Öktem, Ozan.
            "Solving ill-posed inverse problems using iterative deep neural networks."
            Inverse Problems
            33.12 (2017): 124007.
            """,
        )

    def gradientTVnorm2D(self, f):
        """The gradient of the (smoothed) isotropic total variation of the
        images ``f`` (B, C, H, W)."""
        Gx = torch.diff(f, dim=2)
        Gy = torch.diff(f, dim=3)
        tvg = torch.zeros_like(f)

        Gx = torch.cat(
            (torch.zeros_like(Gx[:, :, :1, :]), Gx), dim=2
        )  # Pad Gx with zeros
        Gy = torch.cat(
            (torch.zeros_like(Gy[:, :, :, :1]), Gy), dim=3
        )  # Pad Gy with zeros

        nrm = torch.sqrt(Gx**2 + Gy**2 + 1e-6)

        tvg[:, :, :, :] = (
            tvg[:, :, :, :] + (Gx[:, :, :, :] + Gy[:, :, :, :]) / nrm[:, :, :, :]
        )
        tvg[:, :, :-1, :] = tvg[:, :, :-1, :] - Gx[:, :, 1:, :] / nrm[:, :, 1:, :]
        tvg[:, :, :, :-1] = tvg[:, :, :, :-1] - Gy[:, :, :, 1:] / nrm[:, :, :, 1:]

        return tvg

    def forward(self, x):

        """The reconstructions of the measurements ``x`` (B, 1, ...)."""
        B, C, W, H = x.shape

        f = x.new_zeros(B, 1, *self.op.domain_shape[1:])
        f[:] = self.initial_reconstruction(
            x[:, :1], self.op
        )  # the operator's default (FDK for CT) unless set

        tmp = torch.zeros(f.shape).type_as(f)
        s = tmp.clone()
        for i in range(self.M - 1):
            s = torch.cat((s, tmp), dim=1)

        for i in range(self.n_iters):
            conv = getattr(self, f"{i}_conv")
            gradients = [self.step_size[i] * self.AT(x - self.A(f))]
            if self.tv:
                gradients.append(self.tv_step_size[i] * self.gradientTVnorm2D(f))
            output = conv(torch.cat([f, s, *gradients], dim=1))
            # output last channel
            f = f + output[:, self.M : self.M + 1]
            s = nn.ReLU()(output[:, 0 : self.M])
        return f
