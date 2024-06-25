# This file is part of LION library
# License : GPL-3
#
# Author: Ferdia Sherry
# Modifications: -
# =============================================================================


from .dncnn import DnCNN
from .drunet import DRUNet

import torch


class GradientStepModel(torch.nn.Module):
    def __init__(self, backbone_model: torch.nn.Module):
        super().__init__()
        self._backbone_model = backbone_model

    def forward(self, x, training=True):
        _, grad = self.forward_(x, training=training)
        return x - grad

    def forward_(self, x, training=True):
        x.requires_grad_()
        obj = 0.5 * torch.sum((self._backbone_model(x) - x) ** 2)
        grad = torch.autograd.grad(
            obj, x, torch.ones_like(obj), create_graph=training, retain_graph=training
        )[0]
        return obj, grad


class GSDnCNN(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        int_channels=32,
        kernel_size=(3, 3),
        blocks=5,
        bias_free=True,
        act=torch.nn.ELU(),
        device=None,
    ):
        super().__init__()
        self._backbone_model = DnCNN(
            in_channels=in_channels,
            int_channels=int_channels,
            kernel_size=kernel_size,
            blocks=blocks,
            bias_free=bias_free,
            act=act,
            enforce_positivity=False,
            batch_normalisation=True,
            device=device,
        )
        self._gsm = GradientStepModel(self._backbone_model)

    def forward(self, x, training=True):
        return self._gsm(x, training=training)


class GSDRUNet(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        int_channels=32,
        kernel_size=(3, 3),
        n_blocks=2,
        bias_free=True,
        act=torch.nn.ELU(),
        device=None,
    ):
        super().__init__()
        self._backbone_model = DRUNet(
            in_channels=in_channels,
            out_channels=in_channels,
            int_channels=int_channels,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            bias_free=bias_free,
            act=act,
            enforce_positivity=False,
            device=device,
        )
        self._gsm = GradientStepModel(self._backbone_model)

    def forward(self, x, training=True):
        return self._gsm(x, training=training)
