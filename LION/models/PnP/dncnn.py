# This file is part of LION library
# License : GPL-3
#
# Author: Ferdia Sherry
# Modifications: -
# =============================================================================


import torch


class DnCNN(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        int_channels=64,
        kernel_size=(3, 3),
        blocks=20,
        residual=True,
        bias_free=True,
        act=torch.nn.LeakyReLU(),
        enforce_positivity=True,
        batch_normalisation=True,
        device=None,
    ):
        super().__init__()
        self._bias_free = bias_free
        self._residual = residual
        self._batch_normalisation = batch_normalisation
        self._act = act
        self._enforce_positivity = enforce_positivity
        self.lift = torch.nn.Conv2d(
            in_channels,
            int_channels,
            kernel_size,
            padding=tuple(k // 2 for k in kernel_size),
            device=device,
            bias=not bias_free,
        )
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    int_channels,
                    int_channels,
                    kernel_size,
                    padding=tuple(k // 2 for k in kernel_size),
                    device=device,
                    bias=not bias_free,
                )
                for _ in range(blocks - 2)
            ]
        )
        if batch_normalisation:
            self.bns = torch.nn.ModuleList(
                [
                    torch.nn.BatchNorm2d(
                        int_channels, affine=not bias_free, device=device
                    )
                    for _ in range(blocks - 2)
                ]
            )
        self.project = torch.nn.Conv2d(
            int_channels,
            in_channels,
            kernel_size,
            padding=tuple(k // 2 for k in kernel_size),
            device=device,
            bias=not bias_free,
        )

    def _set_weights_zero(self):
        for conv in self.convs:
            conv.weight.data.zero_()
            if not self._bias_free:
                conv.bias.data.zero_()

    def forward(self, x):
        z = self._act(self.lift(x))
        if self._batch_normalisation:
            for conv, bn in zip(self.convs, self.bns):
                z = self._act(bn(conv(z)))
        else:
            for conv in self.convs:
                z = self._act(conv(z))

        if self._residual:
            z = x - self.project(z)
        else:
            z = self.project(z)
        if self._enforce_positivity:
            return torch.nn.functional.relu(z)
        else:
            return z
