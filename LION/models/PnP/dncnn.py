# This file is part of LION library
# License : GPL-3
#
# Author: Ferdia Sherry
# Modifications: -
# =============================================================================

from inspect import getmembers, isfunction
import torch
from LION.models.LIONmodel import LIONmodel, LIONParameter


class DnCNN(LIONmodel):
    @staticmethod
    def default_parameters():
        return LIONParameter(
            in_channels=1,
            int_channels=64,
            kernel_size=(3, 3),
            blocks=20,
            residual=True,
            bias_free=True,
            act="leaky_relu",
            enforce_positivity=True,
            batch_normalisation=True,
        )

    def __init__(self, model_parameters: LIONParameter):
        super().__init__(model_parameters)
        self._bias_free = model_parameters.bias_free
        self._residual = model_parameters.residual
        self._batch_normalisation = model_parameters.batch_normalisation
        if model_parameters.act.lower() in getmembers(torch.nn.functional, isfunction):
            self._act = torch.nn.functional.__dict__[model_parameters.act]
        else:
            raise ValueError(
                f"`torch.nn.functional` does not export a function '{model_parameters.act}'."
            )
        self._enforce_positivity = self.model_parameters.enforce_positivity
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
