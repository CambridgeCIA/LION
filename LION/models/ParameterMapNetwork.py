from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ExplicitMap(torch.nn.Module):
    """A directly trainable raw parameter map."""

    def __init__(self, lambda_map_raw_init: torch.Tensor) -> None:
        super().__init__()
        self.lambda_map_raw = nn.Parameter(lambda_map_raw_init, requires_grad=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.lambda_map_raw.expand(
            image.shape[0], *self.lambda_map_raw.shape[1:]
        )


class ParameterMapNetwork(nn.Module):
    r"""A wrapper network for estimating regularization parameter maps."""

    def __init__(
        self,
        map_maker: ExplicitMap | nn.Module,
        keys: Sequence[int],
        *,
        initial_scale_factor_raw: float,
        scale_factor_softplus_beta: float,
        lambda_softplus_beta: float,
    ) -> None:
        r"""Initialize Parameter Map Network.

        Parameters
        ----------
        map_maker
            A neural network block for estimating the regularization parameter maps.
        """
        super().__init__()
        self.map_maker = map_maker
        self.scale_factors_raw = nn.ParameterDict(
            {
                str(key): nn.Parameter(torch.tensor(initial_scale_factor_raw))
                for key in keys
            }
        )
        self.scale_factor_softplus_beta = scale_factor_softplus_beta
        self.lambda_softplus_beta = lambda_softplus_beta

    def forward(self, image: torch.Tensor, key: int) -> torch.Tensor:
        r"""Apply the network to estimate regularization parameter maps.

        Parameters
        ----------
        image
            the image from which the regularization parameter maps should be estimated.
        key
            the key for the regularization parameter.
        """
        return nn.functional.softplus(
            self.scale_factors_raw[str(key)], beta=self.scale_factor_softplus_beta
        ) * nn.functional.softplus(
            self.map_maker(image), beta=self.lambda_softplus_beta
        )
