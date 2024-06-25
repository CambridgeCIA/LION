# This file is part of LION library
# License : GPL-3
#
# Author: Ferdia Sherry
# Modifications: -
# =============================================================================


import torch


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        n_channels=64,
        kernel_size=(3, 3),
        bias_free=True,
        act=torch.nn.LeakyReLU(),
        device=None,
    ):
        super().__init__()
        self.res = torch.nn.Sequential(
            torch.nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size,
                padding=tuple(k // 2 for k in kernel_size),
                device=device,
                bias=not bias_free,
            ),
            act,
            torch.nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size,
                padding=tuple(k // 2 for k in kernel_size),
                device=device,
                bias=not bias_free,
            ),
        )

    def forward(self, x):
        res = self.res(x)
        return x + res


def downsample_strideconv(
    in_channels=1,
    out_channels=64,
    kernel_size=2,
    stride=2,
    bias_free=True,
    device=None,
):
    return torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        device=device,
        bias=not bias_free,
    )


def upsample_convtranspose(
    in_channels=64,
    out_channels=1,
    kernel_size=2,
    stride=2,
    bias_free=True,
    device=None,
):
    return torch.nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        device=device,
        bias=not bias_free,
    )


class DRUNet(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        int_channels=64,
        kernel_size=(3, 3),
        n_blocks=4,
        noise_level=0.05,
        bias_free=True,
        act=torch.nn.LeakyReLU(),
        enforce_positivity=True,
        device=None,
    ):
        super().__init__()
        self._enforce_positivity = enforce_positivity
        self.noise_level = noise_level
        self.lift = torch.nn.Conv2d(
            in_channels,
            int_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=not bias_free,
            device=device,
        )

        self.down1 = torch.nn.Sequential(
            *[
                ResBlock(
                    int_channels,
                    kernel_size=kernel_size,
                    bias_free=bias_free,
                    act=act,
                    device=device,
                )
                for _ in range(n_blocks)
            ],
            downsample_strideconv(
                int_channels, int_channels * 2, bias_free=bias_free, device=device
            ),
        )
        self.down2 = torch.nn.Sequential(
            *[
                ResBlock(
                    int_channels * 2,
                    kernel_size=kernel_size,
                    bias_free=bias_free,
                    act=act,
                    device=device,
                )
                for _ in range(n_blocks)
            ],
            downsample_strideconv(
                int_channels * 2, int_channels * 4, bias_free=bias_free, device=device
            ),
        )
        self.down3 = torch.nn.Sequential(
            *[
                ResBlock(
                    int_channels * 4,
                    kernel_size=kernel_size,
                    bias_free=bias_free,
                    act=act,
                    device=device,
                )
                for _ in range(n_blocks)
            ],
            downsample_strideconv(
                int_channels * 4, int_channels * 8, bias_free=bias_free, device=device
            ),
        )

        self.bottleneck = torch.nn.Sequential(
            *[
                ResBlock(
                    int_channels * 8,
                    kernel_size=kernel_size,
                    bias_free=bias_free,
                    act=act,
                    device=device,
                )
                for _ in range(n_blocks)
            ]
        )

        self.up3 = torch.nn.Sequential(
            upsample_convtranspose(
                int_channels * 8, int_channels * 4, bias_free=bias_free, device=device
            ),
            *[
                ResBlock(
                    int_channels * 4,
                    kernel_size=kernel_size,
                    bias_free=bias_free,
                    act=act,
                    device=device,
                )
                for _ in range(n_blocks)
            ],
        )
        self.up2 = torch.nn.Sequential(
            upsample_convtranspose(
                int_channels * 4, int_channels * 2, bias_free=bias_free, device=device
            ),
            *[
                ResBlock(
                    int_channels * 2,
                    kernel_size=kernel_size,
                    bias_free=bias_free,
                    act=act,
                    device=device,
                )
                for _ in range(n_blocks)
            ],
        )
        self.up1 = torch.nn.Sequential(
            upsample_convtranspose(
                int_channels * 2, int_channels, bias_free=bias_free, device=device
            ),
            *[
                ResBlock(
                    int_channels,
                    kernel_size=kernel_size,
                    bias_free=bias_free,
                    act=act,
                    device=device,
                )
                for _ in range(n_blocks)
            ],
        )

        self.project = torch.nn.Conv2d(
            int_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=not bias_free,
            device=device,
        )

    def forward(self, x0):
        x0 = x0
        x1 = self.lift(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.bottleneck(x4)
        x = self.up3(x + x4)
        x = self.up2(x + x3)
        x = self.up1(x + x2)
        x = self.project(x + x1)

        if self._enforce_positivity:
            return torch.nn.functional.relu(x)
        else:
            return x
