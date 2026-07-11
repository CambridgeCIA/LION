from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType

from LION.utils.math import power_method
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct
import LION.CTtools.ct_utils as ct_utils
import LION.utils.utils as ai_utils

# new


import torch
from torch.nn import (
    Module,
    Sequential,
    Conv1d,
    Conv2d,
    BatchNorm1d,
    PReLU,
    ReLU,
    Parameter,
    Hardtanh,
)
from torch.nn.functional import pad
import matplotlib.pyplot as plt
import tomosipo as ts
from tomosipo.torch_support import to_autograd
import json

# from model import ModelBase


class LearnableFilter(Module):
    """
    Learnable frequency-domain filter module for CT sinograms.
    This module replaces traditional filters like Ram-Lak with trainable parameters
    in the frequency domain. It supports shared or per-angle filters.
    Args:
        init_filter (torch.Tensor):
            Initial 1D filter in the frequency domain.
        per_angle (bool):
            If True, uses one filter per angle.
        num_angles (int, optional):
            Required if per_angle=True.
    """

    def __init__(self, init_filter, per_angle=False, num_angles=None):
        super().__init__()
        self.per_angle = per_angle

        if per_angle:
            assert (
                num_angles is not None
            ), "num_angles must be provided when per_angle=True"
            filters = torch.stack(
                [init_filter.clone().detach() for _ in range(num_angles)]
            )
            self.register_parameter("weights", Parameter(filters))
        else:
            filters = torch.stack([init_filter.clone().detach()])  # shape: (1, D)
            self.register_parameter("weights", Parameter(filters))

    def forward(self, x):
        """
        Applies the learnable frequency filter to each projection.
        Args:
            x (Tensor):
                Input sinogram of shape [B, A, D], where B = batch size, A = number of angles, D = number of detectors.
        Returns:
            Tensor: Filtered sinogram of the same shape.
        """

        ftt1d = torch.fft.fft(x, dim=-1)
        if self.per_angle:
            filtered = ftt1d * self.weights[None, :, :]
        else:
            filter_shared = self.weights.expand(ftt1d.shape[1], -1)
            filtered = ftt1d * filter_shared[None, :, :]
        return torch.fft.ifft(filtered, dim=-1).real


class IntermediateResidualBlock(Module):
    """
    Depthwise 1D residual block used for angular interpolation.
    Each channel is processed independently (depthwise convolution),
    allowing flexible angle-wise feature transformation.
    Args:
        channels (int):
            Number of input/output channels (must match).
    """

    def __init__(self, channels):
        super().__init__()
        self.block = Sequential(
            Conv1d(
                channels, channels, kernel_size=3, padding=1, groups=channels, bias=True
            ),
            BatchNorm1d(channels),
            PReLU(),
        )

    def forward(self, x):
        return x + self.block(x)


class DenoisingResidualBlock(Module):
    """
    2D residual block for image denoising.
    """

    def __init__(self, channels):
        super().__init__()
        self.block = Sequential(
            Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            ReLU(inplace=True),
            Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        return x + self.block(x)


class DeepFBPNetwork(LIONmodel):
    """
    Deep Filtered Backprojection Network for CT reconstruction.
    Pipeline:
        - Learnable frequency filter (Ram-Lak based init)
        - Angular interpolation via depthwise convolutions
        - Differentiable backprojection (Tomosipo)
        - Residual CNN-based denoiser
    Args:
        num_detectors (int):
            Number of detector bins.
        num_angles (int):
            Number of projection angles.
        A (ts.Operator):
            Tomosipo projection operator.
        filter_type (str):
            Either 'Filter I' (shared) or per-angle.
        device (torch.device):
            Computation device.
    """

    def __init__(self, geometry: ct.Geometry, model_parameters: LIONParameter = None):
        super().__init__(model_parameters, geometry)
        self._make_operator()

        self.num_detectors = geometry.detector_shape[1]
        self.num_angles_ = len(geometry.angles)

        # Padding to reach power-of-two size for FFT
        self.projection_size_padded = self.compute_projection_size_padded()
        self.padding = self.projection_size_padded - self.num_detectors

        # Initialize Ram-Lak filter in frequency domain
        ram_lak = self.ram_lak_filter(self.projection_size_padded)
        if self.model_parameters.filter_type == "Filter I":
            self.learnable_filter = LearnableFilter(ram_lak, per_angle=False)
        else:
            self.learnable_filter = LearnableFilter(
                ram_lak, per_angle=True, num_angles=self.num_angles_
            )

        # Interpolation blocks
        self.interpolator_1 = IntermediateResidualBlock(1)
        self.interpolator_2 = IntermediateResidualBlock(1)
        self.interpolator_3 = IntermediateResidualBlock(1)
        self.interpolator_conv = Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        # Tomosipo normalization map (1s projection)
        sinogram_ones = torch.ones(
            (1, 1, self.num_angles_, self.num_detectors),
            device=torch.cuda.current_device(),
        )
        self.tomosipo_normalizer = (
            self.AT(sinogram_ones) + 1e-6
        )  # Avoid division by zero in normalization

        # Denoising blocks
        self.denoising_conv_1 = Conv2d(1, 64, kernel_size=1)
        self.denoising_conv_2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoising_res_1 = DenoisingResidualBlock(64)
        self.denoising_res_2 = DenoisingResidualBlock(64)
        self.denoising_res_3 = DenoisingResidualBlock(64)
        self.denoising_output = Sequential(
            Conv2d(64, 1, kernel_size=3, padding=1), Hardtanh(0, 1)
        )

    def compute_projection_size_padded(self):
        """
        Computes the next power-of-two padding size to avoid aliasing.
        """

        return 2 ** int(
            torch.ceil(
                torch.log2(torch.tensor(self.num_detectors, dtype=torch.float32))
            ).item()
        )

    def ram_lak_filter(self, size):
        """
        Generates Ram-Lak filter directly in frequency domain.
        """

        steps = int(size / 2 + 1)
        ramp = torch.linspace(0, 1, steps, dtype=torch.float32)
        down = torch.linspace(1, 0, steps, dtype=torch.float32)
        f = torch.cat([ramp, down[:-2]])
        return f

    # new
    @staticmethod
    def default_parameters():
        DFBP_params = LIONModelParameter()
        DFBP_params.filter_type = "Filter I"
        DFBP_params.mode = "ct"
        DFBP_params.model_input_type = ModelInputType.SINOGRAM
        DFBP_params.normalizer = True
        return DFBP_params

    # new

    def forward(self, x):
        """
        Runs a forward pass through the DeepFBP pipeline.
        Steps:
            1. Apply learnable frequency-domain filter.
            2. Perform angular interpolation via 1D convolutions.
            3. Apply differentiable backprojection using Tomosipo.
            4. Normalize projection output.
            5. Apply residual 2D CNN denoising network.
        Args:
            x (Tensor):
                Input sinogram of shape [B, 1, A, D].
        Returns:
            Tensor:
                Reconstructed image tensor of shape [B, 1, H, W], with pixel values in [0, 1].
        """

        # Initial shape: [B, 1, A, D]
        x = x.squeeze(1)  # [B, A, D]
        x = pad(x, (0, self.padding), mode="constant", value=0)
        x = self.learnable_filter(x)
        x = x[..., : self.num_detectors]  # Remove padding

        # Interpolation network
        x = x.reshape(-1, 1, self.num_detectors)
        x = self.interpolator_1(x)
        x = self.interpolator_2(x)
        x = self.interpolator_3(x)
        x = self.interpolator_conv(x)

        # Reshape back to sinogram format for backprojection
        x = x.view(-1, self.num_angles_, self.num_detectors).unsqueeze(1)

        # Differentiable backprojection
        img = self.AT(x)

        # normlise images
        if self.model_parameters.normalizer == True:
            img = img / self.tomosipo_normalizer

        # Denoising network
        x = self.denoising_conv_1(img)
        x = self.denoising_conv_2(x)
        x = self.denoising_res_1(x)
        x = self.denoising_res_2(x)
        x = self.denoising_res_3(x)
        x = self.denoising_output(x)

        return x
