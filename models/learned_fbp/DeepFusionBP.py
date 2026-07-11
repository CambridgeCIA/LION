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
    ModuleList,
    Sequential,
    Conv1d,
    Conv2d,
    BatchNorm1d,
    PReLU,
    ReLU,
    Parameter,
    BatchNorm2d,
)
from torch.nn.functional import pad
import matplotlib.pyplot as plt
import tomosipo as ts
from tomosipo.torch_support import to_autograd
import json

# from .model import ModelBase


class LearnableFilter(Module):
    """
    Implements a learnable frequency-domain filter for sinograms in CT reconstruction.

    This module replaces traditional analytic filters (e.g., Ram-Lak) with learnable weights
    in the frequency domain, optionally per projection angle.

    Args:
        init_filter (torch.Tensor):
            Initial filter in frequency domain.
        per_angle (bool):
            If True, uses one filter per angle.
        num_angles (int, optional):
            Required when per_angle=True.
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
        Applies learnable frequency-domain filtering to each projection.

        Args:
            x (Tensor):
                Input sinogram of shape [B, A, D].

        Returns:
            Tensor:
                Filtered sinogram of shape [B, A, D].
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
    Depthwise 1D convolutional residual block.

    Used in the angular interpolation subnetwork. Applies a depthwise Conv1d,
    followed by BatchNorm1d and PReLU, with a residual connection.

    Args:
        channels (int):
            Number of input/output channels.
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


class single_back_projections(Module):
    """
    Applies single-angle differentiable backprojections using Tomosipo.

    Each angle in the sparse set is used to compute a separate backprojection,
    forming a stack of angle-specific reconstructions.

    Args:
        angles_sparse (torch.Tensor):
            Sparse angles to backproject.
        src_orig_dist (float):
            Source-to-origin distance.
        num_detectors (int):
            Number of detector bins.
        vg (ts.VolumeGeometry):
            Volume geometry for tomosipo operator.
    """

    def __init__(self, angles_sparse, src_orig_dist, num_detectors, vg):

        super().__init__()
        self.angles_sparse = angles_sparse
        self.src_orig_dist = src_orig_dist
        self.num_detectors = num_detectors
        self.vg = vg

        self.tomosipo_geometries = []
        for angle in self.angles_sparse:
            # Define Fan Beam Geometry for each angle
            proj_geom_single = ts.cone(
                angles=angle,
                src_orig_dist=self.src_orig_dist,
                shape=(1, self.num_detectors),
            )

            # Compute Back Projection
            A_single = ts.operator(self.vg, proj_geom_single)

            # make operator diferenciable
            self.AT = to_autograd(A_single.T, is_2d=True, num_extra_dims=2)

            self.tomosipo_geometries.append(self.AT)

    def forward(self, sinogram):
        """
        Generates a set of backprojections from  sparse-view sinogram.

        Each angle is used to generate a single-angle backprojection using tomosipo.

        Args:
            sinogram (torch.Tensor):
                Noisy sinogram of shape (1, num_angles, num_detectors).

        Returns:
            torch.Tensor:
                Stack of backprojections of shape (n_single_BP, H, W).
        """

        projections = []

        for i, operator in enumerate(self.tomosipo_geometries):

            # Extract only the sinogram at this specific angle
            sinogram_angle = sinogram[:, :, i : i + 1, :]

            # Back projection at single angle
            projection = operator(sinogram_angle)

            projections.append(projection)

        # Stack all projections into a single tensor of shape [view_angles, 362, 362]
        single_back_projection = torch.stack(projections, dim=1).squeeze(2)

        return single_back_projection


class DBP_block(Module):
    """
    Deep CNN for denoising stacked single-angle backprojections.

    Composed of an initial Conv2d+ReLU, 15 intermediate Conv2d+BN+ReLU layers,
    and a final Conv2d output layer. Adapted from the DBP architecture.

    Args:
        channels (int):
            Number of input channels (i.e., number of angles).
    """

    def __init__(self, channels):

        # Initialize the base training infrastructure
        super().__init__()

        # initial layer
        self.conv1 = self.initial_layer(
            in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        # middel layer (15 equal layers)
        self.middle_blocks = ModuleList(
            [
                self.conv_block(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                )
                for _ in range(15)
            ]
        )

        # last layer
        self.final = self.final_layer(
            in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1
        )

        # change parameters
        self.model = Sequential(self.conv1, *self.middle_blocks, self.final)

    def initial_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Builds the initial convolutional block of the network.

        This block consists of:
            - Conv2d
            - ReLU activation

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            kernel_size (int):
                Size of the convolutional kernel.
            stride (int):
                Convolution stride.
            padding (int):
                Zero-padding to add to each side.

        Returns:
            nn.Sequential:
                A sequential block with Conv2d and ReLU.
        """

        initial = Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            ReLU(inplace=True),
        )
        return initial

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Builds an intermediate convolutional block for the DBP architecture.

        This block includes:
            - Conv2d
            - BatchNorm2d
            - ReLU activation

        Args:
            in_channels (int):
                Number of input feature channels.
            out_channels (int):
                Number of output feature channels.
            kernel_size (int):
                Size of the convolutional kernel.
            stride (int):
                Convolution stride.
            padding (int):
                Zero-padding to apply.

        Returns:
            nn.Sequential:
                A sequential block with Conv2d, BatchNorm2d, and ReLU.
        """

        convolution = Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
        )
        return convolution

    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Builds the final convolutional layer that produces the output image.

        This layer does not include an activation function.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels (typically 1).
            kernel_size (int):
                Size of the convolutional kernel.
            stride (int):
                Convolution stride.
            padding (int):
                Zero-padding to apply.

        Returns:
            nn.Conv2d: Final convolutional layer.
        """

        final = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        return final

    def forward(self, x):
        """
        Defines the forward pass of the DBP network.

        The input is passed sequentially through:
            - Initial convolutional block
            - Fifteen intermediate convolutional blocks
            - Final convolutional layer

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where:
                B = batch size,
                C = number of backprojection channels (`n_single_BP`),
                H, W = spatial dimensions.

        Returns:
            torch.Tensor:
                Output tensor of shape (B, 1, H, W), representing the reconstructed image.
        """

        # initial part
        conv1 = self.conv1(x)

        # middle part
        middle = conv1
        for block in self.middle_blocks:
            middle = block(middle)

        # final part
        final_layer = self.final(middle)

        return final_layer


class DeepFusionBPNetwork(LIONmodel):  # (Module) -> (LIONmodel)
    """
    Neural architecture for Deep Fusion Backprojection (DeepFusionBP).

    This module implements the full reconstruction pipeline including:
        - A learnable frequency-domain filter (shared or per-angle).
        - Angular interpolation using a stack of 1D residual convolutions.
        - Differentiable single-angle backprojections via Tomosipo.
        - A deep CNN (based on DBP) to fuse and denoise reconstructed views.

    Args:
        angles_sparse (torch.Tensor):
            Sparse subset of projection angles.
        src_orig_dist (float):
            Source-to-origin distance used in CT geometry.
        num_detectors (int):
            Number of detector bins per projection.
        num_angles (int):
            Number of sparse angles used in this configuration.
        vg (ts.VolumeGeometry):
            Tomosipo volume geometry for reconstruction.
        A (ts.Operator):
            Tomosipo projection operator (unused here but stored).
        filter_type (str):
            Either "Filter I" (shared filter) or "per-angle".
        device (torch.device):
            Device where the model will run (e.g., "cuda").

    Forward Input:
        x (Tensor): Input sinogram of shape [B, 1, A, D], where:
            - B: batch size,
            - A: number of sparse projection angles,
            - D: number of detector bins.

    Returns:
        Tensor:
            Reconstructed CT images of shape [B, 1, H, W].
    """

    def __init__(
        self, geometry: ct.Geometry, model_parameters: LIONParameter = None
    ):  # delete angles_sparse,src_orig_dist,vg
        super().__init__(model_parameters, geometry)

        self._make_operator()

        self.num_detectors = geometry.detector_shape[1]
        self.view_angles = len(geometry.angles)
        self.angles_sparse = geometry.angles  # new, may need to be modify
        self.src_orig_dist = geometry.dso  # new, may need to be modify
        self.vg = ts.volume(
            shape=(
                geometry.image_shape[0],
                geometry.image_shape[1],
                geometry.image_shape[2],
            )
        )  # new

        # Padding parameters
        self.projection_size_padded = self.compute_projection_size_padded()
        self.padding = self.projection_size_padded - self.num_detectors

        # Initialize Ram-Lak filter in frequency domain
        ram_lak = self.ram_lak_filter(self.projection_size_padded)
        if self.model_parameters.filter_type == "Filter I":
            self.learnable_filter = LearnableFilter(ram_lak, per_angle=False)
        else:
            self.learnable_filter = LearnableFilter(
                ram_lak, per_angle=True, num_angles=self.view_angles
            )

        # Interpolation blocks
        self.interpolator_1 = IntermediateResidualBlock(1)
        self.interpolator_2 = IntermediateResidualBlock(1)
        self.interpolator_3 = IntermediateResidualBlock(1)
        self.interpolator_conv = Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        # Single views backprojection
        self.back_projections = single_back_projections(
            self.angles_sparse, self.src_orig_dist, self.num_detectors, self.vg
        )

        # DBP_model for ct reconstructions
        self.dbp_layer = DBP_block(len(self.angles_sparse))

    def compute_projection_size_padded(self):
        """
        Computes the padded projection width for frequency-domain filtering.

        This ensures the sinogram is padded to the nearest power of two to avoid
        aliasing artifacts when applying FFT-based filters.

        Returns:
            int:
                Projection size after zero-padding.
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

    # New
    @staticmethod
    def default_parameters():
        DFuBP_params = LIONModelParameter()
        DFuBP_params.filter_type = "Filter I"
        DFuBP_params.mode = "ct"
        DFuBP_params.model_input_type = ModelInputType.SINOGRAM
        return DFuBP_params

    # new

    def forward(self, x):
        """
        Performs forward pass through the DeepFusionBP network.

        The input sinogram goes through:
            1. Learnable frequency-domain filtering (Ram-Lak initialized).
            2. Angular interpolation via 1D depthwise convolutions.
            3. Single-angle differentiable backprojections (Tomosipo).
            4. A deep CNN that fuses and denoises the backprojections.

        Args:
            x (torch.Tensor): Input sinogram of shape [B, 1, A, D], where:
                - B: batch size,
                - A: number of angles (sparse),
                - D: number of detectors.

        Returns:
            torch.Tensor: Reconstructed image of shape [B, 1, H, W], where H=W=362.
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
        x = x.view(-1, self.view_angles, self.num_detectors).unsqueeze(1)

        # Differentiable backprojection
        projections = self.back_projections(x)

        # reconstruction with DBP model
        img = self.dbp_layer(projections)

        return img
