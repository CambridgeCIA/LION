# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Andrea Sainz Bear
# =============================================================================
from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType

from LION.utils.parameter import LIONParameter
import LION.operators.ct.ct_geometry as ct


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
from LION.utils.citation import citation

# from .model import ModelBase


class LearnableFilter(Module):
    """A learnable filter of the projections, applied in the frequency domain
    (it replaces FBP's ramp filter): ``init_filter`` (its initial frequency
    response), one shared by every angle, or one per angle (``per_angle``, with
    ``num_angles``)."""

    def __init__(self, init_filter, per_angle=False, num_angles=None):
        super().__init__()
        self.per_angle = per_angle

        if per_angle:
            if num_angles is None:
                raise ValueError("num_angles must be provided when per_angle=True")
            filters = torch.stack(
                [init_filter.clone().detach() for _ in range(num_angles)]
            )
            self.register_parameter("weights", Parameter(filters))
        else:
            filters = torch.stack([init_filter.clone().detach()])  # shape: (1, D)
            self.register_parameter("weights", Parameter(filters))

    def forward(self, x):
        """The projections ``x`` (B, angles, detectors) filtered, each along
        the detector."""
        ftt1d = torch.fft.fft(x, dim=-1)
        if self.per_angle:
            filtered = ftt1d * self.weights[None, :, :]
        else:
            filter_shared = self.weights.expand(ftt1d.shape[1], -1)
            filtered = ftt1d * filter_shared[None, :, :]
        return torch.fft.ifft(filtered, dim=-1).real


class IntermediateResidualBlock(Module):
    """A residual block of the interpolation network: a depthwise 1D
    convolution along the detector, batch normalisation and PReLU, each
    projection on its own (as in DeepFBP)."""

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
        """``x`` plus the block of it."""
        return x + self.block(x)


class DenoisingBlock(Module):
    """The image denoiser of FusionFBP, adapted from DBP: a convolution and
    ReLU, fifteen convolution, batch normalisation and ReLU layers, and a last
    convolution, on one image."""

    def __init__(self):

        # Initialize the base training infrastructure
        super().__init__()

        # initial layer
        self.conv1 = self.initial_layer(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
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
        """The first layer: a convolution and ReLU."""

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
        """An intermediate layer: a convolution, batch normalisation and ReLU."""

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
        """The last layer: a convolution, no activation."""

        final = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        return final

    def forward(self, x):
        """The denoised images ``x`` (B, 1, H, W)."""

        # initial part
        conv1 = self.conv1(x)

        # middle part
        middle = conv1
        for block in self.middle_blocks:
            middle = block(middle)

        # final part
        final_layer = self.final(middle)

        return final_layer


class FusionFBPNetwork(LIONmodel):
    """FusionFBP: DeepFBP's learnable filter, interpolation network and
    backprojection, followed by DBP's deep CNN as the denoiser. CT only. Takes
    the measurements (B, 1, angles, detectors).

    ``model_parameters`` (``default_parameters()``):

    - ``filter_type``: "Filter I" (one filter for every angle) or anything else
      (one filter per angle);
    - ``fft_padding_twice_detectors``: the projections zero-padded to at least
      twice the detectors before filtering (True);
    - ``normalizer``: divide the backprojection by the backprojection of ones
      (True)."""

    def __init__(
        self, operator_params: ct.Geometry, model_parameters: LIONParameter = None
    ):
        super().__init__(model_parameters, operator_params)
        self._make_operator(autograd=True)

        self.num_detectors = operator_params.detector_shape[1]
        self.num_angles_ = len(operator_params.angles)

        # Padding parameters
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
        sinogram_ones = torch.ones((1, 1, self.num_angles_, self.num_detectors))
        self.register_buffer(
            "tomosipo_normalizer", self.AT(sinogram_ones) + 1e-6, persistent=False
        )

        # Denoising blocks
        self.denoiser = DenoisingBlock()

    def compute_projection_size_padded(self):
        """The FFT length of the filter: the next power of two of at least
        twice the number of detectors, so filtering isn't circular
        (``fft_padding_twice_detectors``; False: of the number of detectors, as
        older versions did)."""
        n = self.num_detectors
        if self.model_parameters.fft_padding_twice_detectors:
            n = 2 * n
        return 2 ** int(
            torch.ceil(torch.log2(torch.tensor(n, dtype=torch.float32))).item()
        )

    def ram_lak_filter(self, size):
        """The Ram-Lak (ramp) filter of FFT length ``size``, in the frequency
        domain: the learnable filter's initial value."""

        steps = int(size / 2 + 1)
        ramp = torch.linspace(0, 1, steps, dtype=torch.float32)
        down = torch.linspace(1, 0, steps, dtype=torch.float32)
        f = torch.cat([ramp, down[:-2]])
        return f

    @staticmethod
    def default_parameters():
        """One shared filter, padded FFT, normalised backprojection (see the
        class)."""
        FuFBP_params = LIONModelParameter()
        FuFBP_params.filter_type = "Filter I"
        FuFBP_params.model_input_type = ModelInputType.SINOGRAM
        # zero-pad the projections to >= 2x the detectors before the FFT filter
        FuFBP_params.fft_padding_twice_detectors = True
        FuFBP_params.normalizer = True
        return FuFBP_params

    def forward(self, x):

        # Initial shape: [B, 1, A, D]
        """The reconstructions (B, 1, H, W) of the sinograms ``x`` (B, 1,
        angles, detectors): filter, interpolation, backprojection, denoising.
        """
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
        x = x.view(-1, self.num_angles_, self.num_detectors).unsqueeze(1)

        # Differentiable backprojection
        img = self.AT(x)

        # normlise images
        if self.model_parameters.normalizer == True:
            img = img / self.tomosipo_normalizer

        # Denoising network
        x = self.denoiser(img)

        return x

    @staticmethod
    def cite(cite_format="bib"):
        """Where FusionFBP comes from: the MPhil thesis that proposed it (Sainz
        Bear), BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @mastersthesis{sainzbear2025deepfbp,
            title={Deep Filtered Back Projection for CT Reconstruction},
            author={Sainz Bear, Andrea},
            school={Department of Applied Mathematics and Theoretical Physics (DAMTP), University of Cambridge},
            type={MPhil thesis (Data Intensive Science)},
            year={2025}
            }
            """,
            mla="""
            Sainz Bear, Andrea. "Deep Filtered Back Projection for CT Reconstruction." MPhil thesis (Data Intensive Science), DAMTP, University of Cambridge, 2025.
            """,
        )
