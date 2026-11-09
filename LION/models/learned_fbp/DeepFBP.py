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
    Sequential,
    Conv1d,
    Conv2d,
    BatchNorm1d,
    PReLU,
    ReLU,
    Parameter,
)
from torch.nn.functional import pad
from LION.utils.citation import citation

# from model import ModelBase


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


class DenoisingResidualBlock(Module):
    """A 2D residual block of the denoiser: two 3x3 convolutions with a ReLU
    between them, added to the input."""

    def __init__(self, channels):
        super().__init__()
        self.block = Sequential(
            Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            ReLU(inplace=True),
            Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        """``x`` plus the block of it."""
        return x + self.block(x)


class DeepFBPNetwork(LIONmodel):
    """DeepFBP (Tan et al., 2024): FBP with learned parts. A learnable filter
    (initially Ram-Lak), an interpolation network (1D convolutions along the
    detector, each projection on its own), a differentiable backprojection
    (normalised by the backprojection of ones), and a residual CNN denoiser
    with a ReLU output. CT only. Takes the measurements (B, 1, angles,
    detectors).

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
        sinogram_ones = torch.ones((1, 1, self.num_angles_, self.num_detectors))
        self.register_buffer(
            "tomosipo_normalizer", self.AT(sinogram_ones) + 1e-6, persistent=False
        )  # Avoid division by zero in normalization

        # Denoising blocks
        self.denoising_conv_1 = Conv2d(1, 64, kernel_size=1)
        self.denoising_conv_2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoising_res_1 = DenoisingResidualBlock(64)
        self.denoising_res_2 = DenoisingResidualBlock(64)
        self.denoising_res_3 = DenoisingResidualBlock(64)
        self.denoising_output = Sequential(
            Conv2d(64, 1, kernel_size=3, padding=1),
            ReLU(),  # non-negative attenuation; no upper bound
        )

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
        DFBP_params = LIONModelParameter()
        DFBP_params.filter_type = "Filter I"
        DFBP_params.model_input_type = ModelInputType.SINOGRAM
        # zero-pad the projections to >= 2x the detectors before the FFT filter
        DFBP_params.fft_padding_twice_detectors = True
        DFBP_params.normalizer = True
        return DFBP_params

    def forward(self, x):
        """The reconstructions (B, 1, H, W), non-negative, of the sinograms
        ``x`` (B, 1, angles, detectors): filter, interpolation, backprojection,
        denoising."""

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

    @staticmethod
    def cite(cite_format="bib"):
        """Where DeepFBP comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{tan2024deepfbp,
            title={Deep Filtered Back Projection for CT Reconstruction},
            author={Tan, X. and Liu, Xuan and Xiang, Kai and Wang, Jing and Tan, Shan},
            journal={IEEE Access},
            volume={12},
            year={2024},
            doi={10.1109/ACCESS.2024.3357355}
            }
            """,
            mla="""
            Tan, X., Xuan Liu, Kai Xiang, Jing Wang, and Shan Tan. "Deep Filtered Back Projection for CT Reconstruction." IEEE Access 12 (2024).
            """,
        )
