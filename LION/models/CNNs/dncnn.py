# This file is part of LION library
# License : GPL-3
#
# Author  : Ferdia Sherry
# Modifications: -
# =============================================================================

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.utils.parameter import LIONParameter


from inspect import getmembers, isfunction
import torch
from LION.utils.citation import citation


class DnCNN(LIONmodel):
    """DnCNN (Zhang et al., 2017): a denoiser of ``blocks`` convolutional
    layers that learns the noise (``residual``: the output is the input minus
    it). Takes images.

    ``model_parameters`` (``default_parameters()``):

    - ``in_channels``: the images' channels (1);
    - ``int_channels``: the hidden layers' channels (64);
    - ``kernel_size``: 3x3;
    - ``blocks``: the number of layers (20);
    - ``residual``: learn the noise (True);
    - ``bias_free``: no biases, as bias-free denoisers (False);
    - ``act``: a ``torch.nn.functional`` activation ("leaky_relu");
    - ``enforce_positivity``: a ReLU on the output (False);
    - ``batch_normalisation``: as in the paper (True)."""

    def __init__(self, model_parameters: LIONParameter = None):
        super().__init__(model_parameters)

        model_parameters = self.model_parameters

        act = model_parameters.act.lower()
        if act in dict(getmembers(torch.nn.functional, isfunction)):
            self._act = torch.nn.functional.__dict__[act]
        else:
            raise ValueError(
                f"`torch.nn.functional` does not export a function '{model_parameters.act}'."
            )
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

    @staticmethod
    def default_parameters():
        """The paper's DnCNN: 20 layers of 64 channels with batch
        normalisation, learning the noise (see the class)."""
        params = LIONModelParameter()
        params.model_input_type = ModelInputType.IMAGE
        params.in_channels = 1
        params.int_channels = 64
        params.kernel_size = (3, 3)
        params.blocks = 20
        params.residual = True
        params.bias_free = False
        params.act = "leaky_relu"
        params.enforce_positivity = False
        params.batch_normalisation = True
        return params

    @staticmethod
    def cite(cite_format="bib"):
        """Where DnCNN comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{zhang2017beyond,
            title = {Beyond a {{Gaussian Denoiser}}: {{Residual Learning}} of {{Deep CNN}} for {{Image Denoising}}},
            author = {Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
            year = {2017},
            journal = {IEEE Transactions on Image Processing},
            volume = {26},
            number = {7},
            pages = {3142--3155},
            issn = {1057-7149},
            doi = {10.1109/TIP.2017.2662206}
            }
            """,
            mla="""
            Zhang, Kai, Wangmeng, Zuo, Yunjin, Chen, Deyu, Meng, Lei, Zhang.
            "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising".
            IEEE Transactions on Image Processing
            26. 7(2017): 3142–3155.
            """,
        )

    def forward(self, x):
        """The denoised images ``x`` (B, in_channels, H, W)."""
        z = self._act(self.lift(x))
        if self.model_parameters.batch_normalisation:
            for conv, bn in zip(self.convs, self.bns):
                z = self._act(bn(conv(z)))
        else:
            for conv in self.convs:
                z = self._act(conv(z))

        if self.model_parameters.residual:
            z = x - self.project(z)
        else:
            z = self.project(z)
        if self.model_parameters.enforce_positivity:
            return torch.nn.functional.relu(z)
        else:
            return z
