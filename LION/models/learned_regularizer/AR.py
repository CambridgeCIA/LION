# This file is part of LION library
# License : GPL-3
#
# Author  : Zakhar Shumaylov, Subhadip Mukherjee
# Modifications: Ander Biguri, Zakhar Shumaylov, Charlie Shoebridge
# =============================================================================
import torch.nn as nn
from LION.utils.parameter import OperatorParameters
from LION.models.LIONmodel import LIONmodel, ModelInputType, LIONModelParameter
from LION.utils.citation import citation


class AR(LIONmodel):
    """Adversarial regulariser (Lunz, Öktem and Schönlieb, 2018): a CNN mapping
    an image to a number, low for clean images and high for bad
    reconstructions, trained by ``ARSolver`` and used by
    ``LearnedRegularizerReconstructor``. Six 5x5 convolutions (four strided)
    and two fully connected layers, as in the paper. Takes images of the
    operator's domain shape (its size sets the fully connected layer)."""

    def __init__(
        self,
        operator_params: OperatorParameters,
        model_parameters: LIONModelParameter = None,
    ):
        super().__init__(model_parameters, operator_params)

        self.leaky_relu = nn.LeakyReLU()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), padding=2),
            self.leaky_relu,
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding=2),
            self.leaky_relu,
            nn.Conv2d(32, 32, kernel_size=(5, 5), padding=2, stride=2),
            self.leaky_relu,
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2, stride=2),
            self.leaky_relu,
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=2, stride=2),
            self.leaky_relu,
            nn.Conv2d(64, 128, kernel_size=(5, 5), padding=2, stride=2),
            self.leaky_relu,
        )

        # The fully connected layer is sized by the image (the operator's
        # domain): each of the 4 stride-2 convolutions (kernel 5, padding 2)
        # maps n to ceil(n / 2)
        self._make_operator()
        height, width = self.op.domain_shape[-2:]
        for _ in range(4):
            height, width = (height + 1) // 2, (width + 1) // 2
        self.fc = nn.Sequential(
            nn.Linear(128 * height * width, 256),
            self.leaky_relu,
            nn.Linear(256, 1),
        )

    def forward(self, image):
        """The regulariser's value for each image of ``image`` (B, 1, H, W):
        (B, 1)."""
        output = self.convnet(image)
        output = output.view(image.size(0), -1)
        output = self.fc(output)
        return output

    @staticmethod
    def default_parameters():
        """None of its own: the architecture is the paper's."""
        params = LIONModelParameter()
        params.model_input_type = ModelInputType.IMAGE
        return params

    @staticmethod
    def cite(cite_format="bib"):
        """Where AR comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{lunz2018adversarial,
            title={Adversarial regularizers in inverse problems},
            author={Lunz, Sebastian and {\\"O}ktem, Ozan and Sch{\\"o}nlieb, Carola-Bibiane},
            journal={Advances in neural information processing systems},
            volume={31},
            year={2018}
            }
            """,
            mla="""
            Lunz, Sebastian, Ozan Öktem, and Carola-Bibiane Schönlieb
            "Adversarial regularizers in inverse problems."
            neural information processing systems 31 (2018).
            arXiv:2008.02839 (2020).
            """,
        )
