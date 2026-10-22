# This file is part of LION library
# License : GPL-3
#
# Author  : Zakhar Shumaylov, Subhadip Mukherjee
# Modifications: Ander Biguri, Zakhar Shumaylov, Charlie Shoebridge
# =============================================================================
import torch.nn as nn
from LION.models.LIONmodel import LIONmodel, ModelInputType, LIONModelParameter
import LION.CTtools.ct_geometry as ct


class AR(LIONmodel):
    def __init__(self, model_parameters: LIONModelParameter, geometry: ct.Geometry):
        super().__init__(model_parameters, geometry)

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

        # each of the 4 stride-2 convolutions (kernel 5, padding 2) maps n to ceil(n / 2)
        height, width = self.geometry.image_shape[-2:]
        for _ in range(4):
            height, width = (height + 1) // 2, (width + 1) // 2
        self.fc = nn.Sequential(
            nn.Linear(128 * height * width, 256),
            self.leaky_relu,
            nn.Linear(256, 1),
        )

    def forward(self, image):
        output = self.convnet(image)
        output = output.view(image.size(0), -1)
        output = self.fc(output)
        return output

    @staticmethod
    def default_parameters():
        params = LIONModelParameter()
        params.model_input_type = ModelInputType.IMAGE
        return params

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Lunz, Sebastian, Ozan Öktem, and Carola-Bibiane Schönlieb")
            print('"Adversarial regularizers in inverse problems."')
            print("neural information processing systems 31 (2018).")
            print("arXiv:2008.02839 (2020).")
        elif cite_format == "bib":
            string = """
            @article{lunz2018adversarial,
            title={Adversarial regularizers in inverse problems},
            author={Lunz, Sebastian and {\"O}ktem, Ozan and Sch{\"o}nlieb, Carola-Bibiane},
            journal={Advances in neural information processing systems},
            volume={31},
            year={2018}
            }
            """
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )
