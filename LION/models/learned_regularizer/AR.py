# This file is part of LION library
# License : BSD-3
#
# Author  : Zakhar Shumaylov, Subhadip Mukherjee
# Modifications: Ander Biguri, Zakhar Shumaylov, Charlie Shoebridge
# =============================================================================
import torch.nn as nn
from LION.models.LIONmodel import LIONmodel, ModelInputType, ModelParams
import LION.CTtools.ct_geometry as ct


class ARNetworkParams(ModelParams):
    def __init__(
        self,
        n_channels: int,
    ):
        super().__init__(model_input_type=ModelInputType.IMAGE)
        self.n_channels = n_channels


class ARNetwork(LIONmodel):
    def __init__(
        self, model_parameters: ARNetworkParams, geometry_parametrs: ct.Geometry
    ):
        super().__init__(model_parameters, geometry_parametrs)

        self.leaky_relu = nn.LeakyReLU()
        self.convnet = nn.Sequential(
            nn.Conv2d(
                self.model_parameters.n_channels, 16, kernel_size=(5, 5), padding=2
            ),
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

        size = self.geo.image_shape[-1]
        self.fc = nn.Sequential(
            nn.Linear(128 * (size // 2**4) ** 2, 256),
            self.leaky_relu,
            nn.Linear(256, 1),
        )

    def forward(self, image):
        output = self.convnet(image)
        output = output.view(image.size(0), -1)
        output = self.fc(output)
        return output

    # needs changing to original paper
    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Mukherjee, Subhadip, et al.")
            print('"Data-Driven Convex Regularizers for Inverse Problems."')
            print(
                "ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024"
            )
            print("arXiv:2008.02839 (2020).")
        elif cite_format == "bib":
            string = """
            @inproceedings{mukherjee2024data,
            title={Data-Driven Convex Regularizers for Inverse Problems},
            author={Mukherjee, S and Dittmer, S and Shumaylov, Z and Lunz, S and {\"O}ktem, O and Sch{\"o}nlieb, C-B},
            booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
            pages={13386--13390},
            year={2024},
            organization={IEEE}
            }
            """
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )
