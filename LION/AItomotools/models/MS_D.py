# This file is part of AItomotools library
# License : BSD-3
#
# Author  : See ./MS-D. Daniel Pelt et al
# Modifications: Ander Biguri
# =============================================================================


import torch
import torch.nn as nn
from AItomotools.models import AItomomodel
from AItomotools.utils.parameter import Parameter
import msd_pytorch as msd


class MS_D(AItomomodel.AItomoModel):
    def __init__(self, model_parameters=None):
        if model_parameters is None:
            model_parameters = MS_D.default_parameters()
        super().__init__(model_parameters)

        if model_parameters.type == "regression":
            model = msd.MSDRegressionModel(
                model_parameters.c_in,
                model_parameters.c_out,
                model_parameters.depth,
                model_parameters.width,
                dilations=model_parameters.dilations,
                loss="L2",
            )

        elif model_parameters.type == "segmentation":
            if model_parameters.num_labels is None:
                raise ValueError(
                    "For a segmentation network, please set the model_parameters.num_labels vairable to the number of labels in the training set"
                )
            model = msd.MSDSegmentationModel(
                model_parameters.c_in,
                model_parameters.num_labels,
                model_parameters.depth,
                model_parameters.width,
                dilations=model_parameters.dilations,
            )
        # We don't want MS-D to define our optimizer.
        model.optimizer = None
        self.net = model.net

    @staticmethod
    def default_parameters(mode="regression"):
        param = Parameter()
        if mode == "regression":
            param.type = "regression"
            param.c_in = 1
            param.c_out = 1
            param.depth = 100
            param.width = 1
            param.dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        elif mode == "segmentation":
            param.type = "segmentation"
            param.num_labels = None
            param.depth = 100
            param.width = 1
            param.dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return param

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Pelt, Daniël M., and James A. Sethian.")
            print(
                '"A mixed-scale dense convolutional neural network for image analysis."'
            )
            print("\x1B[3mProceedings of the National Academy of Sciences  \x1B[0m")
            print("115.2 (2018): 254-259.")
        elif cite_format == "bib":
            string = """
            @article{pelt2018mixed,
            title={A mixed-scale dense convolutional neural network for image analysis},
            author={Pelt, Dani{\"e}l M and Sethian, James A},
            journal={Proceedings of the National Academy of Sciences},
            volume={115},
            number={2},
            pages={254--259},
            year={2018},
            publisher={National Acad Sciences}
            }
            """
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )

    def forward(self, x):
        return self.net(x)
