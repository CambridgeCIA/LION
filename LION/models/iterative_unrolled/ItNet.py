# This file is part of LION library
# License : BSD-3
#
# Author  : Oliver Coughlan
# Modifications: Ander Biguri
# =============================================================================


import numpy as np
import torch
import torch.nn as nn

from ts_algorithms import fdk
from collections import OrderedDict


import LION.CTtools.ct_geometry as ct

from LION.utils.parameter import LIONParameter
from LION.models import LIONmodel

from LION.models.CNNs.UNets.Unet import UNet


class ItNet(LIONmodel.LIONmodel):
    def __init__(self, geometry: ct.Geometry, model_parameters: LIONParameter = None):
        if geometry is None:
            raise ValueError("Geometry parameters required. ")

        super().__init__(model_parameters, geometry)

        # Create layers per iteration
        for i in range(self.model_parameters.n_iters):
            self.add_module(f"Unet_{i}", UNet(self.model_parameters.Unet_params))

        # Create pytorch compatible operators and send them to autograd
        self._make_operator()

        # Define step size
        if self.model_parameters.step_size is None:
            # compute step size
            self.model_parameters.step_size = np.array(
                [1] * self.model_parameters.n_iters
            )

        elif not hasattr(self.model_parameters.step_size, "__len__"):
            self.model_parameters.step_size = np.array(
                self.model_parameters.step_size * self.model_parameters.n_iters
            )
        elif len(self.model_parameters.step_size) == self.model_parameters.n_iters:
            self.model_parameters.step_size = np.array(self.model_parameters.step_size)
        else:
            raise ValueError("Step size not understood")

        #  Are we learning the step? (with the above initialization)
        if self.model_parameters.learned_step:
            # Enforce positivity by making it 10^step
            if self.model_parameters.step_positive:
                self.step_size = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.ones(1)
                            * 10 ** np.log10(self.model_parameters.step_size[i])
                        )
                        for i in range(self.model_parameters.n_iters)
                    ]
                )
            # Negatives OK
            else:
                self.step_size = nn.ParameterList(
                    [
                        nn.Parameter(torch.ones(1) * self.model_parameters.step_size[i])
                        for i in range(self.model_parameters.n_iters)
                    ]
                )
        else:
            self.step_size = (
                torch.ones(self.model_parameters.n_iters)
                * self.model_parameters.step_size
            )

    @staticmethod
    def default_parameters():
        param = LIONParameter()
        param.learned_step = True
        param.step_positive = False
        param.step_size = [1.1183, 1.3568, 1.4271, 0.0808]
        param.n_iters = 4
        param.Unet_params = UNet.default_parameters()
        param.mode = "ct"
        return param

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Martin Genzel, Ingo Guhring, Jan Macdonald, and Maximilian März. ")
            print(
                '""Near-exact recovery for tomographic inverse problems via deep learning." '
            )
            print("\x1B[3m ICML 2022 \x1B[0m")
            print("(pp. 7368-7381). PMLR")

        elif cite_format == "bib":
            string = """
            @inproceedings{genzel2022near,
            title={Near-exact recovery for tomographic inverse problems via deep learning},
            author={Genzel, Martin and G{\"u}hring, Ingo and Macdonald, Jan and M{\"a}rz, Maximilian},
            booktitle={International Conference on Machine Learning},
            pages={7368--7381},
            year={2022},
            organization={PMLR}
            }"""
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )

    def forward(self, sino):

        B, C, W, H = sino.shape
        img = sino.new_zeros(B, 1, *self.geometry.image_shape[1:])
        update = sino.new_zeros(B, 1, *self.geometry.image_shape[1:])
        # Start from FDK
        for i in range(sino.shape[0]):
            img[i] = fdk(self.op, sino[i])

        for i in range(self.model_parameters.n_iters):
            unet = getattr(self, f"Unet_{i}")
            img = unet(img)

            for j in range(img.shape[0]):
                update[j] = self.step_size[i] * fdk(self.op, self.op(img[j]) - sino[j])
            img = img - update

        return img
