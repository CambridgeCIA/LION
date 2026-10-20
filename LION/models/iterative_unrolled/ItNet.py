# This file is part of LION library
# License : GPL-3
#
# Author  : Oliver Coughlan
# Modifications: Ander Biguri
# =============================================================================


import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict


import LION.CTtools.ct_geometry as ct

from LION.utils.parameter import LIONParameter
from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType

from LION.models.CNNs.UNets.Unet import UNet
from LION.classical_algorithms.fdk import fdk


class ItNet(LIONmodel):
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
            # one scalar: the same step for every iteration
            self.model_parameters.step_size = np.array(
                [self.model_parameters.step_size] * self.model_parameters.n_iters
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
            # a buffer, so it moves with the model (.to(device)) like the rest
            self.register_buffer(
                "step_size",
                torch.as_tensor(self.model_parameters.step_size, dtype=torch.float32),
                persistent=False,
            )

    @staticmethod
    def default_parameters():
        param = LIONModelParameter()
        param.learned_step = True
        param.step_positive = False
        param.step_size = [1.1183, 1.3568, 1.4271, 0.0808]
        param.n_iters = 4
        param.Unet_params = UNet.default_parameters()
        param.mode = "ct"
        param.model_input_type = ModelInputType.SINOGRAM

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
        # Start from FDK (the input data needs no gradient)
        img = fdk(sino, self.op, clip=False)

        for i in range(self.model_parameters.n_iters):
            unet = getattr(self, f"Unet_{i}")
            img = unet(img)
            # Data-consistency step, differentiable end to end: the residual uses
            # the autograd-wrapped operator and the FDK backprojects through
            # autograd too. Residuals are signed, so they are not clipped.
            residual = self.A(img) - sino
            img = img - self.step_size[i] * fdk(
                residual, self.op, clip=False, differentiable=True
            )

        return img
