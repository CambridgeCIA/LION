# This file is part of LION library
# License : BSD-3
#
# Author  : Zakhar Shumaylov, Subhadip Mukherjee
# Modifications: Ander Biguri, Zakhar Shumaylov
# =============================================================================


from typing import Optional
import torch
import torch.nn as nn
from LION.classical_algorithms.fdk import fdk
from LION.exceptions.exceptions import WrongInputTypeException
from LION.models.LIONmodel import LIONmodel, ModelInputType, ModelParams
import LION.CTtools.ct_geometry as ct
import numpy as np
from LION.utils.math import power_method


class network(nn.Module):
    def __init__(self, n_chan=1):
        super(network, self).__init__()

        self.leaky_relu = nn.LeakyReLU()
        self.convnet = nn.Sequential(
            nn.Conv2d(n_chan, 16, kernel_size=(5, 5), padding=2),
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

        size = 1024
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


class ARParams(ModelParams):
    def __init__(
        self,
        early_stopping: bool = False,
        no_steps: int = 150,
        step_size: float = 1e-6,
        momentum: float = 0.5,
        beta_rate: float = 0.95,
    ):
        super().__init__(model_input_type=ModelInputType.SINOGRAM)
        self.early_stopping = early_stopping
        self.no_steps = no_steps
        self.step_size = step_size
        self.momentum = momentum
        self.beta_rate = beta_rate


class AR(LIONmodel):
    def __init__(
        self,
        network: LIONmodel,
        geometry_parameters: ct.Geometry,
        model_parameters: Optional[ARParams] = None,
    ):

        super().__init__(model_parameters, geometry_parameters)
        self.model_parameters: ARParams
        self._make_operator()
        self.network = network
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # First Conv
        self.estimate_alpha()
        self.step_amounts = torch.tensor([150.0])
        self.op_norm = float(power_method(self.op))
        self.model_parameters.step_size = 0.2 / (self.op_norm) ** 2

    def forward(self, y):
        """Expects y to be in the measurement domain.

        Args:
            y (_type_): _description_

        Returns:
            _type_: _description_
        """

        if self.training:
            return self.pool(self.network(
                fdk(y, self.op)
                if self.network.model_parameters.model_input_type
                == ModelInputType.IMAGE
                else y
            ))
        else:  # in eval (validataion or testing, so actually want to do the GD)
            return self.output(y)

    def estimate_alpha(self, dataset=None):
        self.alpha = 1.0
        if dataset is None:
            self.alpha = 1.0
        else:
            residual = 0.0
            for _, (data, target) in enumerate(dataset):
                residual += torch.norm(
                    self.AT(self.A(target) - data), dim=(2, 3)
                ).mean()
                # residual += torch.sqrt(((self.AT(self.A(target) - data))**2).sum())
            self.alpha = residual / len(dataset)
        print("Estimated alpha: " + str(self.alpha))


    def var_energy(self, x, y):
        # return torch.norm(x) + 0.5*(torch.norm(self.A(x)-y,dim=(2,3))**2).sum()#self.lamb * self.forward(x).sum()
        return 0.5 * ((self.A(x) - y) ** 2).sum() + self.alpha * self.pool(self.network(x)).sum()

    ### What is the difference between .sum() and .mean()??? idfk but PSNR is lower when I do .sum

    def output(self, y: torch.Tensor):
        x = fdk(y, self.op)
        x = torch.nn.Parameter(x)
        y.grad

        optimizer = torch.optim.SGD(
            [x], lr=self.model_parameters.step_size, momentum=0.5
        )

        lr = self.model_parameters.step_size
        for i in range(self.model_parameters.no_steps):
            print(i)
            optimizer.zero_grad()

            energy = self.var_energy(x, y)
            energy.backward()

            assert x.grad is not None
            while (
                self.var_energy(x - x.grad * lr, y)
                > energy - 0.5 * lr * (x.grad.norm(dim=(2, 3)) ** 2).mean()
            ):
                lr = self.model_parameters.beta_rate * lr
            for g in optimizer.param_groups:
                g["lr"] = lr

            # what's going on with these?
            # if(j > self.step_amounts.mean().item()):
            #     # print('only for testing')
            #     x.clamp(min=0.0)
            #     return x
            # elif(lr * self.op_norm**2 < 1e-3):
            #     x.clamp(min=0.0)
            #     return x
            optimizer.step()
            # x.clamp(min=0.0) do we need this?
        self.scalar = False
        return x        


    # this functionality is now the responsibility of the LIONSolver
    # def normalise(self,x):
    #     return (x - self.model_parameters.xmin) / (self.model_parameters.xmax - self.model_parameters.xmin)
    # def unnormalise(self,x):
    #     return x * (self.model_parameters.xmax - self.model_parameters.xmin) + self.model_parameters.xmin

    @staticmethod
    def default_parameters() -> ARParams:
        param = ARParams(False, 150, 1e-6, 0.5, 0.95)
        return param

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
