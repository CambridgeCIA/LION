# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Hong Ye Tan
# Modifications: -
# =============================================================================

from LION.models.LIONmodel import LIONmodel
from LION.models.LIONmodelSubclasses import (
    LIONmodelSino,
    LIONmodelRecon,
    forward_decorator,
)
from LION.experiments.ct_experiments import Experiment
from LION.CTtools.ct_utils import make_operator

import warnings
import numpy as np
import torch
import torch.nn as nn
import typing
from LION.CTtools.ct_geometry import Geometry
from tomosipo import Operator
from abc import ABC, abstractmethod, ABCMeta
from typing import Optional

# Credit: Zak, /TDV_files/model.py
class Dataterm(torch.nn.Module, ABCMeta):
    """
    Basic dataterm function
    """

    def __init__(self, geometry: Geometry):
        super(Dataterm, self).__init__()

    def forward(self, x, *args):
        raise NotImplementedError

    @abstractmethod
    def energy(self):
        raise NotImplementedError

    def prox(self, x, *args):
        raise NotImplementedError

    def grad(self, x, *args):
        raise NotImplementedError


class L2DenoiseDataterm_noise(Dataterm):
    def __init__(self, geometry: Geometry):
        super(L2DenoiseDataterm, self).__init__(geometry)

    def energy(self, x, z):
        return 0.5 * (x - z) ** 2

    def prox(self, x, z, tau):
        return (x + tau * z) / (1 + tau)

    def grad(self, x, z):
        return x - z


class L2DenoiseDataterm(Dataterm):
    def __init__(self, geometry: Geometry):
        super(L2DenoiseDataterm, self).__init__(geometry)
        self.A = make_operator(geometry)
        self.AT = self.A.T

    def energy(self, x, z):
        return (0.5 * (self.A(x) - z) ** 2).mean()

    def prox(self, x, z, tau):
        return 0  # (x + tau * z) / (1 + tau)

    def grad(self, x, z):
        return self.AT(self.A(x) - z)


# See Zak's branch: https://github.com/CambridgeCIA/LION/compare/main...Zakobian:LION:main
class LIONPnPreconstructor:
    def __init__(
        self,
        model: LIONmodelRecon,
        geometry: Geometry,
        dataterm: Optional[Dataterm] = None,
    ):
        self.geometry = geometry
        self.model = model
        if isinstance(model, LIONmodelSino) and isinstance(model, LIONmodelRecon):
            self.recon2recon = model.phantom2phantom
        elif isinstance(model, LIONmodelRecon):
            self.recon2recon = model.forward
        else:
            raise NotImplementedError(
                f"Model should be a LIONmodelRecon, currently {model.__class__}"
            )

        # initialize data term
        if dataterm is not None:
            self.dataterm = dataterm
        else:
            self.dataterm = L2DenoiseDataterm(geometry)

    def PGD():
        pass

    def ADMM():
        pass

    def DRS():
        pass

    def HQS():
        pass


class LIONARreconstructor:
    pass
