# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Hong Ye Tan
# Modifications: -
# Created: 16 July 2024
# =============================================================================

from LION.models.LIONmodel import LIONmodel
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct
from abc import ABC, abstractmethod, ABCMeta
from ts_algorithms import fdk
from torch import clip


# Lightweight subclasses to define inputs

class LIONmodelSino(LIONmodel):
    # Consumes sinogram and outputs reconstructions
    def __init__(
        self,
        model_parameters: LIONParameter,  # model parameters
        geometry_parameters: ct.Geometry = None,  # (optional) if your model uses an operator, you may need its parameters. e.g. ct geometry parameters for tomosipo operators
    ):
        super().__init__(model_parameters, geometry_parameters)

    @abstractmethod
    def forward(self, x):
        pass
class LIONmodelRecon(LIONmodel):
    # Consumes phantom and outputs reconstructions
    def __init__(
        self,
        model_parameters: LIONParameter,  # model parameters
        geometry_parameters: ct.Geometry = None,  # (optional) if your model uses an operator, you may need its parameters. e.g. ct geometry parameters for tomosipo operators
    ):
        super().__init__(model_parameters, geometry_parameters)

    @abstractmethod
    def forward(self, x):
        pass

# Wraps a LIONmodelRecon object to turn it into LIONmodelSino object
# Adds a FBP operator before calling phantom -> phantom reconstruction


def forward_decorator(self, f):
    def wrapper(x):
        # print("in wrapper")
        B, C, W, H = x.shape
        # print(x.shape)
        image = x.new_zeros(B, 1, *self.geo.image_shape[1:])
        for i in range(B):
            aux = fdk(self.op, x[i, 0])
            aux = clip(aux, min=0)
            image[i] = aux
        return f(image)

    return wrapper

def Constructor(obj):
    obj.recon2recon = obj.forward
    obj.forward = forward_decorator(obj, obj.forward)
    # create a new class that extends both the existing object class and LIONmodelSino
    # new class will extend both LIONmodelSino and LIONmodelRecon
    obj.__class__ = type(f"{obj.__class__.__name__}Sino", (obj.__class__, LIONmodelSino), obj.__dict__)
    return obj
