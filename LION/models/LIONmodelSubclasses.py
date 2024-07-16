from LIONmodel import LIONmodel
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
class LIONmodelPhantom(LIONmodel):
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

    # TODO: implement check and override for implicit sino2phantom (if checking datatype is possible)

# Wraps a LIONmodelPhantom object to turn it into LIONmodelSino object
# Adds a FBP operator before calling phantom -> phantom reconstruction

class LIONmodelSinoConstructor(LIONmodelSino):
    def __init__(self, baseLIONModelPhantom):
        assert isinstance(baseLIONModelPhantom)
        self.__dict__ = baseLIONModelPhantom.__dict__
        self.__baseObject__ = baseLIONModelPhantom

    def forward(self, x):
        B, C, W, H = x.shape

        image = x.new_zeros(B, 1, *self.geo.image_shape[1:])
        for i in range(B):
            aux = fdk(self.op, x[i, 0])
            aux = clip(aux, min=0)
            image[i] = aux
        
        return self.__baseObject__(image)

    def phantom2phantom(self, x):
        return self.__baseObject__(x)
    
    def __getattr__(self, attr):
        if attr in ['forward']:
            # return the requested method
            return self.forward
        else:
            return getattr(self.__baseObject__, attr)