#%% This is a template of a model for AItomotools. 
# 
# It gives an easy start on how to build a model that fits well with the library. 
# This file has text and examples explaining what you need. 
# Use ut as a template to build your models on. 
#
# The reason to not have all models inherint from this is future-proofing. There is too much complexity in AI to start making
# strict limits on what the models can have. 

#%% Imports

# You will want to import Parameter, as all models must save and use Parameters. 
from AItomotools.utils.parameter import Parameter
# (optional) Given this is a tomography library, it is likely that you will want to load geometries of the tomogprahic problem you are solving, e.g. a ct_geometry
import AItomotools.CTtools.ct_geometry as ct

# (optinal) If your model uses the operator (e.g. the CT operator), you may want to load it here. E.g. for tomosipo:
import tomosipo as ts
from tomosipo.torch_support import to_autograd

# some standard imports, e.g. 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lets define the class. This demo just shows a very simple model that uses both the opeartor of CT and a CNN layer, for demostrational purposes. 
# The model makes no sense, it only sits here for coding demostration purposes. 

class myModel(nn.Module):
    """ My model netweork (title)
    Some more info about it. 
    """

    # Initialization of the models should have only "Parameter()" classes. These should be "topic-wise", with at minimum 1 parameter object being passed. 
    # e.g. a Unet will have only 1 parameter (model_parameters), but the Learned Primal Dual will have 2, one for the model parameters and another one
    # for the geometry parameters of the inverse problem. 
    def __init__(
        self,
        model_parameters:Parameter,                # model parameters
        geometry_parameters:ct.Geometry            # (optional) if your model uses an operator, you may need its parameters. e.g. ct geometry parameters for tomosipo operators
    ):
        super().__init__() # Initialize parent classes. 

        # Pass all relevant parameters to internal storage. 
        self.geo=geometry_parameters
        self.model_parameters = model_parameters



        # (example) make some NN layers, maybe defined by the Parameter file
        #
        # in this case, model_parameters has .bias (True/False) and .channels (list with number of channels in each layer). 
        # for example, model_parameters.channels=[7 10 5 1], and model_parameters.bias=False
        #
        # Albeit this is here for demosntrational purposes, do not clog the __init__ function, add classes that contain subblocks (check LPD.py)
        layer_list = []
        for ii in range(len(self.model_parameters.channels-1)):
            layer_list.append(nn.Conv2d(self.model_parameters.channels[ii], self.model_parameters.channels[ii + 1], 3, padding=s1, bias=self.model_parameters.bias))
            # Have PReLUs all the way except the last
            if ii < layers - 1:
                layer_list.append(torch.nn.PReLU())
        self.block = nn.Sequential(*layer_list)


        # (optional) if your model is for CT reconstruction, you may need the CT operator defined with e.g. tomosipo. This is how its done.
        # model_parameters.mode contains the tomographic mode, e.g. 'ct'
        op = self.__make_operators(self.geo, self.model_parameters.mode)
        self.op=op
        self.A = to_autograd(op, num_extra_dims=1)
        self.AT = to_autograd(op.T, num_extra_dims=1)


    # All classes in AItomotools must have a static method called default_parameters(). 
    # This should return the parameters from the paper the model is from
    @staticmethod
    def default_parameters(mode='ct') -> Parameter:
        # create empty object
        model_params=Parameter()
        # fill
        model_params.mode=mode
        model_params.channels=[7, 10, 5, 1]
        model_params.bias=False
        
        return model_params

    # All classes should have this method, just change the amount of Parameters it returns of you have more/less
    def get_parameters(self):
        return self.model_parameters, self.geo

    # All classes shoudl have this method. Yhis is the example for Learned Primal Dual. 
    # You can obtain this exact text from Google Scholar's page of the paper. 
    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Adler, Jonas, and Ozan Öktem.")
            print("\"Learned primal-dual reconstruction.\"")
            print("\x1B[3mIEEE transactions on medical imaging \x1B[0m")
            print("37.6 (2018): 1322-1332.")
        elif cite_format=="bib":
            string="""
            @article{adler2018learned,
            title={Learned primal-dual reconstruction},
            author={Adler, Jonas and {\"O}ktem, Ozan},
            journal={IEEE transactions on medical imaging},
            volume={37},
            number={6},
            pages={1322--1332},
            year={2018},
            publisher={IEEE}
            }"""
            print(string)
        else:
            raise AttributeError("cite_format not understood, only \"MLA\" and \"bib\" supported")      
        
    # (optional) if your model uses a CT operator, this will create it, for tomosipo backend. 
    @staticmethod
    def __make_operators(geo, mode='ct'):
        if mode.lower() != "ct":
            raise NotImplementedError("Only CT operators supported")
        vg = ts.volume(shape=geo.image_shape, size=geo.image_size)
        pg = ts.cone(
            angles=geo.angles,
            shape=geo.detector_shape,
            size=geo.detector_size,
            src_orig_dist=geo.dso,
            src_det_dist=geo.dsd,
        )
        A = ts.operator(vg, pg)
        return A

    # Mandatory for all models, the forwar pass. 
    def forward(self, g):
        """
        g: sinogram input
        """

        B, C, W, H = g.shape

        # Have some input parsing
        if C != 1:
            raise NotImplementedError("Only 2D CT images supported")
        if len(self.geo.angles) != W or self.geo.detector_shape[1] != H:
            raise ValueError("geo description and sinogram size do not match")

        # As an example our network backprojects and then uses conv layers.
        # This is bogus code, don't follow. 

        # Make some image-shaped channels
        f_network = g.new_zeros(B, model_params.channels[0], *self.geo.image_shape[1:])
        # Backproject.
        for i in range(B):
            aux=self.AT(g[i,0])
            for channel in range(model_params.channels[0]):
                f_network[i,channel]=aux
        # use channels
        f_out=self.block(f_network)
        return f_out
