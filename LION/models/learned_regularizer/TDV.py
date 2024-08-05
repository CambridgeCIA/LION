# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


from LION.models import LIONmodel

from LION.utils.math import power_method
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct
import LION.CTtools.ct_utils as ct_utils
import LION.utils.utils as ai_utils
from LION.models.learned_regularizer.TDV_files.model import VNet

import numpy as np
from pathlib import Path
import warnings

import tomosipo as ts
from tomosipo.torch_support import to_autograd
from ts_algorithms import fdk

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDV(LIONmodel.LIONmodel):
    """Learn Primal Dual network"""

    def __init__(
        self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None
    ):

        if geometry_parameters is None:
            raise ValueError("Geometry parameters required. ")

        super().__init__(model_parameters, geometry_parameters)
        # Pass all relevant parameters to internal storage.
        # AItomotmodel does this:
        # self.geo = geometry_parameters
        # self.model_parameters = model_parameters

        # Create pytorch compatible operators and send them to autograd
        self._make_operator()
        self.op_norm = power_method(self.op)
        # Define step size
        # self.model_parameters.config['lambda']['init'] = 1e-3 / self.op_norm**2
        # self.model_parameters.config['lambda']['max'] = 1 / self.op_norm**2
        if (type(self.model_parameters.config) is dict):
            config = self.model_parameters.config
        else: 
            config = self.model_parameters.config.serialize()
        self.vn = VNet(config,self.A,self.AT,power_method(self.op),efficient=True)

    @staticmethod
    def default_parameters():
        params = LIONParameter()
        params.config = {'S': 10, 
         'R': {'type': 'tdv', 
               'config': {'in_channels': 1, 'out_channels': 1, 'num_features': 16, 'num_scales': 5, 'num_mb': 5, 'multiplier': 1, 'efficient': True}}, 
         'D': {'type': 'denoise', 'config': {'use_prox': False}}, 
         'T_mode': 'learned', 
        #  'T': {'init': 1e1, 'min': 0, 'max': 1000}, 
        'T': {'init': 1e1, 'min': 0, 'max': 1e5}, 
         'lambda_mode': 'learned', 
         'lambda': {'init': 5e0, 'min': 0, 'max': 20}, 
         'pad': 0}
        
        params.mode = "ct"
        return params


    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Adler, Jonas, and Öktem, Ozan.")
            print('"Learned primal-dual reconstruction."')
            print("\x1B[3mIEEE transactions on medical imaging \x1B[0m")
            print("37.6 (2018): 1322-1332.")
        elif cite_format == "bib":
            string = """
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
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )

    def forward(self, g):
        """
        g: sinogram input
        """
        
        B, C, W, H = g.shape

        image = g.new_zeros(B, 1, *self.geo.image_shape[1:])
        for i in range(B):
            aux = fdk(self.op, g[i, 0])
            aux = torch.clip(aux, min=0)
            image[i] = aux
        # print(image)
        image=self.vn(image,g)
        # initialize parameters
        
        return image[-1]
    
    def output(self,g):
        return self.forward(g)
