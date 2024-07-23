# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Hong Ye Tan
# Modifications: -
# =============================================================================

# Lionmodels
from LION.models.LIONmodel import LIONmodel
from LION.models.LIONmodelSubclasses import LIONmodelSino, LIONmodelPhantom, forward_decorator
from LION.experiments.ct_experiments import Experiment

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# imports related to class organization
from abc import ABC, abstractmethod, ABCMeta

class LIONreconstructor(nn.Module):
    def __init__(
            self, 
            model, 
            data = None) -> None:
        
        assert isinstance(model, LIONmodel), "model must be a LIONmodel"
        super().__init__()
        
        self.model = model
        self.model.eval()


        if isinstance(self.model, LIONmodelPhantom) and not isinstance(self.model, LIONmodelSino):
            self.sino2recon = forward_decorator(self.model.forward)

        if data is not None:
            assert isinstance(data, Experiment) or isinstance(data, Dataset) or torch.is_tensor(data), "data must be Experiment, Dataset or batched input"
            if torch.is_tensor(data):
                assert len(data.shape) == 3 or len(data.shape) == 4, f"input Tensor must have shape (C,W,H) or (C,D,H,W), currently {data.shape}"
        
        self.dataloader = data # Contains parameters of images to be reconstructed.

    def forward(self, *args, **kwargs):
        if args:
            data = args[0]
            assert isinstance(data, Experiment) or isinstance(data, Dataset) or torch.is_tensor(data), "data must be Experiment, Dataset or batched input"
        elif self.dataloader:
            data = self.dataloader
        else:
            raise RuntimeError("Either pass data when constructing LIONreconstructor, or pass data into the forward operator.")

        if isinstance(data, Experiment):
            data_iterator = data.get_testing_dataset() # TODO get iterator
        elif isinstance(data, Dataset):
            data_iterator = None # TODO get iterator
        elif torch.is_tensor(data):
            data_iterator = [data]
        else:
            raise NotImplementedError(f"Reconstruction is not supported for data of type {type(self.data)}")
        
        if isinstance(self.model, LIONmodelSino):
            pass
        elif isinstance(self.model, LIONmodelPhantom):
            pass
        else:
            raise NotImplementedError(f"Reconstruction is not supported for model of type {type(self.model)}")


        # TODO iterate
        for dat in data_iterator:
            pass # todo

    def reconstructSino(self, sino):
        if isinstance(self.model, LIONmodelSino):
            return self.model(sino)
        elif isinstance(self.model, LIONmodelPhantom):
            return self.sino2recon(sino)
        else:
            raise NotImplementedError("Did not pass LIONmodelSino or LIONmodelPhantom")

    def reconstructPhantom(self, phantom):
        if not isinstance(self.model, LIONmodelPhantom):
            raise TypeError(f"Passed class is {self.model.__class__} which is not an instance of LIONmodelPhantom")
        if isinstance(self.model, LIONmodelSino):
            return self.model.phantom2phantom(phantom)
            # use phantom2phantom
        else:
            return self.model(phantom)

        

# Postprocessing eg FBPConvNet: takes sinogram and performs FDK reconstruction
# iterative unrolled: also takes sinogram
# class End2EndReconstructor(LIONreconstructor):
#     def __init__(
#             self, 
#             model, 
#             data) -> None:
#         super().__init__(model, data)
