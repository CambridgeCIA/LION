# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Hong Ye Tan
# Modifications: -
# =============================================================================

# Lionmodels
from LION.models.LIONmodel import LIONmodel
from LION.experiments.ct_experiments import Experiment

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# imports related to class organization
from abc import ABC, abstractmethod, ABCMeta

class LIONreconstructor(ABC):
    def __init__(
            self, 
            model, 
            data) -> None:
        
        assert isinstance(model, LIONmodel), "model must be a LIONmodel"
        assert isinstance(data, Experiment) or isinstance(data, Dataset) or torch.is_tensor(data), "data must be Experiment, Dataset or batched input"
        if torch.is_tensor(data):
            assert len(data.shape) == 3 or len(data.shape) == 4, f"input Tensor must have shape (C,W,H) or (C,D,H,W), currently {data.shape}"
        super().__init__()
        
        self.model = model
        self.model.eval()


        self.dataloader = data # Contains parameters of images to be reconstructed.
    
    def __call__(self, *args):
        return self.forward(args)

    @abstractmethod
    def forward(self, *args):
        pass

# Postprocessing eg FBPConvNet: takes sinogram and performs FDK reconstruction
# iterative unrolled: also takes sinogram
# class End2EndReconstructor(LIONreconstructor):
#     def __init__(
#             self, 
#             model, 
#             data) -> None:
#         super().__init__(model, data)
