# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Hong Ye Tan
# Modifications: -
# =============================================================================

# Lionmodels
from LION.models.LIONmodel import LIONmodel
from LION.models.LIONmodelSubclasses import (
    LIONmodelSino,
    LIONmodelRecon,
    forward_decorator,
)
from LION.experiments.ct_experiments import Experiment
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from LION.utils.metrics import SSIM, PSNR
from typing_extensions import Literal
from tqdm import tqdm

# imports related to class organization
from abc import ABC, abstractmethod, ABCMeta


def to_dataset(data):
    # Takes an Experiment, Dataset or data tensor
    # Returns Dataset, containing test if Experiment is passed
    if data is None:
        return None
    assert (
        isinstance(data, Experiment)
        or isinstance(data, Dataset)
        or torch.is_tensor(data)
    ), "data must be Experiment, Dataset or batched input"
    if isinstance(data, Experiment):
        dataset = data.get_testing_dataset()
        has_gt = True
    elif isinstance(data, Dataset):
        dataset = data
        has_gt = True
    elif torch.is_tensor(data):
        if len(data.shape) == 3:
            print(
                "Got data of shape (N,H,W), expanding to shape (N,1,H,W). Please manually expand if this is not desired."
            )
            data = torch.unsqueeze(data, 1)
        elif len(data.shape) == 4:
            pass
        else:
            print(
                f"input Tensor must have shape (C,W,H) or (C,D,H,W), currently {data.shape}"
            )
        # Construct virtual dataloader
        dataset = TensorDataset(data)
    return dataset, has_gt


def reduce_dict(
    dict_input: dict, reduction: Literal["mean", "sum", "none", None]
) -> dict:
    reduced_dict = {}
    for key, val in dict_input.items():
        if reduction == "mean":
            reduced_dict[key] = torch.mean(val)
        if reduction == "none" or reduction is None:
            reduced_dict[key] = val
        if reduction == "sum":
            reduced_dict[key] = torch.sum(val)
    return reduced_dict


class LIONreconstructor(nn.Module):
    def __init__(
        self,
        model,
        data=None,
        has_gt=None,
        metrics={"ssim": SSIM, "psnr": PSNR},
        reduction: Literal["mean", "sum", "none", None] = "mean",
    ) -> None:

        assert isinstance(model, LIONmodel), "model must be a LIONmodel"
        super().__init__()

        self.model = model
        self.model.eval()
        # self.device = self.model.device
        self.dataset = None
        self.has_gt = has_gt  # Flag true if dataset has ground truth available
        self.metrics = metrics  # metrics should take preds and gt, and return tensor of length N, where N is batch size
        self.reduction = reduction

        # Initialize sino2recon and recon2recon where applicable
        # Context from subtyping of LIONmodelRecon and LIONmodelSino
        # LIONmodelSino has priority over LIONmodelRecon for torch forward calls
        # If LIONmodelRecon is passed, sino2recon is achieved using FBP decorator
        if isinstance(self.model, LIONmodelRecon) and not isinstance(
            self.model, LIONmodelSino
        ):
            self.sino2recon = forward_decorator(
                self.model, self.model.forward
            )  # Geometry grabbed from model
            self.recon2recon = self.model.forward
        elif isinstance(self.model, LIONmodelRecon) and isinstance(
            self.model, LIONmodelSino
        ):
            self.sino2recon = self.model.forward
            self.recon2recon = self.model.recon2recon
        elif not isinstance(self.model, LIONmodelRecon) and isinstance(
            self.model, LIONmodelSino
        ):
            self.sino2recon = self.model.forward
            self.recon2recon = None
        else:
            raise TypeError(
                f"Model is not a LIONmodelSino or LIONmodelRecon, model class {model.__class__}"
            )

        if data is not None:
            self.dataset, self.has_gt = to_dataset(data)

        # override has_gt if given
        # used in case dataset or experiment does not have gt available
        if has_gt is not None:
            self.has_gt = has_gt

    def forward(
        self,
        data=None,
        reduction: Literal["mean", "sum", "none", None] = None,
        **kwargs,
    ):
        # default arguments
        batch_size = kwargs.get("batch_size", 1)
        subset_size = kwargs.get("subset_size", None)

        if data is not None:
            assert (
                isinstance(data, Experiment)
                or isinstance(data, Dataset)
                or torch.is_tensor(data)
            ), "data must be Experiment, Dataset or batched input"
            dataset = to_dataset(data)
        elif self.dataset:
            dataset = self.dataset
        else:
            raise RuntimeError(
                "Either pass data when constructing LIONreconstructor, or pass data into the forward operator."
            )

        if reduction is None:
            reduction = self.reduction  # allow override reduction

        # for creating the subset if passed
        if subset_size is not None:
            dataset = Subset(dataset, range(min(subset_size, len(dataset))))

        # if isinstance(data, Experiment):
        #     data_iterator = data.get_testing_dataset() # TODO get iterator
        # elif isinstance(data, Dataset):
        #     data_iterator = None # TODO get iterator
        # elif torch.is_tensor(data):
        #     data_iterator = [data]
        # else:
        #     raise NotImplementedError(f"Reconstruction is not supported for data of type {type(self.data)}")

        # dataset = dataset.to(self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        if self.has_gt:
            # preload to prevent excessive copying
            eval_metrics = dict.fromkeys(self.metrics.keys())
            # default value does not work since torch.zeros will all refer to same object
            for key in eval_metrics.keys():
                eval_metrics[key] = torch.zeros(len(dataset))

            ctr = 0  # track where to replace
            with torch.no_grad():
                for sino, gt in tqdm(dataloader):
                    batch_len = len(
                        gt
                    )  # usually should be batch_size, but may be smaller at the end. this catches edge case
                    # concatenate metrics
                    reconstruction = self.reconstructSino(sino)
                    for key, metric in self.metrics.items():
                        computed_metric = metric(reconstruction, gt)
                        eval_metrics[key][ctr : ctr + len(gt)] = computed_metric

                    ctr += batch_len
            return reduce_dict(eval_metrics, reduction)
        else:  # no gt provided, compute reconstruction.
            print(
                "Ground truth not provided (or flagged False), creating reconstructions"
            )
            with torch.no_grad():
                # recons = torch.Tensor([]).to(self.device)
                for sino in tqdm(dataloader):
                    current_recon = self.reconstructSino(sino)
                    recons = torch.cat(recons, current_recon.cpu())
                return recons

    def reconstructSino(self, sino):
        return self.sino2recon(sino)

    def reconstructRecon(self, recon):
        if self.recon2recon is None:
            warnings.warn(
                f"Model of type {self.model.__class__} does not support recon2recon"
            )
            return recon
        else:
            return self.recon2recon(recon)

    def compute_metrics(self, data=None, metrics=[SSIM, PSNR], **kwargs):
        if data is not None:
            assert isinstance(data, Experiment) or isinstance(
                data, Dataset
            ), "data must be Experiment or Dataset for metrics"
        assert metrics, "Cannot have empty metrics"

        # grab data
        # default arguments
        batch_size = kwargs.get(batch_size, 1)
        shuffle = kwargs.get(batch_size, True)

        dataloader = DataLoader()
        pass
