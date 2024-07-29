from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
import LION.experiments.ct_experiments as ct_experiments
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.models.post_processing.FBPConvNet import FBPConvNet


class LIONtrainingLoss(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[LIONmodel] = None

    @classmethod
    def from_torch(cls, torch_loss, op=None):
        class TempLoss(LIONtrainingLoss):
            def __init__(self):
                super().__init__()
                self.op = op

            def forward(self, sino: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
                assert (
                    self.model is not None
                ), "Model required but not set. Please call set_model"

                if self.model.model_parameters.model_input_type == ModelInputType.IMAGE:
                    assert (
                        self.op is not None
                    ), "Operator must be provided  to LIONtrainingLoss for model that reconstructs from an image"
                    data = fdk(sino, self.op)

                    if hasattr(self, "do_normalize") and self.do_normalize is not None:
                        assert (
                            hasattr(self, "normalize") and self.normalize is not None
                        ), "do_normalize True but no normalization function not set"
                        data = self.normalize(data)
                else:
                    data = sino

                return torch_loss(self.model(data), gt)

        return TempLoss()

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def forward(self, sino: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("")


def basic_torch_loss_to_lion(torch_loss: nn.Module, op=None) -> LIONtrainingLoss:
    class TempLoss(LIONtrainingLoss):
        def __init__(self) -> None:
            super().__init__()
            self.loss = torch_loss
            self.op = op

        def forward(self, sino: torch.Tensor, gt: torch.Tensor):
            assert (
                self.model is not None
            ), "Model required but not set. Please call set_model"
            if self.model.model_parameters.model_input_type == ModelInputType.IMAGE:
                assert (
                    self.op is not None
                ), "Operator must be provided to LIONtrainingLoss for model that reconstructs from an image"
                data = fdk(sino, self.op)
            else:
                data = sino
            return self.loss(self.model(data), gt)

    return TempLoss()


# demo for reference
if __name__ == "__main__":
    device = torch.device("cuda:1")
    experiment = ct_experiments.clinicalCTRecon()
    op = make_operator(experiment.geo)

    dataset = experiment.get_training_dataset()
    dataloader = DataLoader(dataset, 1)

    sino, gt = next(iter(dataloader))
    sino = sino.to(device)
    gt = gt.to(device)
    bad_recon = fdk(sino, op)

    model = FBPConvNet(geometry_parameters=experiment.geo).to(device)
    model.model_parameters.model_input_type = ModelInputType.SINOGRAM

    loss_fn = LIONtrainingLoss.from_torch(nn.L1Loss())
    loss_fn.set_model(model)
    loss = loss_fn(sino, gt)
    print(loss)
