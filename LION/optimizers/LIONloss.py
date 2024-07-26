from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional
import inspect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
from LION.exceptions.exceptions import LossSchemaException
import LION.experiments.ct_experiments as ct_experiments
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.models.CNNs.MS_D import MS_D


class LossRequirement(Enum):
    SINOGRAM = 0
    NOISY_RECON = 1
    PREDICTION = 2
    GROUND_TRUTH = 3


class LIONloss(nn.Module):
    def __init__(self, loss_fn: Callable, schema: list[LossRequirement], model: Optional[LIONmodel]) -> None:
        """_summary_

        Args:
            loss_fn (Callable): _description_
            schema (list[LossRequirement]): The order of schema is the order arguments will be put into loss_fn

        Raises:
            LossSchemaException: _description_
        """
        # make sure loss_fn signature and requirements match
        if (param_count := len(inspect.signature(loss_fn).parameters)) != len(schema):
            raise LossSchemaException(
                f"requirement schema of length {len(schema)} cannot correspond to loss function with {param_count} parameters"
            )
        
        # make sure schema only specifies each type once (this is a LION developer error)
        observed_requirements = []
        for requirement in schema: 
            if requirement in observed_requirements:
                assert len(schema) == len(set(schema)), f"Requirement {requirement} is repeated in schema {schema}"
            else:
                observed_requirements.append(requirement)

        self.loss_fn = loss_fn
        self.schema = schema
        self.model = model

    def forward(self, *args):
        if len(args) != len(self.schema):
            raise TypeError(f"Expected {len(self.schema)} arguments, {len(args)} were provided")
        
        return self.loss_fn(*args)

class LIONloss2(nn.Module, ABC):
    def __init__(self, model: LIONmodel) -> None:
        super().__init__()
        self.model = model
    
    @abstractmethod
    def forward(self, sino: torch.Tensor, gt: torch.Tensor):
        pass

class LIONmse2(LIONloss2):
    def __init__(self, model) -> None:
        super().__init__(model)
        assert self.model is None, "need model"
        self.loss = nn.MSELoss()

    def forward(self, sino, target):
        if self.model.model_parameters.model_input_type == ModelInputType.IMAGE:
            data = fdk(sino, self.model.op)
        else:
            data = sino
        return self.loss(self.model(data), target)
    
class WeirdLoss2(LIONloss2):
    def __init__(self, model) -> None:
        super().__init__(model)
        assert self.model is None, "need model"

    def forward(self, sino, target):
        bad_recon = fdk(sino, self.model.op)
        return self.model(bad_recon) - self.model(target)


class LIONmse(LIONloss):
    def __init__(self):
        super().__init__(loss_fn=nn.MSELoss(), schema=[LossRequirement.PREDICTION, LossRequirement.GROUND_TRUTH], model=None)
    

class WeirdLoss(LIONloss):
    def __init__(self, model):
        super().__init__(loss_fn=self.loss, schema=[LossRequirement.SINOGRAM, LossRequirement.GROUND_TRUTH], model=model)
    
    def loss(self, sino, gt):
        print("doing weird stuff with sino gt and model...")
        return 1
    


if __name__ == "__main__":
    loss_fn = LIONmse()

    experiment = ct_experiments.clinicalCTRecon()
    op = make_operator(experiment.geo)

    dataset = experiment.get_training_dataset()
    dataloader = DataLoader(dataset, 1)

    sino, gt = next(iter(dataloader))
    bad_recon = fdk(sino, op)

    model = MS_D()
    # clean_recon = model(bad_recon)

    # args = []
    # for param in loss_fn.schema:
    #     if param == LossRequirement.GROUND_TRUTH:
    #         args.append(gt)
    #     elif param == LossRequirement.NOISY_RECON:
    #         args.append(bad_recon)
    #     elif param == LossRequirement.PREDICTION:
    #         args.append(clean_recon)
    #     elif param == LossRequirement.SINOGRAM:
    #         args.append(sino)
    # loss = loss_fn(*args)
    # print(loss)

    loss_fn2 = WeirdLoss2(model)

    loss = loss_fn2(sino, gt)
    print(loss)

