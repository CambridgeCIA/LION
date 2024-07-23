from enum import Enum
from typing import Callable
import inspect
import torch.nn as nn
from torch.utils.data import DataLoader

from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
from LION.exceptions.exceptions import LossSchemaException
import LION.experiments.ct_experiments as ct_experiments
from LION.models.CNNs.MS_D import MS_D


class LossRequirement(Enum):
    SINOGRAM = 0
    NOISY_RECON = 1
    PREDICTION = 2
    GROUND_TRUTH = 3


class LIONloss:
    def __init__(self, loss_fn: Callable, schema: list[LossRequirement]) -> None:
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

    def __call__(self, *args):
        if len(args) != len(self.schema):
            raise TypeError(f"Expected {len(self.schema)} arguments, {len(args)} were provided")
        
        return self.loss_fn(*args)


class LIONmse(LIONloss):
    def __init__(self):
        super().__init__(loss_fn=nn.MSELoss(), schema=[LossRequirement.PREDICTION, LossRequirement.GROUND_TRUTH])

if __name__ == "__main__":
    loss_fn = LIONmse()

    experiment = ct_experiments.clinicalCTRecon()
    op = make_operator(experiment.geo)

    dataset = experiment.get_training_dataset()
    dataloader = DataLoader(dataset, 1)

    sino, gt = next(iter(dataloader))
    bad_recon = fdk(sino, op)

    model = MS_D()
    clean_recon = model(bad_recon)

    args = []
    for param in loss_fn.schema:
        if param == LossRequirement.GROUND_TRUTH:
            args.append(gt)
        elif param == LossRequirement.NOISY_RECON:
            args.append(bad_recon)
        elif param == LossRequirement.PREDICTION:
            args.append(clean_recon)
        elif param == LossRequirement.SINOGRAM:
            args.append(sino)
    loss = loss_fn(*args)
    print(loss)

