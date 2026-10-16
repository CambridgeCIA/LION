from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import mlflow
import torch
import yaml
from tqdm import tqdm

from LION.config.ExperimentConfig import ExperimentConfig, OptimConfig
from LION.models.model_loader import ModelLoader, ModelSetup, load_random_states
from LION.train.training_components import ParameterGroupBuilder
from LION.utils.count_parameters import count_parameters
from LION.utils.debug import debug_finite_model, debug_finite_tensor
from LION.utils.log import CompactListDumper, to_yaml_friendly
from LION.utils.reproduce import make_reproducible_base
from LION.utils.types import TrainingState

ExperimentT = TypeVar("ExperimentT", bound=ExperimentConfig)
ModelT = TypeVar("ModelT", bound=torch.nn.Module)


class Trainer(ABC, Generic[ExperimentT]):
    """Domain-independent training base.

    This class owns reproducibility, model/optimizer construction, progress, backpropagation, shared lifecycle hooks.
    """

    def __init__(
        self,
        config: ExperimentT,
        parameter_group_builder: ParameterGroupBuilder,
        model_loader: ModelLoader[ExperimentT, ModelT],
    ) -> None:
        self.config = config
        self.parameter_group_builder = parameter_group_builder
        self.model_loader = model_loader
        self.state = TrainingState()

    def setup(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        ...

    def finalize(self) -> None:
        pass

    def setup_rng(self) -> None:
        random_states = load_random_states(self.config.pretrained)
        self.rng = make_reproducible_base(
            self.config.reproduce, random_states=random_states
        )

    def setup_model_and_optimizer(self, *, model_setup: ModelSetup) -> None:
        model, optimizer_state_dict = self.model_loader(self.config, setup=model_setup)
        print(f"Model: {type(model)}")
        _, trainable_params, _ = count_parameters(model)
        print(f"Number of trainable parameters: {trainable_params}")
        self.model = model.to(device=self.config.train.device)
        self.optimizer = init_optimizer(
            self.config.optim, self.model, self.parameter_group_builder
        )

        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

    def _setup(self, *, epoch_size: int, model_setup: ModelSetup) -> None:
        self.setup_rng()
        self.setup_model_and_optimizer(model_setup=model_setup)

        config = self.config
        if config.pretrained.pretrained_epoch is not None:
            prev_epoch = config.pretrained.pretrained_epoch
            if config.pretrained.last_global_step is not None:
                prev_global_step = config.pretrained.last_global_step
            else:
                # NOTE: Assume using 1-indexing
                prev_global_step = config.pretrained.pretrained_epoch * epoch_size
        else:
            if config.pretrained.last_global_step is not None:
                raise ValueError(
                    "pretrained.last_global_step requires pretrained.pretrained_epoch."
                )
            # NOTE: Assume using 1-indexing
            prev_epoch, prev_global_step = 0, 0

        self.state.one_indexed_epoch = prev_epoch
        self.state.one_indexed_global_step = prev_global_step

    def optimizer_step_from_loss(self, loss: torch.Tensor) -> float:
        debug_finite_tensor("loss before backward", loss)
        debug_finite_model(self.model, "before backward")

        loss.backward()
        debug_finite_model(self.model, "after backward before step")

        self.optimizer.step()
        debug_finite_model(self.model, "after optimizer step")

        return float(loss.item())

    def train_epoch_iterator(self, desc: str):
        # Assume we're still in the previous epoch.
        # E.g., if training from scratch then the current 1-indexed epoch is 0.
        num_train_epochs_left = self.config.train.num_epochs - self.one_indexed_epoch
        return tqdm(range(num_train_epochs_left), desc=desc, dynamic_ncols=True)

    def increment_epoch(self) -> None:
        self.state.one_indexed_epoch += 1

    def increment_global_step(self) -> None:
        self.state.one_indexed_global_step += 1

    @property
    def one_indexed_epoch(self) -> int:
        return self.state.one_indexed_epoch

    @property
    def one_indexed_step(self) -> int:
        return self.state.one_indexed_global_step


def run_trainer_with_mlflow(trainer: Trainer, run_name: str) -> None:
    trainer.setup()
    with mlflow.start_run(run_name=run_name) as run:
        assert (
            trainer.config.dirs.run_dir is not None
        ), "dirs.run_dir must be specified in the config."
        trainer.config.dirs.mlflow_run_id_path.write_text(f"{run.info.run_id}\n")
        trainer.train()
        trainer.finalize()


def init_optimizer(
    config: OptimConfig,
    model: torch.nn.Module,
    parameter_group_builder: ParameterGroupBuilder,
) -> torch.optim.Optimizer:
    param_groups = parameter_group_builder(config, model)
    if config.name == "Adam":
        return torch.optim.Adam(param_groups)
    if config.name == "AdamW":
        return torch.optim.AdamW(param_groups)
    raise ValueError(f"Unsupported optimizer: {config.name}")


def save_typed_config(config: ExperimentConfig) -> None:
    if config.dirs.run_dir is None:
        raise ValueError("config.dirs.run_dir must be specified.")
    config.dirs.run_dir.mkdir(parents=True, exist_ok=True)
    with config.dirs.requested_config_path.open("w") as f:
        yaml.dump(
            to_yaml_friendly(dataclasses.asdict(config)),
            f,
            Dumper=CompactListDumper,
            sort_keys=False,
        )
