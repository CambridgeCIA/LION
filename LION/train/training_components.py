"""TODO: Do these belong in training_components.py? They are not only used for training right?"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import torch

from LION.config.ExperimentConfig import ExperimentConfig, OptimConfig

ExperimentT = TypeVar("ExperimentT", bound=ExperimentConfig)
ModelT = TypeVar("ModelT", bound=torch.nn.Module)
OptimConfigT = TypeVar("OptimConfigT", bound=OptimConfig)
ForwardDataT = TypeVar("ForwardDataT")

OptimizerParamGroup = dict[str, object]


@dataclass(kw_only=True)
class ModelInput:
    input_image: torch.Tensor
    initial_image: torch.Tensor | None

    def multiply(self, factor: float) -> Self:
        self.input_image = factor * self.input_image
        if self.initial_image is not None:
            self.initial_image = factor * self.initial_image
        return self

    def to(self, device) -> Self:
        self.input_image = self.input_image.to(device=device)
        if self.initial_image is not None:
            self.initial_image = self.initial_image.to(device=device)
        return self


@dataclass(frozen=True, kw_only=True)
class ForwardResult:
    """Standard result returned by every injected forward runner."""

    reconstruction: torch.Tensor
    extra_output: torch.Tensor | None = None

    @property
    def lambda_maps(self) -> torch.Tensor | None:
        """Named view of the optional SAPM output."""
        return self.extra_output


class ParameterGroupBuilder(ABC, Generic[ModelT, OptimConfigT]):
    """Build optimizer parameter groups."""

    def __call__(
        self, config: OptimConfigT, model: ModelT
    ) -> list[OptimizerParamGroup]:
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        assert len(parameters) > 0, "No trainable parameters found."

        print(f"Model type is {type(model)}.")
        print(f"Found {len(parameters)} trainable parameter tensors.")
        return [
            {
                "params": parameters,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            }
        ]


class ForwardRunner(ABC, Generic[ExperimentT, ModelT, ForwardDataT]):
    """Run one domain-specific model forward pass through a uniform interface."""

    @abstractmethod
    def __call__(
        self,
        model: ModelT,
        data: ForwardDataT,
        config: ExperimentT,
        *,
        skip_grad_checkpoint: bool,
    ) -> ForwardResult:
        ...


make_param_groups_base = ParameterGroupBuilder()
