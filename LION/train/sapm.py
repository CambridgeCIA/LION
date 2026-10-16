from __future__ import annotations

import torch

from LION.config.sapm_config import SAPMOptimConfig
from LION.train.training_components import OptimizerParamGroup, ParameterGroupBuilder


class SAPMParameterGroupBuilder(
    ParameterGroupBuilder[torch.nn.Module, SAPMOptimConfig]
):
    """Build non-overlapping optimizer groups for SAPM models."""

    def __call__(
        self, config: SAPMOptimConfig, model: torch.nn.Module
    ) -> list[OptimizerParamGroup]:
        base_params: list[torch.nn.Parameter] = []
        scale_params: list[torch.nn.Parameter] = []

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            leaf_name = name.split(".")[-1]
            if "scale_factors_raw" in name or leaf_name in {"t_1_v", "t_0_w"}:
                scale_params.append(parameter)
            else:
                base_params.append(parameter)

        print(f"Model type is {type(model)}.")
        print(
            f"Found {len(base_params)} base and {len(scale_params)} scale parameter tensors."
        )

        param_groups: list[OptimizerParamGroup] = []

        def add_group(
            params: list[torch.nn.Parameter],
            *,
            learning_rate: float | None,
            weight_decay: float | None,
        ) -> None:
            if not params or learning_rate is None:
                return
            param_groups.append(
                {"params": params, "lr": learning_rate, "weight_decay": weight_decay}
            )

        add_group(
            base_params,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        add_group(
            scale_params,
            learning_rate=config.scale_factors_raw_learning_rate,
            weight_decay=config.scale_factors_raw_weight_decay,
        )

        if not param_groups:
            raise ValueError("No trainable parameters found for optimizer.")
        return param_groups


make_param_groups_sapm_common = SAPMParameterGroupBuilder()
