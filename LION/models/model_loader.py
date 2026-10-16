from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from mrpro.operators import LinearOperator
from torch import nn

from LION.config.ExperimentConfig import CNNConfig, ExperimentConfig, PretrainedConfig
from LION.config.sapm_config import (
    AdaptivePDHGConfig,
    ExplicitMapConfig,
    SAPMModelConfig,
)
from LION.models.ParameterMapNetwork import ExplicitMap, ParameterMapNetwork
from LION.models.unet import UNet
from LION.utils.reproduce import RandomStates
from LION.utils.types import StateDict

ExperimentT = TypeVar("ExperimentT", bound=ExperimentConfig)
ModelT = TypeVar("ModelT", bound=nn.Module)


@dataclass(frozen=True, kw_only=True)
class ModelSetup:
    """Typed model-construction inputs shared by training and evaluation."""

    keys: Sequence[int] | None = None
    map_shape: Sequence[int] | None = None
    forward_op: LinearOperator | None = None
    example_image: torch.Tensor | None = None


def load_random_states(config: PretrainedConfig | None) -> RandomStates:
    rng_state = None
    torch_random_state = None
    np_random_state = None
    random_random_state = None

    if (
        config is not None
        and config.pretrained_id is not None
        and config.pretrained_id != "download"
    ):
        path = config.pretrained_path
        if path is not None and config.is_checkpoint:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            rng_state = checkpoint.get("rng_state")
            torch_random_state = checkpoint.get("torch_random_state")
            np_random_state = checkpoint.get("np_random_state")
            random_random_state = checkpoint.get("random_random_state")

    return RandomStates(
        rng_state=rng_state,
        torch_random_state=torch_random_state,
        np_random_state=np_random_state,
        random_random_state=random_random_state,
    )


def load_from_state_dict(
    model: ModelT, state_dict: StateDict, *, strict: bool = True
) -> ModelT:
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model


def _make_explicit_map_initial_tensor(
    map_shape: Sequence[int], config: ExplicitMapConfig
) -> torch.Tensor:
    tensor_shape = (1, *map_shape)

    if config.initial_values is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        return torch.rand(tensor_shape, generator=generator, dtype=torch.float32)

    if isinstance(config.initial_values, (int, float)):
        return torch.full(
            tensor_shape, float(config.initial_values), dtype=torch.float32
        )

    n_maps = map_shape[0]
    values = torch.tensor(
        [float(value) for value in config.initial_values], dtype=torch.float32
    )
    assert values.numel() == n_maps, f"Expected {n_maps}. Got {values.numel()}."

    broadcast_shape = (1, n_maps, *((1,) * (len(map_shape) - 1)))
    return values.view(broadcast_shape).expand(tensor_shape).clone()


class ModelLoader(ABC, Generic[ExperimentT, ModelT]):
    """Construct a model and optionally restore its optimizer state."""

    @abstractmethod
    def __call__(
        self,
        config: ExperimentT,
        setup: ModelSetup | None = None,
    ) -> tuple[ModelT, StateDict | None]:
        ...

    def load_pretrained(
        self,
        model: ModelT,
        config: PretrainedConfig,
    ) -> tuple[ModelT, StateDict | None]:
        optimizer_state_dict = None
        model.eval()
        path = config.pretrained_path
        if path is None:
            return model, optimizer_state_dict

        print(f"Loading pretrained model `{config.pretrained_id}` from {path} ...")

        if config.is_checkpoint:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model_state_dict"]
            if config.load_optimizer_state:
                optimizer_state_dict = checkpoint["optimizer_state_dict"]
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=False)

        model = load_from_state_dict(model, state_dict)
        return model, optimizer_state_dict

    def make_sapm_parameter_map_network(
        self,
        model_config: SAPMModelConfig,
        *,
        keys: Sequence[int],
        map_shape: Sequence[int] | None,
        expected_map_shape_ndim: int,
    ) -> ParameterMapNetwork:
        pdhg_config = model_config.pdhg
        assert isinstance(pdhg_config, AdaptivePDHGConfig), f"Got {type(pdhg_config)}."

        if model_config.cnn is not None:
            return ParameterMapNetwork(
                self.init_unet(model_config.cnn),
                keys,
                initial_scale_factor_raw=pdhg_config.initial_scale_factor_raw,
                scale_factor_softplus_beta=pdhg_config.scale_factor_softplus_beta,
                lambda_softplus_beta=pdhg_config.lambda_softplus_beta,
            )

        assert map_shape is not None
        assert len(map_shape) == expected_map_shape_ndim
        assert model_config.explicit_map is not None
        lambda_map_init = _make_explicit_map_initial_tensor(
            map_shape=map_shape,
            config=model_config.explicit_map,
        )
        return ParameterMapNetwork(
            map_maker=ExplicitMap(lambda_map_raw_init=lambda_map_init),
            keys=keys,
            initial_scale_factor_raw=pdhg_config.initial_scale_factor_raw,
            scale_factor_softplus_beta=pdhg_config.scale_factor_softplus_beta,
            lambda_softplus_beta=pdhg_config.lambda_softplus_beta,
        )

    def init_unet(self, cnn_config: CNNConfig) -> UNet:
        return UNet(
            dim=cnn_config.dim,
            padding_mode=cnn_config.padding_mode,
            n_filters=cnn_config.init_filters,
            n_enc_stages=cnn_config.n_blocks,
            n_ch_in=cnn_config.in_channels,
            n_ch_out=cnn_config.out_channels,
            n_convs_per_stage=cnn_config.n_convs_per_block,
            kernel_size=cnn_config.kernel_size,
            bias=cnn_config.bias,
            pooling_kernel_size=cnn_config.downsampling_kernel,
        )
