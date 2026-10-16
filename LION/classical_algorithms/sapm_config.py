"""Configuration schema for spatially adaptive parameter-map (SAPM) models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from zsu.config.config_utils import make_dataclass
from zsu.config.DataConfig import DataConfig
from zsu.config.ExperimentConfig import (
    CNNConfig,
    ConjugateGradientConfig,
    ExperimentConfig,
    OptimConfig,
    PretrainedConfig,
    ReproduceConfig,
    TrainConfig,
    UnrolledModelConfig,
)
from zsu.config.LogConfig import make_log_config
from zsu.config.RuntimeDirs import RuntimeDirs
from zsu.operators.FiniteDifferenceOp import PadModeIn

SAPMBackend = Literal["new", "old"]
SAPMRegularisation = Literal["TV", "TGV"]


@dataclass(kw_only=True)
class AdaptivePDHGConfig:
    """PDHG configuration for adaptive TV/TGV models."""

    regularisation: SAPMRegularisation
    primal_stepsize: float | None
    dual_stepsize: float | None

    initial_scale_factor_raw: float
    scale_factor_softplus_beta: float
    lambda_softplus_beta: float

    fd_pad_mode: PadModeIn | Sequence[PadModeIn]
    fd_dim: Sequence[int]


@dataclass(kw_only=True)
class OldAdaptivePDHGConfig:
    """PDHG configuration for the older TvPdhgNet / TgvPdhgNet backend."""

    regularisation: SAPMRegularisation
    primal_stepsize: float
    dual_stepsize: float
    theta: float | None

    lambda_1_v_beta: float
    lambda_0_w_beta: float | None
    uses_softplus_positive: bool

    lambda_1_v_init_bias: float | None
    lambda_0_w_init_bias: float | None
    lambda_1_v_log_scale: float | None
    lambda_0_w_log_scale: float | None

    scale_factor_1_v: float
    scale_factor_0_w: float | None
    scale_factor_1_v_beta: float
    scale_factor_0_w_beta: float | None
    learns_scale_factor: bool

    uses_scalar_lambda0: bool = False
    uses_scalar_lambda1: bool = False


@dataclass(kw_only=True)
class ExplicitMapConfig:
    """Initialisation settings for a directly learned raw lambda map.

    ``initial_values`` specifies the raw values assigned to
    ``lambda_map_raw``.

    A scalar fills every map entry with the same raw value.

    A list supplies one raw value for each map channel. Each value is
    broadcast across the spatial and, for MRI, temporal dimensions.

    If ``initial_values`` is None, the raw map is initialised from a
    uniform distribution on ``[0, 1)`` using ``seed``.
    """

    initial_values: float | list[float] | None = None
    seed: int = 42


@dataclass(kw_only=True)
class SAPMModelConfig(UnrolledModelConfig):
    """Shared unrolled adaptive TV/TGV model configuration.

    ``cg`` is optional because MRI currently uses a CG reconstruction as a
    configurable model input, while denoising does not require CG. ``backend``
    defaults to the current MRpro-based implementation so existing MRI YAML
    files do not need a new field.
    """

    pdhg: AdaptivePDHGConfig | OldAdaptivePDHGConfig
    cnn: CNNConfig | None
    cg: ConjugateGradientConfig | None = None
    backend: SAPMBackend = "new"
    explicit_map: ExplicitMapConfig | None = None


@dataclass(kw_only=True)
class SAPMOptimConfig(OptimConfig):
    """Shared optimizer configuration for adaptive TV/TGV models."""

    scale_factors_raw_learning_rate: float | None = None
    scale_factors_raw_weight_decay: float | None = None


def make_sapm_model_config(model_data: dict) -> SAPMModelConfig:
    """Build the SAPM model config."""
    return SAPMModelConfig(
        **{
            **model_data,
            "backend": model_data.get("backend", "new"),
            "cg": make_dataclass(ConjugateGradientConfig, model_data.get("cg")),
            "cnn": make_dataclass(CNNConfig, model_data.get("cnn")),
            "explicit_map": make_dataclass(
                ExplicitMapConfig, model_data.get("explicit_map")
            ),
            "pdhg": make_dataclass(AdaptivePDHGConfig, model_data.get("pdhg", {})),
        }
    )


def load_sapm_config(
    data: dict[str, Any],
    experiment_config_cls: type[ExperimentConfig],
    data_config_cls: type[DataConfig],
) -> ExperimentConfig:
    return experiment_config_cls(
        reproduce=make_dataclass(ReproduceConfig, data.get("reproduce", {})),
        data=make_dataclass(data_config_cls, data.get("data", {})),
        model=make_sapm_model_config(data.get("model", {})),
        optim=make_dataclass(SAPMOptimConfig, data.get("optim", {})),
        train=make_dataclass(TrainConfig, data.get("train", {})),
        log=make_log_config(data.get("log", {})),
        pretrained=make_dataclass(PretrainedConfig, data.get("pretrained", {})),
        dirs=make_dataclass(RuntimeDirs, data.get("dirs", {})),
        render_only=bool(data.get("render_only", False)),
        render_saved_results=bool(data.get("render_saved_results", True)),
        gen_lambda_maps=bool(data.get("gen_lambda_maps", False)),
        lambda_map_checkpoint_stems=list(data.get("lambda_map_checkpoint_stems") or []),
    )
