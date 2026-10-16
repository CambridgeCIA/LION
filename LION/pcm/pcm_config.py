from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from LION.classical_algorithms.spgl1_torch import SPGL1Parameter
from LION.config.config_utils import load_yaml, make_dataclass
from LION.config.ExperimentConfig import (
    ExperimentConfig,
    ExperimentConfigLoader,
    PretrainedConfig,
    ReproduceConfig,
    RuntimeDirs,
    TrainConfig,
)
from LION.config.LogConfig import make_log_config
from LION.config.sapm_config import (
    SAPMModelConfig,
    SAPMOptimConfig,
    make_sapm_model_config,
)
from LION.data_loaders.pcm.data_primitives import (
    PCMDataCategory,
    PCMDataParameter,
    PCMSamplingParameter,
    SamplesParameter,
)
from LION.pcm.noise import PCMNoiseSimulation, load_pcm_noise_simulation
from LION.train.self_supervised_train import SelfSupervisedCriterion
from LION.utils.parameter import LIONParameter


@dataclass(kw_only=True)
class SupervisedPCMDataConfig(PCMDataParameter):
    """PCM data configuration for supervised training."""

    train_perm_range: list[int]
    val_perm_range: list[int]
    test_perm_range: list[int]


@dataclass(kw_only=True)
class ZeroShotPCMDataConfig(PCMDataParameter):
    """PCM data configuration for single-sample zero-shot training."""

    p_target_train: float
    p_target_val: float | None
    base_train_partition_seed: int
    base_val_partition_seed: int

    this_sample_id: int = -1
    this_category: PCMDataCategory | None = None


@dataclass(kw_only=True)
class PCMSSDUConfig(LIONParameter):
    """SSDU measurement-splitting configuration."""

    p_target_train: float
    p_target_val: float


@dataclass(kw_only=True)
class TrainedPCMMethodParameter(LIONParameter):
    """Configuration shared by trained PCM reconstruction methods."""

    run_dir: Path
    checkpoint: str
    scale_factor_case: tuple[float, int] | None = None

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)


@dataclass(kw_only=True)
class SupervisedPCMMethodParameter(TrainedPCMMethodParameter):
    """Configuration for a trained supervised PCM reconstruction method."""


@dataclass(kw_only=True)
class SelfSupervisedPCMMethodParameter(TrainedPCMMethodParameter):
    """Configuration for a trained dataset self-supervised PCM reconstruction method."""


@dataclass(kw_only=True)
class ZeroShotPCMMethodParameter(LIONParameter):
    """Configuration for a zero-shot PCM reconstruction method."""

    config_path: Path
    criterion: SelfSupervisedCriterion = "best_val_ssdu_loss"

    def __post_init__(self) -> None:
        self.config_path = Path(self.config_path)


@dataclass(kw_only=True)
class SAPMSupervisedPCMConfig(ExperimentConfig):
    """Full configuration for supervised PCM SAPM training."""

    data: SupervisedPCMDataConfig  # type: ignore[assignment]
    model: SAPMModelConfig  # type: ignore[assignment]
    optim: SAPMOptimConfig  # type: ignore[assignment]
    sampling: PCMSamplingParameter = field(default_factory=PCMSamplingParameter)
    spgl1: SPGL1Parameter | None = None
    noise: PCMNoiseSimulation | None = None
    uses_normalized_op: bool = True


@dataclass(kw_only=True)
class SAPMSelfSupervisedPCMConfig(SAPMSupervisedPCMConfig):
    """Full configuration for dataset self-supervised PCM SAPM training."""

    ssdu: PCMSSDUConfig


@dataclass(kw_only=True)
class SAPMZeroShotPCMConfig(ExperimentConfig):
    """Full configuration for zero-shot PCM SAPM training."""

    data: ZeroShotPCMDataConfig  # type: ignore[assignment]
    model: SAPMModelConfig  # type: ignore[assignment]
    optim: SAPMOptimConfig  # type: ignore[assignment]
    sampling: PCMSamplingParameter = field(default_factory=PCMSamplingParameter)
    spgl1: SPGL1Parameter | None = None
    noise: PCMNoiseSimulation | None = None
    uses_normalized_op: bool = True


SAPMPCMConfig = (
    SAPMSupervisedPCMConfig | SAPMSelfSupervisedPCMConfig | SAPMZeroShotPCMConfig
)


def _load_pcm_sapm_common(path: Path) -> tuple[dict, dict, SamplesParameter, dict]:
    """Load fields shared by supervised, self-supervised, and zero-shot PCM SAPM configs."""
    data = load_yaml(path)

    data_dict = dict(data.get("data", {}))
    samples = make_dataclass(SamplesParameter, data_dict.pop("samples", {}))
    j_order = data_dict.get("j_order")
    assert j_order is not None

    common = dict(
        reproduce=make_dataclass(ReproduceConfig, data.get("reproduce", {})),
        model=make_sapm_model_config(data.get("model", {})),
        optim=make_dataclass(SAPMOptimConfig, data.get("optim", {})),
        train=make_dataclass(TrainConfig, data.get("train", {})),
        log=make_log_config(data.get("log", {})),
        pretrained=make_dataclass(PretrainedConfig, data.get("pretrained", {})),
        dirs=make_dataclass(RuntimeDirs, data.get("dirs", {})),
        sampling=make_dataclass(PCMSamplingParameter, data.get("sampling", {})),
        spgl1=make_dataclass(SPGL1Parameter, data.get("spgl1")),
        noise=load_pcm_noise_simulation(data.get("noise"), j_order=j_order),
        uses_normalized_op=bool(data.get("uses_normalized_op", True)),
        render_only=bool(data.get("render_only", False)),
        render_saved_results=bool(data.get("render_saved_results", True)),
        gen_lambda_maps=bool(data.get("gen_lambda_maps", False)),
        lambda_map_checkpoint_stems=list(data.get("lambda_map_checkpoint_stems") or []),
    )
    return data, data_dict, samples, common


def load_pcm_sapm_supervised_config_from_path(path: Path) -> SAPMSupervisedPCMConfig:
    """Load a supervised PCM TV/TGV SAPM config from YAML."""
    _, data_dict, samples, common = _load_pcm_sapm_common(path)
    return SAPMSupervisedPCMConfig(
        data=SupervisedPCMDataConfig(**data_dict, samples=samples), **common
    )


class SAPMSupervisedPCMConfigLoader(ExperimentConfigLoader[SAPMSupervisedPCMConfig]):
    def __call__(self, path: Path) -> SAPMSupervisedPCMConfig:
        return cast(
            SAPMSupervisedPCMConfig, load_pcm_sapm_supervised_config_from_path(path)
        )


load_pcm_sapm_supervised_config = SAPMSupervisedPCMConfigLoader()


def load_pcm_sapm_self_supervised_config_from_path(
    path: Path,
) -> SAPMSelfSupervisedPCMConfig:
    """Load a dataset self-supervised PCM TV/TGV SAPM config from YAML."""
    data, data_dict, samples, common = _load_pcm_sapm_common(path)
    ssdu = make_dataclass(PCMSSDUConfig, data.get("ssdu", {}))
    return SAPMSelfSupervisedPCMConfig(
        data=SupervisedPCMDataConfig(**data_dict, samples=samples), ssdu=ssdu, **common
    )


class SAPMSelfSupervisedPCMConfigLoader(
    ExperimentConfigLoader[SAPMSelfSupervisedPCMConfig]
):
    def __call__(self, path: Path) -> SAPMSelfSupervisedPCMConfig:
        return cast(
            SAPMSelfSupervisedPCMConfig,
            load_pcm_sapm_self_supervised_config_from_path(path),
        )


load_pcm_sapm_self_supervised_config = SAPMSelfSupervisedPCMConfigLoader()


def load_pcm_sapm_zero_shot_config_from_path(path: Path) -> SAPMZeroShotPCMConfig:
    """Load a zero-shot PCM TV/TGV SAPM config from YAML."""
    _, data_dict, samples, common = _load_pcm_sapm_common(path)
    return SAPMZeroShotPCMConfig(
        data=ZeroShotPCMDataConfig(**data_dict, samples=samples), **common
    )


class SAPMZeroShotPCMConfigLoader(ExperimentConfigLoader[SAPMZeroShotPCMConfig]):
    def __call__(self, path: Path) -> SAPMZeroShotPCMConfig:
        return cast(
            SAPMZeroShotPCMConfig, load_pcm_sapm_zero_shot_config_from_path(path)
        )


load_pcm_sapm_zero_shot_config = SAPMZeroShotPCMConfigLoader()
