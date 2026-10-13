from __future__ import annotations

import argparse
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, TypeVar

import torch
import tyro

from LION.config.DataConfig import DataConfig
from LION.config.LogConfig import LogConfig
from LION.config.RuntimeDirs import RuntimeDirs

OptimName = Literal["Adam", "AdamW"]
PaddingMode = Literal["zeros", "reflect", "replicate", "circular"]
ImageType = Literal[
    "zeros",
    "adjoint",
    "zero_filled",
    "cg",
    "spgl1",
    "noisy",
    "iterative_sense",
    "input_image",
    "neighbour_average",
]


@dataclass(kw_only=True)
class ReproduceConfig:
    """Reproducibility configuration."""

    base_seed: int = 2026
    random_seed: int = 0
    torch_seed: int = 0
    numpy_seed: int = 0
    rng_seed: int = 0


@dataclass(kw_only=True)
class CNNConfig:
    """Configuration for the CNN block."""

    uses_simple_cnn: bool
    uses_new_unet: bool

    init_filters: int
    n_blocks: int

    padding_mode: PaddingMode = "circular"
    bias: bool = False

    dim: int
    in_channels: int
    out_channels: int

    kernel_size: int = 3

    activation: str = "LeakyReLU"

    # For U-Net
    n_convs_per_block: int | None = None  # Usually 2
    downsampling_kernel: int | None = None  # Usually 2
    downsampling_mode: str | None = None  # Usually max_pool or average
    upsampling_kernel: int | None = None  # Usually 2
    upsampling_mode: str | None = None  # Usually linear_interpolation or conv_transpose


@dataclass(kw_only=True)
class OptimConfig:
    """Configuration for optimizer construction."""

    name: OptimName
    learning_rate: float
    weight_decay: float


@dataclass(kw_only=True)
class TrainConfig:
    """Configuration for common training."""

    num_epochs: int = 1000
    batch_size: int = 1
    device: Literal["auto", "cpu", "cuda"] = "auto"


@dataclass(kw_only=True)
class PretrainedConfig:
    """Configuration for loading pretrained model states."""

    pretrained_id: str | None = None
    pretrained_path: Path | None = None
    pretrained_epoch: int | None = None
    last_global_step: int | None = None
    is_checkpoint: bool = False
    load_optimizer_state: bool = False

    def __post_init__(self) -> None:
        """Normalise path-like fields."""
        if self.pretrained_path is not None:
            object.__setattr__(self, "pretrained_path", Path(self.pretrained_path))


@dataclass(kw_only=True)
class ConjugateGradientConfig:
    """Configuration for the conjugate gradient solver."""

    max_iters: int
    initial_value: str
    tolerance: float
    implicit_backward: bool


@dataclass(kw_only=True)
class ModelConfig:
    """Configuration for a generic model."""

    name: str
    input_image: ImageType


@dataclass(kw_only=True)
class UnrolledModelConfig(ModelConfig):
    """Configuration for unrolled optimization."""

    n_iters: int
    initial_image: ImageType | None

    use_checkpoint: bool = False
    checkpoint_chunk_size: int | None = None
    checkpoint_use_reentrant: bool = False


@dataclass(kw_only=True)
class ExperimentConfig:
    """Configuration for experiments."""

    reproduce: ReproduceConfig
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    log: LogConfig
    pretrained: PretrainedConfig = field(default_factory=PretrainedConfig)
    dirs: RuntimeDirs = field(default_factory=RuntimeDirs)
    render_only: bool = False
    render_saved_results: bool = True
    gen_lambda_maps: bool = False
    lambda_map_checkpoint_stems: list[str] = field(default_factory=list)
    """Additional checkpoint stems whose lambda-map artifacts should be generated in render-only mode.

    For example, ``['current']`` loads ``checkpoints/current.pt``. If the
    checkpoint records epoch 90, its lambda artifacts are written under
    ``recons/epoch_0090`` without requiring or creating a reconstruction.
    """


ExperimentT = TypeVar("ExperimentT", bound=ExperimentConfig)


class ExperimentConfigLoader(ABC, Generic[ExperimentT]):
    """Load one typed experiment configuration from a YAML path."""

    @abstractmethod
    def __call__(self, path: Path) -> ExperimentT:
        ...


def resolve_device(device: str) -> str:
    """Resolve the runtime device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return device


def get_permutation_typed(
    config: DataConfig, dirs: RuntimeDirs, dataset_length: int | None = None
) -> list[int]:
    """Load or generate the supervised train/validation/test permutation."""
    permutation_file = config.permutation_file

    if permutation_file is not None:
        print(f"Loading permutation from {permutation_file}")  # noqa: T201

        permutation = []
        with Path(permutation_file).open("r") as f:
            for line in f:
                permutation.append(int(line.strip()))

        return permutation

    if config.permutation_seed is None:
        generator = torch.Generator()
    else:
        generator = torch.Generator().manual_seed(config.permutation_seed)

    assert (
        dataset_length is not None
    ), "dataset_length must be provided when generating a new permutation"
    generated_permutation = torch.randperm(dataset_length, generator=generator)

    if dirs.permutation_file is not None:
        with dirs.permutation_file.open("w") as f:
            for idx in generated_permutation:
                f.write(f"{idx.item()}\n")

    return generated_permutation.tolist()


def parse_config_path() -> tuple[Path | None, list[str]]:
    """Parse only the YAML config path before tyro handles the rest."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    args, remaining_args = parser.parse_known_args()
    return args.config, remaining_args


def load_experiment_config_cli(
    config_cls: type[ExperimentT], load_config: ExperimentConfigLoader[ExperimentT]
) -> ExperimentT:
    """Load a YAML experiment config, apply CLI overrides, and resolve the device."""
    config_path, remaining_args = parse_config_path()

    help_requested = "--help" in remaining_args or "-h" in remaining_args

    if config_path is None:
        if help_requested:
            print(
                "No --config was provided, so default values from a YAML file are not shown.\n"
                "Showing available config override options from the config class.\n"
                "\n"
                "For YAML-specific default values, run:\n"
                "  python script.py --config path/to/config.yaml --help\n"
            )
            tyro.cli(config_cls, args=remaining_args)
            exit(0)

        raise ValueError("Missing required argument: --config")

    config = tyro.cli(config_cls, default=load_config(config_path), args=remaining_args)
    device = resolve_device(config.train.device)
    return dataclasses.replace(
        config, train=dataclasses.replace(config.train, device=device)
    )
