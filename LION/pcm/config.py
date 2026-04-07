"""Configuration dataclasses and preset definitions for the PCM experiment."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import tyro
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from LION.utils.plot_helper import PlotHelper

DataType = Literal["image", "hadamard_measurement_vector", "original_measurement_data"]
RandomisingScheme = Literal["uniform", "multilevel"]
SupportedDenoiser = Literal["drunet", "gs_drunet"]


@dataclass(frozen=True)
class DataConfig:
    """Input data configuration.

    Attributes
    ----------
    data_dir: Path
        Directory containing the ``.npy`` PCM data files.
    data_name: str
        Stem of the input file.
    data_type: DataType
        Kind of raw input stored in the file.
    j_order: int
        Walsh-Hadamard order ``J`` such that the image size is ``2**J``.
    inverse_sign: bool
        Whether to multiply the loaded data by ``-1``.
    tests_scale_ground_truth: bool
        Whether to min-max normalise the reconstructed ground truth image.
    is_out_of_distribution: bool
        Whether the denoiser input/output should be affine-rescaled.
    r_high: float | None
        Upper end of the expected range for out-of-distribution rescaling.
    r_low: float | None
        Lower end of the expected range for out-of-distribution rescaling.
    scale_eps: float
        Small constant for safe range scaling.
    """

    data_dir: Path
    data_name: str
    data_type: DataType
    j_order: int
    inverse_sign: bool
    tests_scale_ground_truth: bool = False
    is_out_of_distribution: bool = False
    r_high: float | None = None
    r_low: float | None = None
    scale_eps: float = 1e-12

    @property
    def data_filename(self) -> str:
        """Return the filename associated with the configured data stem."""
        return f"{self.data_name}.npy"


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime and output configuration.

    Attributes
    ----------
    root_output_dir: Path
        Path to the directory where experiment outputs will be written.
        A subdirectory will be created for each trial.
    device: Literal["cpu", "cuda", "mps", "auto"]
        Device to use for computation.
    noise_seed: int
        Seed for the random number generator used in noise generation.
    noise_std: float
        Standard deviation of the noise added to the data.
    num_trials: int
        Number of trials to run. Trials are indexed from 0 to num_trials-1.
    num_trials_skip: int
        Number of first trials to skip.
        E.g. set to num_trials_skip to 1 to skip trial 0 and start from trial 1.
        Since each trial generates a random seed deterministically based on its index,
        this is useful for resuming an experiment after a crash or interruption.
    """

    root_output_dir: Path
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    noise_seed: int = 42
    noise_std: float = 0.0
    num_trials: int = 1
    num_trials_skip: int = 0


@dataclass(frozen=True)
class PnPConfig:
    """Configuration for PnP-ADMM reconstruction.

    Attributes
    ----------
    enabled : bool
        Whether to run the PnP-ADMM reconstruction.
    """

    enabled: bool = True
    denoiser_name: SupportedDenoiser = "gs_drunet"
    iters: int = 1
    eta: float = 0.01
    cg_iters: int = 20
    cg_eps: float = 1e-20
    cg_rel_tol: float = 0.0
    drunet_sigma: float = 0.05
    eta_candidates: tuple[float, ...] = field(
        default_factory=lambda: (
            1e-5,
            5e-5,
            1e-4,
            1e-3,
            5e-3,
            1e-2,
            2e-2,
            3e-2,
            4e-2,
            5e-2,
            1e-1,
            1.0,
            10.0,
            20.0,
            50.0,
            100.0,
        )
    )
    iter_candidates: tuple[int, ...] = field(
        default_factory=lambda: (1, 20, 50, 100, 150)
    )
    sigma_candidates: tuple[float, ...] = field(
        default_factory=lambda: (0.01, 0.02, 0.05, 0.1)
    )
    notes: dict[str, str] = field(
        default_factory=lambda: {
            "eta_1e-5": "Undersampling artefacts may remain if eta is too small.",
            "eta_1e-2": "Generally good in the original exploratory script.",
            "eta_100": "Observed NaN for 100 percent sampling in one exploratory note.",
            "cg_eps": "The original script noted little change from the default because CG often terminates quickly.",
        }
    )


@dataclass(frozen=True)
class SPGL1Config:
    """Configuration for SPGL1 reconstruction.

    Attributes
    ----------
    enabled : bool
        Whether to run the SPGL1 reconstruction.
    """

    enabled: bool = True
    factor: float = 1.0
    max_iter: int = 100
    debias_max_iter: int = 10
    debias_support_tol: float = 1e-5
    debias_tol: float = 1e-7
    wavelet_name: str = "db4"
    max_iter_candidates: tuple[int, ...] = field(
        default_factory=lambda: (100, 200, 1000)
    )
    debias_max_iter_candidates: tuple[int, ...] = field(
        default_factory=lambda: (10, 100)
    )
    debias_support_tol_candidates: tuple[float, ...] = field(
        default_factory=lambda: (1e-5, 1e-6, 1e-7)
    )


# TODO: Put physics config in `Experiment` class
@dataclass(frozen=True)
class SamplingConfig:
    """Sampling and test-case configuration."""

    randomising_scheme: RandomisingScheme = "uniform"
    sampling_ratios: tuple[float, ...] = (0.2,)
    coarse_j_values: tuple[int, ...] | None = None
    coarse_j_offset_from_j_order: int | None = 2
    reverse_test_cases: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration for one PCM experiment run."""

    @dataclass(frozen=True)
    class Data:
        """Input data configuration."""

        data_dir: Path
        data_name: str
        data_type: DataType
        j_order: int
        inverse_sign: bool
        tests_scale_ground_truth: bool = False
        is_out_of_distribution: bool = False
        r_high: float | None = None
        r_low: float | None = None
        scale_eps: float = 1e-12

        @property
        def data_filename(self) -> str:
            """Return the filename associated with the configured data stem."""
            return f"{self.data_name}.npy"

    name: str
    runtime: RuntimeConfig
    data: Data
    plot: PlotHelper
    pnp: PnPConfig = field(default_factory=PnPConfig)
    spgl1: SPGL1Config = field(default_factory=SPGL1Config)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


# Backward-compatible alias for external modules importing DataConfig.
DataConfig = ExperimentConfig.Data


DEFAULT_DATA_DIR = Path("../pdo/data/photocurrent_data")
DEFAULT_OUTPUT_DIR = Path("../pdo")


_CURRENT_PRESET = ExperimentConfig(
    name="si_2_256_512x512_image",
    runtime=RuntimeConfig(
        root_output_dir=DEFAULT_OUTPUT_DIR, noise_seed=42, num_trials=1
    ),
    data=ExperimentConfig.Data(
        data_dir=DEFAULT_DATA_DIR,
        data_name="Si_2_256_512x512",
        data_type="image",
        j_order=9,
        inverse_sign=True,
        tests_scale_ground_truth=False,
        is_out_of_distribution=True,
        r_high=2e-5,
        r_low=-2e-6,
        scale_eps=1e-12,
    ),
    plot=PlotHelper(
        zoom=2.5,
        loc="lower right",
        loc1=2,
        loc2=1,
        roi=(322, 85, 100, 100),
        clim=(0.0, 1.5e-5),
        cmap_max=0.8,
        adds_insets=True,
        show_rect=True,
    ),
    pnp=PnPConfig(
        enabled=True,
        denoiser_name="gs_drunet",
        iters=1,
        eta=0.01,
        cg_iters=20,
        cg_eps=1e-20,
        cg_rel_tol=0.0,
        drunet_sigma=0.05,
    ),
    spgl1=SPGL1Config(
        enabled=True,
        factor=1e5,
        max_iter=100,
        debias_max_iter=10,
        debias_support_tol=1e-5,
        debias_tol=1e-7,
        wavelet_name="db4",
    ),
    sampling=SamplingConfig(
        randomising_scheme="uniform",
        sampling_ratios=(0.2,),
        coarse_j_values=None,
        coarse_j_offset_from_j_order=2,
        reverse_test_cases=True,
    ),
)


PRESETS: dict[str, ExperimentConfig] = {
    "si_2_256_512x512_image": _CURRENT_PRESET,
    "si_256_512x512_image": replace(
        _CURRENT_PRESET,
        name="si_256_512x512_image",
        data=replace(
            _CURRENT_PRESET.data,
            data_name="Si_256_512x512",
            j_order=9,
            r_high=1e-4,
            r_low=-5e-6,
        ),
        plot=replace(
            _CURRENT_PRESET.plot,
            roi=(160, 60, 120, 120),
            loc="lower left",
            clim=(0.0, 3e-5),
        ),
    ),
    "si_256_measurement_data": replace(
        _CURRENT_PRESET,
        name="si_256_measurement_data",
        data=replace(
            _CURRENT_PRESET.data,
            data_name="Si_256_measurement_data",
            data_type="original_measurement_data",
            j_order=8,
            r_high=1e-6,
            r_low=-1e-6,
        ),
        plot=replace(
            _CURRENT_PRESET.plot,
            zoom=2.0,
            loc="lower left",
            roi=(80, 30, 60, 60),
            clim=(0.0, 1e-6),
        ),
        spgl1=replace(_CURRENT_PRESET.spgl1, factor=1e7),
    ),
    "si_2_256_measurement_data": replace(
        _CURRENT_PRESET,
        name="si_2_256_measurement_data",
        data=replace(
            _CURRENT_PRESET.data,
            data_name="Si_2_256_measurement_data",
            data_type="original_measurement_data",
            j_order=8,
            r_high=1e-6,
            r_low=-1e-6,
        ),
        plot=replace(
            _CURRENT_PRESET.plot,
            zoom=2.0,
            loc="lower left",
            roi=(32, 42, 50, 50),
            clim=(0.0, 4e-7),
        ),
        spgl1=replace(_CURRENT_PRESET.spgl1, factor=1e7),
    ),
    "cigs_example_256x256": replace(
        _CURRENT_PRESET,
        name="cigs_example_256x256",
        data=replace(
            _CURRENT_PRESET.data,
            data_name="example_CIGS_256x256",
            data_type="image",
            j_order=8,
            inverse_sign=False,
            is_out_of_distribution=False,
            r_high=None,
            r_low=None,
        ),
        plot=replace(
            _CURRENT_PRESET.plot,
            zoom=2.5,
            loc="center left",
            loc1=3,
            loc2=4,
            roi=(110, 210, 40, 40),
            clim=(0.0, 1.0),
        ),
        spgl1=replace(_CURRENT_PRESET.spgl1, factor=1.0),
    ),
}


def show_preset_help() -> None:
    """Show help message about using presets and exit."""
    preset_names = "\n".join(f"  - {name}" for name in sorted(PRESETS))

    body = Text()
    body.append("--preset [PRESET_NAME]\n", style="bold cyan")
    body.append(
        (
            "      Use a saved ExperimentConfig as the base configuration.\n"
            "      Without --preset, every field marked '(required)' below must be passed.\n"
            "      Any explicitly passed flag overrides the preset value.\n\n"
        ),
        style="cyan",
    )
    body.append("Available preset names:\n", style="bold")
    body.append(preset_names)

    console = Console()
    console.print(
        Panel(
            body,
            title="[bold red]NOTE: Preset support[/bold red]",
            border_style="red",
        )
    )


def pop_preset_arg(argv: list[str]) -> tuple[str | None, list[str]]:
    """Pop the --preset argument from argv, if present, and return the preset name and remaining args."""
    preset_name: str | None = None
    remaining: list[str] = []

    i = 0
    while i < len(argv):
        token = argv[i]

        if token == "--preset":
            if i + 1 >= len(argv):
                raise SystemExit("Error: --preset requires a value.")
            if preset_name is not None:
                raise SystemExit("Error: --preset was passed more than once.")
            preset_name = argv[i + 1]
            i += 2
            continue

        if token.startswith("--preset="):
            if preset_name is not None:
                raise SystemExit("Error: --preset was passed more than once.")
            preset_name = token.split("=", 1)[1]
            i += 1
            continue

        remaining.append(token)
        i += 1

    if preset_name is not None and preset_name not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise SystemExit(
            f"Error: invalid preset {preset_name!r}. Valid presets are: {valid}."
        )

    return preset_name, remaining


def parse_experiment_config(argv: list[str] | None = None) -> ExperimentConfig:
    """Parse command-line arguments into a custom dataclass instance."""
    if argv is None:
        argv = sys.argv[1:]

    if "-h" in argv or "--help" in argv:
        show_preset_help()

    preset_name, remaining = pop_preset_arg(argv)

    preset_help_reminder = (
        "    NOTE: Preset support is available through `--preset`! See details at the top.\n"
        "    If you want to use a preset as the base configuration, pass `--preset PRESET_NAME`.\n"
        "    Without --preset, every field marked '(required)' below must be passed.\n"
    )

    base_description = (ExperimentConfig.__doc__ or "").strip()
    description = f"{base_description}\n\n{preset_help_reminder}"

    if preset_name is None:
        return tyro.cli(ExperimentConfig, args=remaining, description=description)

    return tyro.cli(
        ExperimentConfig,
        args=remaining,
        default=PRESETS[preset_name],
        description=description,
    )
