"""Configuration dataclasses and preset definitions for the PCM experiment."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

DataType = Literal["image", "hadamard_measurement_vector", "original_measurement_data"]
RandomisingScheme = Literal["uniform", "multilevel"]


@dataclass(frozen=True)
class PlotConfig:
    """Plotting parameters for the PCM visualisations.

    Parameters
    ----------
    zoom : float
        Zoom factor used for the inset.
    loc : str
        Inset location string passed to the plotting helper.
    loc1 : int
        Connector location on the main axes.
    loc2 : int
        Connector location on the inset axes.
    roi : tuple[int, int, int, int]
        Region of interest as ``(x, y, width, height)``.
    clim : tuple[float, float]
        Colour limits for the plotted images.
    cmap_max : float, default=0.8
        Maximum fraction of the ``afmhot`` colormap to use.
    adds_insets : bool, default=True
        Whether to draw inset zooms.
    show_rect : bool, default=True
        Whether to draw the ROI rectangle.
    """

    zoom: float
    loc: str
    loc1: int
    loc2: int
    roi: tuple[int, int, int, int]
    clim: tuple[float, float]
    cmap_max: float = 0.8
    adds_insets: bool = True
    show_rect: bool = True


@dataclass(frozen=True)
class DataConfig:
    """Input data configuration.

    Parameters
    ----------
    data_dir : Path
        Directory containing the ``.npy`` PCM data files.
    data_name : str
        Stem of the input file.
    data_type : {"image", "hadamard_measurement_vector", "original_measurement_data"}
        Kind of raw input stored in the file.
    j_order : int
        Walsh-Hadamard order ``J`` such that the image size is ``2**J``.
    inverse_sign : bool
        Whether to multiply the loaded data by ``-1``.
    tests_scale_ground_truth : bool, default=False
        Whether to min-max normalise the reconstructed ground truth image.
    is_out_of_distribution : bool, default=False
        Whether the denoiser input/output should be affine-rescaled.
    r_high : float | None, default=None
        Upper end of the expected range for out-of-distribution rescaling.
    r_low : float | None, default=None
        Lower end of the expected range for out-of-distribution rescaling.
    scale_eps : float, default=1e-12
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
    """Runtime and output configuration."""

    root_output_dir: Path
    device: str = "auto"
    noise_seed: int = 42
    noise_std: float = 0.0
    num_trials: int = 1
    num_trials_skip: int = 0


@dataclass(frozen=True)
class PnPConfig:
    """Configuration for PnP-ADMM reconstruction."""

    enabled: bool = True
    denoiser_name: str = "gs_drunet"
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
    """Configuration for SPGL1 reconstruction."""

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

    name: str
    runtime: RuntimeConfig
    data: DataConfig
    plot: PlotConfig
    pnp: PnPConfig = field(default_factory=PnPConfig)
    spgl1: SPGL1Config = field(default_factory=SPGL1Config)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


DEFAULT_DATA_DIR = Path("../pdo/data/photocurrent_data")
DEFAULT_OUTPUT_DIR = Path("../pdo")


_CURRENT_PRESET = ExperimentConfig(
    name="si_2_256_512x512_image",
    runtime=RuntimeConfig(
        root_output_dir=DEFAULT_OUTPUT_DIR,
        device="auto",
        noise_seed=42,
        noise_std=0.0,
        num_trials=1,
        num_trials_skip=0,
    ),
    data=DataConfig(
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
    plot=PlotConfig(
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


def get_preset(name: str) -> ExperimentConfig:
    """Return a named preset.

    Parameters
    ----------
    name : str
        Preset name.

    Returns
    -------
    ExperimentConfig
        Requested experiment configuration.
    """
    try:
        return PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PRESETS))
        raise KeyError(
            f"Unknown preset '{name}'. Available presets: {available}"
        ) from exc
