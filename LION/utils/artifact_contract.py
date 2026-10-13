"""Result artifact paths shared by PCM testing and rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch
import yaml

ARTIFACT_SCHEMA_VERSION = 1

RECONSTRUCTION_TENSOR_FILENAME = "reconstruction.pt"
LAMBDA_1_V_TENSOR_FILENAME = "lambda_1_v.pt"
LAMBDA_0_W_TENSOR_FILENAME = "lambda_0_w.pt"
RESULT_MANIFEST_FILENAME = "manifest.yaml"


@dataclass(frozen=True, kw_only=True)
class ResultPaths:
    """Resolved paths for one reconstruction result."""

    root: Path

    @property
    def reconstruction_tensor(self) -> Path:
        return self.root / RECONSTRUCTION_TENSOR_FILENAME

    @property
    def lambda_1_v_tensor(self) -> Path:
        return self.root / LAMBDA_1_V_TENSOR_FILENAME

    @property
    def lambda_0_w_tensor(self) -> Path:
        return self.root / LAMBDA_0_W_TENSOR_FILENAME

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def lambda_maps_images_dir(self) -> Path:
        return self.images_dir / "lambda_maps"

    @property
    def manifest(self) -> Path:
        return self.root / RESULT_MANIFEST_FILENAME

    def mkdir(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


def pcm_test_dir(run_dir: Path, *, sampling_ratio: float, coarse_j: int) -> Path:
    """Return the result directory for one PCM sampling condition."""
    return Path(run_dir) / f"test_s_{sampling_ratio:.2f}_cJ_{coarse_j}"


def pcm_result_paths(
    run_dir: Path,
    *,
    sampling_ratio: float,
    coarse_j: int,
    epoch: int,
    step: int,
    category: str,
    sample_id: int,
) -> ResultPaths:
    """Return result paths for one PCM sample/checkpoint pair."""
    root = (
        pcm_test_dir(run_dir, sampling_ratio=sampling_ratio, coarse_j=coarse_j)
        / f"epoch_{epoch:04d}_step_{step:06d}"
        / category
        / f"sample_{sample_id:04d}"
    )
    return ResultPaths(root=root)


def pcm_metrics_path(run_dir: Path, *, sampling_ratio: float, coarse_j: int) -> Path:
    """Return the metrics CSV path for one PCM sampling condition."""
    return (
        pcm_test_dir(run_dir, sampling_ratio=sampling_ratio, coarse_j=coarse_j)
        / "results.csv"
    )


def candidate_checkpoint_stems(epoch_or_stem: str) -> set[str]:
    """Return accepted checkpoint stems for one CLI checkpoint specifier."""
    text = str(epoch_or_stem)
    candidates = {text}
    if text.isdigit():
        epoch = int(text)
        candidates.update(
            {
                f"{epoch}",
                f"{epoch:03d}",
                f"{epoch:04d}",
                f"epoch_{epoch:03d}",
                f"epoch_{epoch:04d}",
            }
        )
    return candidates


def parse_checkpoint_epoch(stem: str) -> int | None:
    """Parse an epoch from an ``epoch_XXXX`` or numeric checkpoint stem."""
    if stem.startswith("epoch_"):
        stem = stem.removeprefix("epoch_")
    return int(stem) if stem.isdigit() else None


def load_saved_artifacts(
    paths: ResultPaths,
    *,
    load_reconstruction: bool,
    load_lambda_maps: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Load requested saved reconstruction/lambda tensors."""
    reconstruction = None
    if load_reconstruction and paths.reconstruction_tensor.exists():
        reconstruction = torch.load(
            paths.reconstruction_tensor, map_location="cpu", weights_only=False
        )

    lambda_maps = None
    if load_lambda_maps and paths.lambda_1_v_tensor.exists():
        lambda_1_v = torch.load(
            paths.lambda_1_v_tensor, map_location="cpu", weights_only=False
        )
        if paths.lambda_0_w_tensor.exists():
            lambda_0_w = torch.load(
                paths.lambda_0_w_tensor, map_location="cpu", weights_only=False
            )
            lambda_maps = torch.stack((lambda_1_v, lambda_0_w), dim=0)
        else:
            lambda_maps = lambda_1_v

    return reconstruction, lambda_maps


def write_result_manifest(
    paths: ResultPaths,
    *,
    reconstruction_tensor_written: bool,
    reconstruction_images_written: bool,
    lambda_1_v_tensor_written: bool,
    lambda_0_w_tensor_written: bool,
    lambda_images_written: bool,
) -> None:
    """Write a small manifest describing the artifacts that were produced."""
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "domain": "pcm",
        "artifacts": {
            "reconstruction_tensor": paths.reconstruction_tensor.name
            if reconstruction_tensor_written
            else None,
            "reconstruction_images_dir": paths.images_dir.name
            if reconstruction_images_written
            else None,
            "lambda_1_v_tensor": paths.lambda_1_v_tensor.name
            if lambda_1_v_tensor_written
            else None,
            "lambda_0_w_tensor": paths.lambda_0_w_tensor.name
            if lambda_0_w_tensor_written
            else None,
            "lambda_maps_images_dir": (
                str(paths.lambda_maps_images_dir.relative_to(paths.root))
                if lambda_images_written
                else None
            ),
        },
    }
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(yaml.safe_dump(payload, sort_keys=False))


FINAL_RECON_METRICS_KEY = "final_recon"
TEST_MSE_RECON_METRICS_KEY = "test_mse_recon"
VAL_SSDU_LOSS_RECON_METRICS_KEY = "val_ssdu_loss_recon"
BEST_TEST_MSE_EPOCH_KEY = "best_test_mse_epoch"
BEST_VAL_SSDU_LOSS_EPOCH_KEY = "best_val_ssdu_loss_epoch"


class ResultName(StrEnum):
    """Stable result directory names used by self-supervised training."""

    FINAL = "final"
    BEST_TEST_MSE = "best_test_mse"
    BEST_VAL_SSDU_LOSS = "best_val_ssdu_loss"


@dataclass(frozen=True)
class ReconstructionMetricsContract:
    """One reconstruction metric block written to ``main_results.yaml``."""

    output_name: str
    yaml_key: str
    epoch_source: int | str
    required: bool = True


COMMON_RECONSTRUCTION_METRICS = (
    ReconstructionMetricsContract("final", FINAL_RECON_METRICS_KEY, "epochs"),
    ReconstructionMetricsContract(
        "test_mse_recon",
        TEST_MSE_RECON_METRICS_KEY,
        BEST_TEST_MSE_EPOCH_KEY,
        required=False,
    ),
    ReconstructionMetricsContract(
        "val_ssdu_loss_recon",
        VAL_SSDU_LOSS_RECON_METRICS_KEY,
        BEST_VAL_SSDU_LOSS_EPOCH_KEY,
        required=False,
    ),
)


@dataclass(frozen=True)
class SelfSupervisedDomainContract:
    """Minimal shared result contract for one self-supervised domain."""

    domain: str
    condition_key: str
    condition_label: str
    condition_type: type[int] | type[float] | type[str]
    run_dir_regex: re.Pattern[str]
    run_dir_format: str
    initial_reconstruction: ReconstructionMetricsContract
    default_mse_exp_scale: int = 0

    @property
    def reconstructions(self) -> tuple[ReconstructionMetricsContract, ...]:
        return (self.initial_reconstruction, *COMMON_RECONSTRUCTION_METRICS)

    def format_run_dir_name(
        self, *, condition: int | float | str, sample_id: int
    ) -> str:
        return self.run_dir_format.format(condition=condition, sample_id=sample_id)

    def parse_run_dir_name(self, name: str) -> tuple[int | float | str, int] | None:
        match = self.run_dir_regex.fullmatch(name)
        if match is None:
            return None
        return self.condition_type(match.group("condition")), int(match.group("sample"))


PCM_SELF_SUPERVISED_CONTRACT = SelfSupervisedDomainContract(
    domain="pcm",
    condition_key="sampling_case",
    condition_label="",
    condition_type=str,
    run_dir_regex=re.compile(r"^(?P<condition>.+)_sample_(?P<sample>\d+)$"),
    run_dir_format="{condition}_sample_{sample_id:04d}",
    initial_reconstruction=ReconstructionMetricsContract(
        "zero_filled",
        "zero_filled",
        -1,
    ),
    default_mse_exp_scale=0,
)


BEST_RECONSTRUCTION_METRICS_KEYS = {
    "test_mse": TEST_MSE_RECON_METRICS_KEY,
    "val_ssdu_loss": VAL_SSDU_LOSS_RECON_METRICS_KEY,
}


def make_self_supervised_main_results(
    *,
    contract: SelfSupervisedDomainContract,
    sample_id: int,
    condition: int | float | str,
    epochs: int,
    initial_metrics: dict[str, float],
    final_metrics: dict[str, float],
    best: dict[str, float | int],
    time: dict[str, float],
) -> dict[str, object]:
    """Build the common ``main_results.yaml`` structure."""
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "domain": contract.domain,
        "sample_id": sample_id,
        contract.condition_key: condition,
        "epochs": epochs,
        contract.initial_reconstruction.yaml_key: initial_metrics,
        FINAL_RECON_METRICS_KEY: final_metrics,
        "best": best,
        "time": time,
    }
