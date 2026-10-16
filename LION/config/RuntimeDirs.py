from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from LION.utils.artifact_contract import ResultPaths

REQUESTED_CONFIG_FILENAME = "requested_config.yaml"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
METRICS_FILENAME = "metrics.csv"
MAIN_RESULTS_FILENAME = "main_results.yaml"
MLFLOW_RUN_ID_FILENAME = "mlflow_run_id.txt"


@dataclass(kw_only=True)
class RuntimeDirs:
    """Resolved output paths for one run."""

    results_dir: Path | None = None
    run_name: str | None = None
    run_dir: Path | None = None
    checkpoints_dir: Path | None = None
    recons_dir: Path | None = None
    figures_dir: Path | None = None
    images_dir: Path | None = None
    generated_dir: Path | None = None
    output_dir: Path | None = None
    permutation_file: Path | None = None

    def require_run_dir(self) -> Path:
        if self.run_dir is None:
            raise ValueError("dirs.run_dir must be specified.")
        return Path(self.run_dir)

    def require_recons_dir(self) -> Path:
        if self.recons_dir is None:
            raise ValueError("dirs.recons_dir must be specified.")
        return Path(self.recons_dir)

    def require_checkpoints_dir(self) -> Path:
        if self.checkpoints_dir is None:
            raise ValueError("dirs.checkpoints_dir must be specified.")
        return Path(self.checkpoints_dir)

    @property
    def requested_config_path(self) -> Path:
        return self.require_run_dir() / REQUESTED_CONFIG_FILENAME

    @property
    def resolved_config_path(self) -> Path:
        return self.require_run_dir() / RESOLVED_CONFIG_FILENAME

    @property
    def metrics_path(self) -> Path:
        return self.require_run_dir() / METRICS_FILENAME

    @property
    def main_results_path(self) -> Path:
        return self.require_run_dir() / MAIN_RESULTS_FILENAME

    @property
    def mlflow_run_id_path(self) -> Path:
        return self.require_run_dir() / MLFLOW_RUN_ID_FILENAME

    def result_paths(self, name: str) -> ResultPaths:
        """Return canonical result paths for one named reconstruction."""
        return ResultPaths(root=self.require_recons_dir() / name)

    def mkdir(self) -> None:
        for path in (
            self.run_dir,
            self.checkpoints_dir,
            self.recons_dir,
            self.figures_dir,
            self.images_dir,
            self.generated_dir,
            self.output_dir,
        ):
            if path is not None:
                Path(path).mkdir(parents=True, exist_ok=True)


def get_run_dirs(results_dir: Path, run_name: str, mkdir: bool = False) -> RuntimeDirs:
    """Resolve the standard directories for one experiment run."""
    results_dir = Path(results_dir)
    run_dir = results_dir / run_name
    dirs = RuntimeDirs(
        results_dir=results_dir,
        run_name=run_name,
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        recons_dir=run_dir / "recons",
        figures_dir=run_dir / "figures",
        images_dir=run_dir / "images",
        generated_dir=run_dir / "generated",
        output_dir=run_dir / "output",
        permutation_file=run_dir / "permutation.txt",
    )
    if mkdir:
        dirs.mkdir()
    return dirs
