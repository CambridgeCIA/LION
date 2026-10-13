from __future__ import annotations

import csv
import dataclasses
import random
from math import ceil
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import mlflow
import numpy as np
import torch
import yaml

from LION.config.ExperimentConfig import ExperimentConfig
from LION.config.RuntimeDirs import RuntimeDirs
from LION.utils.log import CompactListDumper, to_yaml_friendly
from LION.utils.types import TrainingState

ExperimentT = TypeVar("ExperimentT", bound=ExperimentConfig)

LogBy = Literal["epoch", "step"]
LogStage = Literal["train", "val", "test"]
LogKind = Literal["loss", "metrics", "params"]
BestMode = Literal["min", "max"]


class Logger(Generic[ExperimentT]):
    def __init__(
        self,
        config: ExperimentT,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rng: torch.Generator,
        training_state: TrainingState,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.rng = rng
        self.training_state = training_state

        dirs = self.config.dirs
        self.config_log_dict = Logger.make_config_log_dict(
            config=self.config, dirs=dirs
        )

        if dirs.run_dir is None:
            raise ValueError("dirs.run_dir must be specified in the config.")
        with dirs.resolved_config_path.open("w") as f:
            yaml.dump(
                self.config_log_dict, f, Dumper=CompactListDumper, sort_keys=False
            )

    @property
    def one_indexed_epoch(self) -> int:
        return self.training_state.one_indexed_epoch

    @property
    def one_indexed_step(self) -> int:
        return self.training_state.one_indexed_global_step

    @staticmethod
    def _dataclass_asdict(value: object) -> dict[str, Any]:
        if not dataclasses.is_dataclass(value) or isinstance(value, type):
            raise TypeError(f"Expected dataclass instance, got {type(value)}")
        return dataclasses.asdict(value)

    @staticmethod
    def make_config_log_dict(
        *, config: object, dirs: RuntimeDirs | None = None
    ) -> dict[str, Any]:
        config_dict = Logger._dataclass_asdict(config)
        if dirs is not None:
            config_dict["dirs"] = dataclasses.asdict(dirs)
        return to_yaml_friendly(config_dict)

    @staticmethod
    def flatten_dataclass_config(
        config: object, key_prefix: str | None = None
    ) -> dict[str, Any]:
        if dataclasses.is_dataclass(config):
            config = Logger._dataclass_asdict(config)
        if not isinstance(config, dict):
            raise TypeError(f"Expected dataclass or dictionary, got {type(config)}")

        flat_config: dict[str, Any] = {}
        for key, value in config.items():
            combined_key = f"{key_prefix}.{key}" if key_prefix else str(key)
            if isinstance(value, dict):
                flat_config.update(
                    Logger.flatten_dataclass_config(value, key_prefix=combined_key)
                )
            else:
                flat_config[combined_key] = value
        return flat_config

    def log_config_to_mlflow(self) -> None:
        if not self.config.log.log_mlflow:
            return
        mlflow.log_params(
            {k: str(v) for k, v in Logger.flatten_dataclass_config(self.config).items()}
        )
        mlflow.log_text(
            yaml.dump(self.config_log_dict, Dumper=CompactListDumper, sort_keys=False),
            "config.yaml",
        )

    def log_metric(self, metric: str, value: float, *, step: int) -> None:
        if self.config.log.log_mlflow:
            mlflow.log_metric(metric, value, step=step)
        if self.config.log.log_csv:
            self._append_metric_to_csv(metric=metric, value=value, step=step)

    def _append_metric_to_csv(self, *, metric: str, value: float, step: int) -> None:
        metrics_path = self.config.dirs.metrics_path
        needs_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
        with metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if needs_header:
                writer.writerow(["step", "metric", "value"])
            writer.writerow([step, metric, float(value)])

    def log_index(self, log_by: LogBy) -> int:
        return self.one_indexed_epoch if log_by == "epoch" else self.one_indexed_step

    def checkpoint_name(self, log_by: LogBy) -> str:
        if log_by == "epoch":
            return f"epoch_{self.one_indexed_epoch:04d}"
        return f"step_{self.one_indexed_step:06d}"

    def should_log(self, stage: LogStage, kind: LogKind, log_by: LogBy) -> bool:
        suffix = "epochs" if log_by == "epoch" else "n_steps"
        every_n = getattr(self.config.log, f"log_{stage}_{kind}_every_{suffix}")
        return self.is_logging(one_indexed_i_=self.log_index(log_by), every_n_=every_n)

    @staticmethod
    def update_best_value(
        value: float, best_value: float, *, mode: BestMode
    ) -> tuple[float, bool]:
        improved = value < best_value if mode == "min" else value > best_value
        return (value if improved else best_value), improved

    def is_within_checkpoint_range(self, log_by: LogBy) -> bool:
        if log_by == "epoch":
            current = self.one_indexed_epoch
            start = self.config.log.ckpt.start_epoch
            end = self.config.log.ckpt.end_epoch
        else:
            current = self.one_indexed_step
            start = self.config.log.ckpt.start_step
            end = self.config.log.ckpt.end_step

        if start is not None and current < start:
            return False
        if end is not None and current > end:
            return False
        return True

    def snapshot_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached CPU copy of the current model state."""
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def is_checkpoint_epoch(self) -> bool:
        cur_epoch = self.one_indexed_epoch
        if (
            self.config.log.ckpt.epochs is not None
            and cur_epoch in self.config.log.ckpt.epochs
        ):
            return True
        if not self.is_within_checkpoint_range("epoch"):
            return False
        return self.is_logging(
            one_indexed_i_=cur_epoch,
            every_n_=self.config.log.ckpt.every_n_epochs,
        )

    def is_checkpoint_step(self) -> bool:
        cur_step = self.one_indexed_step
        if (
            self.config.log.ckpt.steps is not None
            and cur_step in self.config.log.ckpt.steps
        ):
            return True
        if not self.is_within_checkpoint_range("step"):
            return False
        return self.is_logging(
            one_indexed_i_=cur_step, every_n_=self.config.log.ckpt.every_n_steps
        )

    def _save_checkpoint(self, ckpt_name: str) -> None:
        if not self.config.log.ckpt.local and not self.config.log.ckpt.mlflow:
            return

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_state": self.rng.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "np_random_state": np.random.get_state(),
            "random_random_state": random.getstate(),
            "epoch": self.one_indexed_epoch,
            "global_step": self.one_indexed_step,
            "config": self.config_log_dict,
        }

        checkpoints_dir = self.config.dirs.require_checkpoints_dir()
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoints_dir / f"{ckpt_name}.pt"
        torch.save(checkpoint, checkpoint_path)

        if self.config.log.ckpt.mlflow:
            mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        if not self.config.log.ckpt.local:
            checkpoint_path.unlink()

    def save_best_checkpoint_if_needed(
        self, *, is_best: dict[object, bool] | None, criterion: str | None
    ) -> None:
        if is_best is None:
            return

        if self.config.log.ckpt.log_best:
            assert criterion is not None
            for label, improved in is_best.items():
                if improved:
                    suffix = f"_{label}" if label != "" else ""
                    self._save_checkpoint(f"best_{criterion}{suffix}")

        block_size = self.config.log.ckpt.best_checkpoint_epoch_block_size
        if block_size is not None and block_size > 0 and criterion is not None:
            if self.one_indexed_epoch == 0:
                suffix = "_initial"
            else:
                block_end = ceil(self.one_indexed_epoch / block_size) * block_size
                block_start = block_end - block_size + 1
                suffix = f"_new_best_in_epochs_{block_start:04d}_{block_end:04d}"
            for label, improved in is_best.items():
                if improved:
                    label_suffix = f"_{label}" if label != "" else ""
                    self._save_checkpoint(f"best_{criterion}{label_suffix}{suffix}")

    def save_checkpoint_if_needed(
        self,
        ckpt_name: str | None = None,
        *,
        is_best: dict[object, bool] | None = None,
        criterion: str | None = None,
        log_by: LogBy = "epoch",
    ) -> None:
        if ckpt_name is None:
            ckpt_name = self.checkpoint_name(log_by)

        is_scheduled = (
            self.is_checkpoint_epoch()
            if log_by == "epoch"
            else self.is_checkpoint_step()
        )
        if is_scheduled:
            self._save_checkpoint(ckpt_name)

        self.save_best_checkpoint_if_needed(is_best=is_best, criterion=criterion)

        if self.config.log.ckpt.log_cur:
            self._save_checkpoint("current")
        if (
            log_by == "epoch"
            and self.config.log.ckpt.log_final
            and self.one_indexed_epoch == self.config.train.num_epochs
        ):
            self._save_checkpoint("final")

    def is_logging(self, *, one_indexed_i_: int, every_n_: int | None) -> bool:
        return (every_n_ is not None and every_n_ > 0) and (
            one_indexed_i_ % every_n_ == 0
        )

    def in_logging_step(self, every_n_steps: int | None) -> bool:
        return self.is_logging(
            one_indexed_i_=self.one_indexed_step, every_n_=every_n_steps
        )

    def in_logging_epoch(self, every_n_epochs: int | None) -> bool:
        return self.is_logging(
            one_indexed_i_=self.one_indexed_epoch, every_n_=every_n_epochs
        )

    def is_train_loss_log_epoch(self) -> bool:
        return self.should_log("train", "loss", "epoch")

    def is_train_metrics_log_epoch(self) -> bool:
        return self.should_log("train", "metrics", "epoch")

    def is_train_params_log_epoch(self) -> bool:
        return self.should_log("train", "params", "epoch")

    def is_val_loss_log_epoch(self) -> bool:
        return self.should_log("val", "loss", "epoch")

    def is_val_metrics_log_epoch(self) -> bool:
        return self.should_log("val", "metrics", "epoch")

    def is_test_loss_log_epoch(self) -> bool:
        return self.should_log("test", "loss", "epoch")

    def is_test_metrics_log_epoch(self) -> bool:
        return self.should_log("test", "metrics", "epoch")

    def is_train_loss_log_step(self) -> bool:
        return self.should_log("train", "loss", "step")

    def is_train_metrics_log_step(self) -> bool:
        return self.should_log("train", "metrics", "step")

    def is_train_params_log_step(self) -> bool:
        return self.should_log("train", "params", "step")

    def is_val_loss_log_step(self) -> bool:
        return self.should_log("val", "loss", "step")

    def is_val_metrics_log_step(self) -> bool:
        return self.should_log("val", "metrics", "step")

    def is_test_loss_log_step(self) -> bool:
        return self.should_log("test", "loss", "step")

    def is_test_metrics_log_step(self) -> bool:
        return self.should_log("test", "metrics", "step")

    def save_main_results(self, main_results: dict) -> None:
        save_path = self.config.log.final_metrics_path
        if save_path is None:
            save_path = self.config.dirs.main_results_path
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as f:
            yaml.dump(
                to_yaml_friendly(main_results),
                f,
                Dumper=CompactListDumper,
                sort_keys=False,
            )

    def is_validating(self, log_by: LogBy) -> bool:
        if self.one_indexed_step == 0 and self.one_indexed_epoch == 0:
            return self.config.log.val_before_start
        return self.should_log("val", "loss", log_by) or self.should_log(
            "val", "metrics", log_by
        )
