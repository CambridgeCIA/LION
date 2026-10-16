"""Shared zero-shot/self-supervised training loop for single-sample inverse problems."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

import torch

from LION.config.ExperimentConfig import ExperimentConfig
from LION.models.model_loader import ModelLoader, ModelSetup
from LION.train.Logger import Logger
from LION.train.Trainer import Trainer
from LION.train.training_components import ForwardResult, ParameterGroupBuilder
from LION.utils.artifact_contract import (
    BEST_RECONSTRUCTION_METRICS_KEYS,
    ResultName,
    SelfSupervisedDomainContract,
    make_self_supervised_main_results,
)
from LION.utils.types import StateDict

ExperimentT = TypeVar("ExperimentT", bound=ExperimentConfig)
DataT = TypeVar("DataT")
SplitT = TypeVar("SplitT")
ProblemT = TypeVar("ProblemT", bound="SelfSupervisedProblem")
TrainerT = TypeVar("TrainerT", bound="SelfSupervisedTrainer")
LoggerT = TypeVar("LoggerT", bound="SelfSupervisedLoggerBase")

SelfSupervisedCriterion = Literal["final", "best_val_ssdu_loss"]


class SelfSupervisedProblem(
    ABC, Generic[ExperimentT, DataT, SplitT, TrainerT, LoggerT]
):
    """Domain adapter used by the shared self-supervised trainer."""

    def __init__(self, config: ExperimentT, sample_id: int) -> None:
        super().__init__()
        self.config = config
        self.sample_id = sample_id
        self.target_image: torch.Tensor | None = None
        self.model_setup = ModelSetup()

    @abstractmethod
    def setup(self, trainer: SelfSupervisedTrainer) -> None:
        ...

    @abstractmethod
    def make_logger(self, trainer: TrainerT) -> LoggerT:
        ...

    @abstractmethod
    def has_validation(self) -> bool:
        ...

    @abstractmethod
    def make_train_split(self) -> SplitT:
        ...

    @abstractmethod
    def get_val_split(self) -> SplitT:
        ...

    @abstractmethod
    def get_test_data(self) -> DataT:
        ...

    @abstractmethod
    def run_forward(
        self,
        trainer: SelfSupervisedTrainer,
        data: DataT | SplitT,
        *,
        model: torch.nn.Module | None = None,
        skip_grad_checkpoint: bool,
    ) -> ForwardResult:
        ...

    def run_final_forward(
        self,
        trainer: SelfSupervisedTrainer,
        data: DataT | SplitT,
        *,
        result_name: str,
        model: torch.nn.Module | None = None,
    ) -> ForwardResult:
        del result_name
        return self.run_forward(trainer, data, model=model, skip_grad_checkpoint=True)

    def compute_lambda_maps(
        self,
        trainer: SelfSupervisedTrainer,
        data: DataT | SplitT,
        *,
        model: torch.nn.Module | None = None,
    ) -> torch.Tensor | None:
        del trainer, data, model
        return None

    @abstractmethod
    def compute_target_loss(self, recon: torch.Tensor, split: SplitT) -> torch.Tensor:
        ...

    @abstractmethod
    def postprocess_recon(self, recon: torch.Tensor) -> torch.Tensor:
        ...

    def postprocess_result(self, result: ForwardResult) -> ForwardResult:
        recon = self.postprocess_recon(result.reconstruction)
        extra_output = (
            result.extra_output.detach().cpu()
            if result.extra_output is not None
            else None
        )
        return ForwardResult(reconstruction=recon, extra_output=extra_output)

    def after_step_cleanup(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abstractmethod
    def train_epoch_desc(self) -> str:
        ...


class SelfSupervisedTimer:
    """Accumulate wall-clock timings for training and evaluation."""

    _timing_fields = {
        "total_time": "total_time",
        "total_train_forward_time": "total_train_forward",
        "total_train_backward_time": "total_train_backward",
        "total_val_forward_time": "total_val_forward",
        "total_test_forward_time": "total_test_forward",
        "total_sim_time": "total_sim",
        "total_compute_metrics_time": "total_compute_metrics",
    }

    def start_timer(self) -> None:
        self.start_time = time.time()
        for field in self._timing_fields:
            setattr(self, field, 0.0)

    def log_timings(self, epoch: int) -> None:
        for field, metric in self._timing_fields.items():
            self.log_metric(metric, getattr(self, field), step=epoch)

    def add_elapsed(self, field: str, start_time: float) -> None:
        setattr(self, field, getattr(self, field) + time.time() - start_time)

    def update_total_time(self) -> None:
        self.total_time = time.time() - self.start_time


class SelfSupervisedLoggerBase(Logger, SelfSupervisedTimer, ABC, Generic[ProblemT]):
    """Best-checkpoint and timing logic shared by self-supervised image problems."""

    def __init__(
        self,
        *,
        trainer: SelfSupervisedTrainer,
        checkpoint_label: str,
        problem: ProblemT,
    ) -> None:
        super().__init__(
            config=trainer.config,
            model=trainer.model,
            optimizer=trainer.optimizer,
            rng=trainer.rng,
            training_state=trainer.state,
        )
        self.trainer = trainer
        self.checkpoint_label = checkpoint_label
        self.problem = problem
        self.best_val_ssdu_loss = float("inf")
        self.best_val_ssdu_loss_epoch = 0
        self.best_val_ssdu_loss_state_dict: StateDict | None = None
        self.best_test_mse = float("inf")
        self.best_test_mse_epoch = 0
        self.best_test_mse_state_dict: StateDict | None = None

    def update_best_val_ssdu_loss(self, loss_value: float) -> None:
        self.best_val_ssdu_loss, improved = self.update_best_value(
            loss_value, self.best_val_ssdu_loss, mode="min"
        )
        if not improved:
            return
        self.best_val_ssdu_loss_state_dict = self.snapshot_state_dict()
        self.best_val_ssdu_loss_epoch = self.one_indexed_epoch
        self.save_best_checkpoint_if_needed(
            is_best={self.checkpoint_label: True}, criterion="val_ssdu_loss"
        )

    def update_best_test_mse(self, mse: float | None) -> None:
        if mse is None:
            return
        self.best_test_mse, improved = self.update_best_value(
            mse, self.best_test_mse, mode="min"
        )
        if not improved:
            return
        self.best_test_mse_state_dict = self.snapshot_state_dict()
        self.best_test_mse_epoch = self.one_indexed_epoch
        self.save_best_checkpoint_if_needed(
            is_best={self.checkpoint_label: True}, criterion="test_mse"
        )

    def get_best_result(
        self, criterion: Literal["val_ssdu_loss", "test_mse"]
    ) -> ForwardResult | None:
        if criterion == "test_mse":
            state_dict = self.best_test_mse_state_dict
            result_name = ResultName.BEST_TEST_MSE.value
        else:
            state_dict = self.best_val_ssdu_loss_state_dict
            result_name = ResultName.BEST_VAL_SSDU_LOSS.value

        if state_dict is None:
            return None
        return self.trainer.recon_with_state_dict(state_dict, result_name=result_name)

    @abstractmethod
    def initial_save(self) -> None:
        ...

    @abstractmethod
    def log_train(self, loss_value: float, recon: torch.Tensor) -> None:
        ...

    @abstractmethod
    def log_val(self, loss_value: float, recon: torch.Tensor) -> None:
        ...

    @abstractmethod
    def log_test(self, recon: torch.Tensor) -> tuple[float, float, float, float]:
        ...

    @abstractmethod
    def finalize(self, final_result: ForwardResult) -> None:
        ...


MetricTuple = tuple[float, float, float, float]


class SelfSupervisedImageLogger(
    SelfSupervisedLoggerBase[ProblemT], ABC, Generic[ProblemT]
):
    """Shared metric logging and final-result handling for image reconstruction."""

    def __init__(
        self,
        *,
        trainer: SelfSupervisedTrainer,
        checkpoint_label: str,
        problem: ProblemT,
        contract: SelfSupervisedDomainContract,
        condition: int | float | str,
        target_image: torch.Tensor,
    ) -> None:
        super().__init__(
            trainer=trainer, checkpoint_label=checkpoint_label, problem=problem
        )
        self.contract = contract
        self.condition = condition
        self.target_image = target_image

    @abstractmethod
    def compute_metrics(self, reconstruction: torch.Tensor) -> MetricTuple:
        ...

    @abstractmethod
    def initial_reconstruction(self) -> torch.Tensor:
        ...

    @abstractmethod
    def save_result(self, result: ForwardResult, name: str) -> None:
        ...

    def log_train_model_params(self) -> None:
        pass

    @staticmethod
    def metrics_dict(metrics: MetricTuple) -> dict[str, float]:
        return dict(zip(("mse", "psnr", "ssim", "haarpsi"), metrics, strict=True))

    def compute_and_log_metrics_epoch(
        self,
        reconstruction: torch.Tensor,
        stage: Literal["train", "val", "test"],
    ) -> MetricTuple:
        t_start = time.time()
        metrics = self.compute_metrics(reconstruction)
        self.add_elapsed("total_compute_metrics_time", t_start)
        for name, value in self.metrics_dict(metrics).items():
            self.log_metric(f"{stage}_{name}_epoch", value, step=self.one_indexed_epoch)
        return metrics

    def log_train(self, loss_value: float, recon: torch.Tensor) -> None:
        if self.should_log("train", "loss", "epoch"):
            self.log_metric(
                "train_ssdu_loss_epoch", loss_value, step=self.one_indexed_epoch
            )
        if self.should_log("train", "metrics", "epoch"):
            self.compute_and_log_metrics_epoch(recon, "train")
        self.log_train_model_params()

    def log_val(self, loss_value: float, recon: torch.Tensor) -> None:
        if self.should_log("val", "loss", "epoch"):
            self.log_metric(
                "val_ssdu_loss_epoch", loss_value, step=self.one_indexed_epoch
            )
        if self.should_log("val", "metrics", "epoch"):
            self.compute_and_log_metrics_epoch(recon, "val")

    def log_test(self, recon: torch.Tensor) -> MetricTuple:
        return self.compute_and_log_metrics_epoch(recon, "test")

    def finalize_results(
        self,
        final_result: ForwardResult,
        best_results: dict[Literal["test_mse", "val_ssdu_loss"], ForwardResult | None],
    ) -> None:
        self.update_total_time()
        best = {
            "best_val_ssdu_loss": self.best_val_ssdu_loss,
            "best_val_ssdu_loss_epoch": self.best_val_ssdu_loss_epoch,
            "best_test_mse": self.best_test_mse,
            "best_test_mse_epoch": self.best_test_mse_epoch,
        }
        main_results = make_self_supervised_main_results(
            contract=self.contract,
            sample_id=self.problem.sample_id,
            condition=self.condition,
            epochs=self.one_indexed_epoch,
            initial_metrics=self.metrics_dict(
                self.compute_metrics(self.initial_reconstruction())
            ),
            final_metrics=self.metrics_dict(
                self.compute_metrics(final_result.reconstruction)
            ),
            best=best,
            time={
                "total": self.total_time,
                "total_train_forward": self.total_train_forward_time,
                "total_train_backward": self.total_train_backward_time,
                "total_val_forward": self.total_val_forward_time,
                "total_test_forward": self.total_test_forward_time,
                "total_compute_metrics": self.total_compute_metrics_time,
            },
        )

        for criterion, result in best_results.items():
            if result is not None:
                main_results[
                    BEST_RECONSTRUCTION_METRICS_KEYS[criterion]
                ] = self.metrics_dict(self.compute_metrics(result.reconstruction))

        self.save_main_results(main_results)
        self.save_result(final_result, ResultName.FINAL.value)
        for criterion, result in best_results.items():
            if result is not None:
                self.save_result(result, f"best_{criterion}")

    def finalize(self, final_result: ForwardResult) -> None:
        self.finalize_results(
            final_result,
            {
                "test_mse": self.get_best_result("test_mse"),
                "val_ssdu_loss": self.get_best_result("val_ssdu_loss"),
            },
        )


class SelfSupervisedTrainer(Trainer[ExperimentT], Generic[ExperimentT, ProblemT]):
    """Shared single-sample SSDU trainer."""

    def __init__(
        self,
        config: ExperimentT,
        parameter_group_builder: ParameterGroupBuilder,
        model_loader: ModelLoader,
        problem: ProblemT,
        epoch_size: int = 1,
    ) -> None:
        super().__init__(config, parameter_group_builder, model_loader)
        self.problem = problem
        super()._setup(epoch_size=epoch_size, model_setup=problem.model_setup)
        self.problem.setup(self)
        self.logger = self.problem.make_logger(self)

    @staticmethod
    def _clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward_backward_train(self) -> tuple[float, torch.Tensor]:
        t_start_sim = time.time()
        split = self.problem.make_train_split()
        self.logger.add_elapsed("total_sim_time", t_start_sim)

        self.optimizer.zero_grad(set_to_none=True)

        t_start_forward = time.time()
        result = self.problem.run_forward(self, split, skip_grad_checkpoint=False)
        self.logger.add_elapsed("total_train_forward_time", t_start_forward)

        loss = self.problem.compute_target_loss(result.reconstruction, split)

        t_start_backward = time.time()
        loss_value = self.optimizer_step_from_loss(loss)
        self.logger.add_elapsed("total_train_backward_time", t_start_backward)

        recon_cpu = self.problem.postprocess_recon(result.reconstruction)

        del split, result, loss
        self.problem.after_step_cleanup()
        self._clear_cuda_cache()
        return loss_value, recon_cpu

    @torch.no_grad()
    def forward_val(self) -> tuple[float, torch.Tensor]:
        split = self.problem.get_val_split()

        t_start_forward = time.time()
        result = self.problem.run_forward(self, split, skip_grad_checkpoint=True)
        self.logger.add_elapsed("total_val_forward_time", t_start_forward)

        loss = self.problem.compute_target_loss(result.reconstruction, split)
        recon_cpu = self.problem.postprocess_recon(result.reconstruction)
        loss_value = float(loss.item())

        del result, loss
        self.problem.after_step_cleanup()
        self._clear_cuda_cache()
        return loss_value, recon_cpu

    @torch.no_grad()
    def forward_test_result(self, *, result_name: str | None = None) -> ForwardResult:
        data = self.problem.get_test_data()
        t_start_forward = time.time()
        if result_name is None:
            result = self.problem.run_forward(self, data, skip_grad_checkpoint=True)
        else:
            result = self.problem.run_final_forward(self, data, result_name=result_name)
        self.logger.add_elapsed("total_test_forward_time", t_start_forward)
        return result

    @torch.no_grad()
    def forward_test(self) -> torch.Tensor:
        result = self.forward_test_result()
        recon_cpu = self.problem.postprocess_recon(result.reconstruction)
        del result
        self.problem.after_step_cleanup()
        self._clear_cuda_cache()
        return recon_cpu

    def validate_if_needed(self) -> None:
        if not self.problem.has_validation():
            return
        if not self.logger.is_validating("epoch"):
            return

        was_training = self.model.training
        self.model.eval()
        loss_value, recon_cpu = self.forward_val()
        if was_training:
            self.model.train()

        self.logger.log_val(loss_value, recon_cpu)
        self.logger.update_best_val_ssdu_loss(loss_value)
        del recon_cpu
        self._clear_cuda_cache()

    def test_if_needed(self) -> None:
        if not self.logger.should_log("test", "metrics", "epoch"):
            return

        was_training = self.model.training
        self.model.eval()
        recon_cpu = self.forward_test()
        if was_training:
            self.model.train()

        mse, _, _, _ = self.logger.log_test(recon_cpu)
        self.logger.update_best_test_mse(mse)
        del recon_cpu
        self._clear_cuda_cache()

    def train(self) -> None:
        self.logger.start_timer()
        self.logger.log_config_to_mlflow()
        self.logger.initial_save()
        self.validate_if_needed()
        self.test_if_needed()

        self.logger.save_checkpoint_if_needed(log_by="epoch")
        self.model.train()

        for _ in self.train_epoch_iterator(desc=self.problem.train_epoch_desc()):
            self.increment_epoch()
            self.increment_global_step()

            loss_value, recon_cpu = self.forward_backward_train()
            self.logger.log_train(loss_value, recon_cpu)
            del recon_cpu

            self.validate_if_needed()
            self.test_if_needed()
            self.logger.save_checkpoint_if_needed(log_by="epoch")
            self._clear_cuda_cache()

            self.logger.update_total_time()
            self.logger.log_timings(self.one_indexed_epoch)

    @torch.no_grad()
    def get_result(self, selection: SelfSupervisedCriterion) -> ForwardResult:
        """Return the selected full-data reconstruction result."""
        if selection == "best_val_ssdu_loss":
            result = self.logger.get_best_result("val_ssdu_loss")
            if result is not None:
                return result
        elif selection != "final":
            raise ValueError(
                f"Unsupported self-supervised result selection: {selection!r}"
            )

        result = self.forward_test_result(result_name=ResultName.FINAL.value)
        return self.problem.postprocess_result(result)

    def finalize(self) -> None:
        self.model.eval()
        final_result = self.get_result("final")
        self.logger.finalize(final_result)
        del final_result
        self.problem.after_step_cleanup()
        self._clear_cuda_cache()

    def recon_with_state_dict(
        self, state_dict: StateDict, *, result_name: str | None = None
    ) -> ForwardResult:
        model, _ = self.model_loader(self.config, setup=self.problem.model_setup)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.config.train.device)
        model.eval()

        data = self.problem.get_test_data()
        with torch.no_grad():
            if result_name is None:
                result = self.problem.run_forward(
                    self, data, model=model, skip_grad_checkpoint=True
                )
            else:
                result = self.problem.run_final_forward(
                    self,
                    data,
                    result_name=result_name,
                    model=model,
                )
            result_cpu = self.problem.postprocess_result(result)

        del model, result
        self._clear_cuda_cache()
        return result_cpu

    def lambda_maps_with_state_dict(self, state_dict: StateDict) -> torch.Tensor | None:
        model, _ = self.model_loader(self.config, setup=self.problem.model_setup)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.config.train.device)
        model.eval()

        data = self.problem.get_test_data()
        with torch.no_grad():
            lambda_maps = self.problem.compute_lambda_maps(self, data, model=model)
            lambda_maps_cpu = (
                None if lambda_maps is None else lambda_maps.detach().cpu()
            )

        del model, lambda_maps
        self._clear_cuda_cache()
        return lambda_maps_cpu
