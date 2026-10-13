"""Supervised SAPM training for PCM reconstruction."""

from __future__ import annotations

import dataclasses
import time
from datetime import datetime

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from LION.config.ExperimentConfig import (
    ExperimentConfigLoader,
    load_experiment_config_cli,
)
from LION.config.RuntimeDirs import get_run_dirs
from LION.data_loaders.pcm.data_loader import PCMSyntheticPNGTorchDataset
from LION.data_loaders.pcm.data_primitives import SamplesParameter
from LION.models.AdaptiveTGVPCMNetwork import AdaptiveTGVPCMNetwork
from LION.models.AdaptiveTVPCMNetwork import AdaptiveTVPCMNetwork
from LION.models.model_loader import ModelLoader, ModelSetup
from LION.pcm.pcm_config import SAPMSupervisedPCMConfig
from LION.pcm.pcm_sapm import PCMForwardData, make_pcm_forward_data
from LION.pcm.simulation import PCMAcquisition, sample_and_simulate_pcm_acquisition
from LION.pcm_tools.pcm_sampling import make_sampling_cases
from LION.train.Logger import LogBy, Logger
from LION.train.Trainer import Trainer, run_trainer_with_mlflow, save_typed_config
from LION.train.training_components import ForwardRunner, ParameterGroupBuilder
from LION.utils.metrics import compute_metrics
from LION.utils.reproduce import seed_from_parts


class SupervisedPCMTrainer(Trainer[SAPMSupervisedPCMConfig]):
    """Supervised trainer for unrolled TV/TGV PCM reconstruction."""

    def __init__(
        self,
        config: SAPMSupervisedPCMConfig,
        parameter_group_builder: ParameterGroupBuilder,
        model_loader: ModelLoader[
            SAPMSupervisedPCMConfig, AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork
        ],
        forward_runner: ForwardRunner[
            SAPMSupervisedPCMConfig,
            AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork,
            PCMForwardData,
        ],
        loss_function: torch.nn.Module,
    ) -> None:
        super().__init__(config, parameter_group_builder, model_loader)
        self.forward_runner = forward_runner
        self.loss_function = loss_function

    def setup(self) -> None:
        assert self.config.uses_normalized_op
        j_order = self.config.data.j_order
        assert j_order is not None
        self.cases = make_sampling_cases(j_order=j_order, sampling=self.config.sampling)

        assert self.config.data.samples is not None
        assert self.config.data.samples.ids is not None
        sample_ids = self.config.data.samples.ids

        train_range = self.config.data.train_perm_range
        val_range = self.config.data.val_perm_range
        test_range = self.config.data.test_perm_range

        train_ids = sample_ids[train_range[0] - 1 : train_range[1]]
        val_ids = sample_ids[val_range[0] - 1 : val_range[1]]
        test_ids = sample_ids[test_range[0] - 1 : test_range[1]]

        self.train_dataset = PCMSyntheticPNGTorchDataset(
            dataclasses.replace(
                self.config.data, samples=SamplesParameter(ids=train_ids)
            )
        )
        self.val_dataset = PCMSyntheticPNGTorchDataset(
            dataclasses.replace(self.config.data, samples=SamplesParameter(ids=val_ids))
        )
        self.test_dataset = PCMSyntheticPNGTorchDataset(
            dataclasses.replace(
                self.config.data, samples=SamplesParameter(ids=test_ids)
            )
        )

        batch_size = self.config.train.batch_size
        steps_per_epoch = (len(self.train_dataset) + batch_size - 1) // batch_size

        n_maps = 1 if self.config.model.pdhg.regularisation == "TV" else 2
        n = self.config.data.n
        super()._setup(
            epoch_size=steps_per_epoch,
            model_setup=ModelSetup(
                keys=tuple(range(len(self.cases))), map_shape=(n_maps, n, n)
            ),
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, generator=self.rng
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )

        self.logger = SupervisedPCMLogger(
            self.config,
            self.model,
            self.optimizer,
            self.rng,
            epoch_size=len(self.train_dataset),
            cases=self.cases,
            training_state=self.state,
        )

    def make_acquisition(
        self,
        target_image: torch.Tensor,
        *,
        key: int,
        rng: np.random.Generator,
        device: torch.device | str,
    ) -> PCMAcquisition:
        sampling_ratio, coarse_j = self.cases[key]
        j_order = self.config.data.j_order
        assert j_order is not None

        target_image = target_image.unsqueeze(1).to(device)  # Add channel dimension
        return sample_and_simulate_pcm_acquisition(
            target_image,
            j_order=j_order,
            sampling=self.config.sampling,
            sampling_ratio=sampling_ratio,
            coarse_j=coarse_j,
            noise=self.config.noise,
            rng=rng,
            device=device,
        )

    def make_forward_data(
        self,
        target_image: torch.Tensor,
        *,
        key: int,
        rng: np.random.Generator,
        device: torch.device | str,
    ) -> PCMForwardData:
        sampling_ratio, coarse_j = self.cases[key]
        acquisition = self.make_acquisition(
            target_image, key=key, rng=rng, device=device
        )
        return make_pcm_forward_data(
            config=self.config,
            op=acquisition.op,
            measurement=acquisition.measurement,
            key=key,
            sampling_ratio=sampling_ratio,
            coarse_j=coarse_j,
            device=device,
        )

    def make_train_forward_data(
        self,
        target_image: torch.Tensor,
        *,
        key: int,
        rng: np.random.Generator,
        device: torch.device | str,
    ) -> PCMForwardData:
        return self.make_forward_data(target_image, key=key, rng=rng, device=device)

    def make_val_forward_data(
        self,
        target_image: torch.Tensor,
        *,
        key: int,
        rng: np.random.Generator,
        device: torch.device | str,
    ) -> PCMForwardData:
        return self.make_forward_data(target_image, key=key, rng=rng, device=device)

    def forward_step(self, data: PCMForwardData, *, skip_grad_checkpoint: bool):
        return self.forward_runner(
            model=self.model,
            data=data,
            config=self.config,
            skip_grad_checkpoint=skip_grad_checkpoint,
        )

    def forward_backward_train(
        self, data: PCMForwardData, *, target_image: torch.Tensor
    ) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        result = self.forward_step(data, skip_grad_checkpoint=False)
        return self.optimizer_step_from_loss(
            self.loss_function(result.reconstruction, target_image)
        )

    @torch.no_grad()
    def forward_val(
        self, data: PCMForwardData, *, target_image: torch.Tensor
    ) -> tuple[float, torch.Tensor]:
        result = self.forward_step(data, skip_grad_checkpoint=True)
        loss = self.loss_function(result.reconstruction, target_image)
        return loss.item(), result.reconstruction.detach().cpu()

    def validate_if_needed(self, log_by: LogBy) -> bool:
        self.logger.reset_val_logging(log_by)

        if not self.logger.is_validating(log_by):
            return False

        device = self.config.train.device
        self.model.eval()

        with torch.no_grad():
            for key in range(len(self.cases)):
                self.logger.reset_cur_val_metrics()
                val_rng = np.random.default_rng(self.config.reproduce.base_seed + key)

                for target_image in tqdm(
                    self.val_loader,
                    desc=f"Validation {self.logger.case_name(key)}",
                    dynamic_ncols=True,
                ):
                    data = self.make_val_forward_data(
                        target_image, key=key, rng=val_rng, device=device
                    )
                    target_bchw = target_image.unsqueeze(1)

                    t_start_forward = time.time()
                    loss_value, recon = self.forward_val(
                        data, target_image=target_bchw.to(device)
                    )

                    self.logger.update_cur_val_metrics(
                        loss_value=loss_value,
                        t_start_forward=t_start_forward,
                        target_image=target_bchw,
                        recon_image=recon,
                    )

                self.logger.update_best_val_metrics(key)

        self.model.train()
        return True

    def train(self) -> None:
        device = self.config.train.device

        self.logger.log_config_to_mlflow()

        self.validate_if_needed("epoch")
        self.logger.save_checkpoint_if_needed(log_by="epoch")

        if self.logger.is_checkpoint_step():
            self.logger.save_checkpoint_if_needed(log_by="step")

        for _ in self.train_epoch_iterator(desc="Training Epochs"):
            self.increment_epoch()

            epoch_loss_sum = 0.0
            self.logger.start_timer()

            for target_image in tqdm(
                self.train_loader, desc="Training Steps", dynamic_ncols=True
            ):
                self.increment_global_step()
                batch_size = target_image.shape[0]

                if len(self.cases) > 1:
                    key = int(
                        torch.randint(
                            0, len(self.cases), (1,), generator=self.rng
                        ).item()
                    )
                else:
                    key = 0

                with torch.no_grad():
                    t_start_sim = time.time()
                    sampling_seed = seed_from_parts(
                        "pcm_supervised_sampling",
                        str(self.config.reproduce.base_seed),
                        str(self.one_indexed_epoch),
                        str(self.one_indexed_step),
                        str(key),
                    )
                    data = self.make_train_forward_data(
                        target_image,
                        key=key,
                        rng=np.random.default_rng(sampling_seed),
                        device=device,
                    )
                    target_bchw = target_image.unsqueeze(1).to(device)
                    self.logger.update_epoch_sim_time(t_start_sim)

                t_start_forward_backward = time.time()
                loss_value = self.forward_backward_train(data, target_image=target_bchw)
                self.logger.update_epoch_forward_backward_time(t_start_forward_backward)

                epoch_loss_sum += loss_value * batch_size

                t_start_cleanup = time.time()
                del data, target_image, target_bchw
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.update_epoch_cleanup_time(t_start_cleanup)

                self.logger.log_train_step(loss_value)

                validated = self.validate_if_needed("step")
                if validated or self.logger.is_checkpoint_step():
                    self.logger.save_checkpoint_if_needed(log_by="step")

            self.logger.log_train_epoch(epoch_loss_sum)

            self.validate_if_needed("epoch")
            self.logger.save_checkpoint_if_needed(log_by="epoch")

    def finalize(self) -> None:
        self.logger.finalize()


def run_pcm_dataset_train_cli(
    config_cls: type[SAPMSupervisedPCMConfig],
    load_config: ExperimentConfigLoader[SAPMSupervisedPCMConfig],
    make_param_groups: ParameterGroupBuilder,
    load_model: ModelLoader[
        SAPMSupervisedPCMConfig, AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork
    ],
    forward_func: ForwardRunner[
        SAPMSupervisedPCMConfig,
        AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork,
        PCMForwardData,
    ],
    trainer_cls: type[SupervisedPCMTrainer],
) -> None:
    config = load_experiment_config_cli(config_cls, load_config)

    experiment_name = config.log.exp_code
    mlflow.set_experiment(experiment_name)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{config.model.name}"
    if config.log.run_suffix is not None:
        run_name += f"_{config.log.run_suffix}"

    results_dir = config.log.results_dir / "pcm" / experiment_name
    config = dataclasses.replace(config, dirs=get_run_dirs(results_dir, run_name))

    save_typed_config(config)

    trainer = trainer_cls(
        config, make_param_groups, load_model, forward_func, torch.nn.MSELoss()
    )
    run_trainer_with_mlflow(trainer=trainer, run_name=run_name)


def run_pcm_supervised_train_cli(
    config_cls: type[SAPMSupervisedPCMConfig],
    load_config: ExperimentConfigLoader[SAPMSupervisedPCMConfig],
    make_param_groups: ParameterGroupBuilder,
    load_model: ModelLoader[
        SAPMSupervisedPCMConfig, AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork
    ],
    forward_func: ForwardRunner[
        SAPMSupervisedPCMConfig,
        AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork,
        PCMForwardData,
    ],
) -> None:
    run_pcm_dataset_train_cli(
        config_cls=config_cls,
        load_config=load_config,
        make_param_groups=make_param_groups,
        load_model=load_model,
        forward_func=forward_func,
        trainer_cls=SupervisedPCMTrainer,
    )


class SupervisedPCMTimer:
    def start_timer(self) -> None:
        self.cur_epoch_start_time = time.time()
        self.cur_epoch_sim_time_sum = 0.0
        self.cur_epoch_forward_backward_train_time_sum = 0.0
        self.cur_epoch_cleanup_time_sum = 0.0

    def update_epoch_sim_time(self, sim_start_time: float) -> None:
        self.cur_epoch_sim_time_sum += time.time() - sim_start_time

    def update_epoch_forward_backward_time(self, forward_start_time: float) -> None:
        self.cur_epoch_forward_backward_train_time_sum += (
            time.time() - forward_start_time
        )

    def update_epoch_cleanup_time(self, cleanup_start_time: float) -> None:
        self.cur_epoch_cleanup_time_sum += time.time() - cleanup_start_time

    def log_epoch_time(self, epoch: int) -> None:
        end_time = time.time()
        self.log_metric(
            "epoch_total_time_seconds", end_time - self.cur_epoch_start_time, step=epoch
        )
        self.log_metric(
            "epoch_total_sim_time_seconds", self.cur_epoch_sim_time_sum, step=epoch
        )
        self.log_metric(
            "epoch_total_forward_backward_time_seconds",
            self.cur_epoch_forward_backward_train_time_sum,
            step=epoch,
        )
        self.log_metric(
            "epoch_total_cleanup_time_seconds",
            self.cur_epoch_cleanup_time_sum,
            step=epoch,
        )


class SupervisedPCMLogger(Logger[SAPMSupervisedPCMConfig], SupervisedPCMTimer):
    def __init__(
        self,
        config: SAPMSupervisedPCMConfig,
        model: AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork,
        optimizer: torch.optim.Optimizer,
        rng: torch.Generator,
        epoch_size: int,
        cases: list[tuple[float, int]],
        training_state,
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            optimizer=optimizer,
            rng=rng,
            training_state=training_state,
        )
        self.epoch_size = epoch_size
        self.cases = cases

        keys = range(len(cases))
        self.last_val_losses = {key: float("inf") for key in keys}
        self.best_val_losses = {key: float("inf") for key in keys}
        self.best_val_mse = {key: float("inf") for key in keys}
        self.best_val_psnr = {key: float("-inf") for key in keys}
        self.best_val_ssim = {key: float("-inf") for key in keys}
        self.best_val_haarpsi = {key: float("-inf") for key in keys}

    def case_name(self, key: int) -> str:
        sampling_ratio, coarse_j = self.cases[key]
        return f"s{sampling_ratio:.2f}_cJ{coarse_j}"

    def log_model_params_if_needed(self, log_by: LogBy) -> None:
        if not self.should_log("train", "params", log_by):
            return
        step = self.log_index(log_by)

        assert self.model.lambda_map_network is not None
        for key in range(len(self.cases)):
            value = self.model.lambda_map_network.scale_factors_raw[str(key)]
            self.log_metric(
                f"lambda_map_scale_factor_raw_{self.case_name(key)}_{log_by}",
                float(value.item()),
                step=step,
            )

    def log_train_step(self, loss_value: float) -> None:
        if self.should_log("train", "loss", "step"):
            self.log_metric("train_loss_step", loss_value, step=self.one_indexed_step)
        self.log_model_params_if_needed("step")

    def log_train_epoch(self, epoch_loss_sum: float) -> None:
        self.log_epoch_time(self.one_indexed_epoch)
        if self.should_log("train", "loss", "epoch"):
            self.log_metric(
                "train_loss_epoch",
                epoch_loss_sum / self.epoch_size,
                step=self.one_indexed_epoch,
            )
        self.log_model_params_if_needed("epoch")

    def reset_val_logging(self, log_by: LogBy) -> None:
        self.val_log_by = log_by
        self.val_mlflow_step = self.log_index(log_by)

    def reset_cur_val_metrics(self) -> None:
        self.val_loss_sum = 0.0
        self.val_count = 0
        self.val_mse_sum = 0.0
        self.val_psnr_sum = 0.0
        self.val_ssim_sum = 0.0
        self.val_haarpsi_sum = 0.0
        self.forward_val_time = 0.0

    def update_cur_val_metrics(
        self,
        *,
        loss_value: float,
        t_start_forward: float,
        target_image: torch.Tensor,
        recon_image: torch.Tensor,
    ) -> None:
        self.forward_val_time += time.time() - t_start_forward

        batch_size = target_image.shape[0]
        self.val_loss_sum += loss_value * batch_size
        self.val_count += batch_size

        if not self.should_log("val", "metrics", self.val_log_by):
            return

        for batch_idx in range(batch_size):
            target_sample = target_image[batch_idx : batch_idx + 1]
            metrics = compute_metrics(
                recon=recon_image[batch_idx : batch_idx + 1],
                target=target_sample,
                data_range=float((target_sample.max() - target_sample.min()).item()),
            )
            self.val_mse_sum += metrics["mse"]
            self.val_psnr_sum += metrics["psnr"]
            self.val_ssim_sum += metrics["ssim"]
            self.val_haarpsi_sum += metrics["haarpsi"]

    def update_best_val_metrics(self, key: int) -> None:
        val_loss_mean = self.val_loss_sum / self.val_count
        self.last_val_losses[key] = val_loss_mean
        name = self.case_name(key)

        self.best_val_losses[key], improved = self.update_best_value(
            val_loss_mean, self.best_val_losses[key], mode="min"
        )
        if improved:
            self.save_best_checkpoint_if_needed(
                is_best={name: True}, criterion="val_loss"
            )

        log_by = self.val_log_by
        step = self.val_mlflow_step
        if self.should_log("val", "loss", log_by):
            self.log_metric(f"val_{name}_loss_{log_by}", val_loss_mean, step=step)

        if self.should_log("val", "metrics", log_by):
            val_mse_mean = self.val_mse_sum / self.val_count
            val_psnr_mean = self.val_psnr_sum / self.val_count
            val_ssim_mean = self.val_ssim_sum / self.val_count
            val_haarpsi_mean = self.val_haarpsi_sum / self.val_count

            self.best_val_mse[key], _ = self.update_best_value(
                val_mse_mean, self.best_val_mse[key], mode="min"
            )
            self.best_val_psnr[key], _ = self.update_best_value(
                val_psnr_mean, self.best_val_psnr[key], mode="max"
            )
            self.best_val_ssim[key], _ = self.update_best_value(
                val_ssim_mean, self.best_val_ssim[key], mode="max"
            )
            self.best_val_haarpsi[key], _ = self.update_best_value(
                val_haarpsi_mean, self.best_val_haarpsi[key], mode="max"
            )

            self.log_metric(f"val_{name}_mse_{log_by}", val_mse_mean, step=step)
            self.log_metric(f"val_{name}_psnr_{log_by}", val_psnr_mean, step=step)
            self.log_metric(f"val_{name}_ssim_{log_by}", val_ssim_mean, step=step)
            self.log_metric(f"val_{name}_haarpsi_{log_by}", val_haarpsi_mean, step=step)

        self.log_metric(
            f"val_{name}_forward_val_time_seconds_{log_by}",
            self.forward_val_time,
            step=step,
        )

    def finalize(self) -> None:
        main_results = {}
        for key in range(len(self.cases)):
            name = self.case_name(key)
            main_results[f"last_val_loss_{name}"] = self.last_val_losses[key]
            main_results[f"best_val_loss_{name}"] = self.best_val_losses[key]
            main_results[f"best_val_mse_{name}"] = self.best_val_mse[key]
            main_results[f"best_val_psnr_{name}"] = self.best_val_psnr[key]
            main_results[f"best_val_ssim_{name}"] = self.best_val_ssim[key]
            main_results[f"best_val_haarpsi_{name}"] = self.best_val_haarpsi[key]
        self.save_main_results(main_results)
