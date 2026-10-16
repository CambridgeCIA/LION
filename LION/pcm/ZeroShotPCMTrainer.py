"""Zero-shot SAPM training for PCM reconstruction."""

from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
from tqdm import tqdm

from LION.config.ExperimentConfig import (
    ExperimentConfigLoader,
    load_experiment_config_cli,
)
from LION.config.RuntimeDirs import get_run_dirs
from LION.data_loaders.pcm.data_loader import PCMSyntheticPNGTorchDataset
from LION.models.AdaptiveTGVPCMNetwork import AdaptiveTGVPCMNetwork
from LION.models.AdaptiveTVPCMNetwork import AdaptiveTVPCMNetwork
from LION.models.model_loader import ModelLoader, ModelSetup
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.pcm_config import (
    SAPMZeroShotPCMConfig,
    ZeroShotPCMMethodParameter,
    load_pcm_sapm_zero_shot_config_from_path,
)
from LION.pcm.pcm_result_writer import PCMResultWriter
from LION.pcm.pcm_sapm import (
    PCMForwardData,
    forward_func_sapm_pcm,
    load_sapm_pcm_model,
    make_pcm_forward_data,
)
from LION.pcm.pcm_ssdu import PCMSSDUSplit, make_pcm_ssdu_split
from LION.pcm.reconstruct import AdaptiveMapReconOutput
from LION.pcm.simulation import PCMAcquisition, sample_and_simulate_pcm_acquisition
from LION.pcm_tools.pcm_sampling import make_sampling_cases
from LION.train.sapm import make_param_groups_sapm_common
from LION.train.self_supervised_train import (
    SelfSupervisedImageLogger,
    SelfSupervisedProblem,
    SelfSupervisedTrainer,
)
from LION.train.Trainer import run_trainer_with_mlflow, save_typed_config
from LION.train.training_components import (
    ForwardResult,
    ForwardRunner,
    ParameterGroupBuilder,
)
from LION.utils.artifact_contract import PCM_SELF_SUPERVISED_CONTRACT
from LION.utils.metrics import compute_metrics_2d
from LION.utils.reproduce import seed_from_parts


class ZeroShotPCMProblem(
    SelfSupervisedProblem[
        SAPMZeroShotPCMConfig,
        PCMForwardData,
        PCMSSDUSplit,
        SelfSupervisedTrainer,
        "ZeroShotPCMLogger",
    ]
):
    """PCM adapter for the shared single-sample zero-shot trainer."""

    def __init__(
        self,
        *,
        config: SAPMZeroShotPCMConfig,
        category: str,
        sample_id: int,
        key: int,
        sampling_ratio: float,
        coarse_j: int,
        target_image: torch.Tensor,
        forward_runner: ForwardRunner[
            SAPMZeroShotPCMConfig,
            AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork,
            PCMForwardData,
        ],
        acquisition: PCMAcquisition | None = None,
    ) -> None:
        super().__init__(config, sample_id)
        self.category = category
        self.key = key
        self.sampling_ratio = sampling_ratio
        self.coarse_j = coarse_j
        self.input_target_image = target_image
        self.forward_runner = forward_runner
        self.acquisition = acquisition

        n_maps = 1 if config.model.pdhg.regularisation == "TV" else 2
        n = config.data.n
        self.model_setup = ModelSetup(keys=(key,), map_shape=(n_maps, n, n))

    @property
    def condition_name(self) -> str:
        return f"{self.category}_s{self.sampling_ratio:.2f}_cJ{self.coarse_j}"

    def setup(self, trainer: SelfSupervisedTrainer) -> None:
        assert self.config.uses_normalized_op
        device = self.config.train.device
        self.target_image = self.input_target_image.unsqueeze(0).unsqueeze(0).to(device)

        j_order = self.config.data.j_order
        assert j_order is not None

        if self.acquisition is None:
            sampling_seed = seed_from_parts(
                "pcm_full_sampling",
                str(self.config.reproduce.base_seed),
                self.category,
                str(self.sample_id),
                f"{self.sampling_ratio:.12g}",
                str(self.coarse_j),
            )
            sampling_rng = np.random.default_rng(sampling_seed)
            acquisition = sample_and_simulate_pcm_acquisition(
                self.target_image,
                j_order=j_order,
                sampling=self.config.sampling,
                sampling_ratio=self.sampling_ratio,
                coarse_j=self.coarse_j,
                noise=self.config.noise,
                rng=sampling_rng,
                device=device,
                seed=sampling_seed,
            )
            full_indices = acquisition.sampled_indices
            full_op = acquisition.op
            full_measurement = acquisition.measurement
        else:
            full_indices = np.asarray(self.acquisition.sampled_indices)
            full_op = self.acquisition.op
            full_measurement = self.acquisition.measurement

        full_zero_filled = full_op.pseudo_inv(full_measurement)

        self.full_indices = full_indices
        self.full_measurement = full_measurement
        self.full_zero_filled = full_zero_filled.detach().cpu()

        self.full_data = self._make_forward_data(
            op=full_op,
            measurement=full_measurement,
        )

        self.val_split = self._make_validation_split()

        if self.val_split is None:
            self.train_source_indices = self.full_indices
            self.train_source_measurement = self.full_measurement
        else:
            self.train_source_indices = self.val_split.input_indices
            self.train_source_measurement = self.val_split.input_measurement

        train_seed = seed_from_parts(
            "pcm_train_split",
            *self._partition_seed_parts(),
            str(self.config.data.base_train_partition_seed),
        )
        self.train_split_rng = np.random.default_rng(train_seed)

    def _partition_seed_parts(self) -> tuple[str, ...]:
        parts = (
            self.category,
            str(self.sample_id),
            f"{self.sampling_ratio:.12g}",
            str(self.coarse_j),
        )
        if self.acquisition is not None and self.acquisition.seed is not None:
            return (*parts, str(self.acquisition.seed))
        return parts

    def _make_forward_data(
        self, *, op: PhotocurrentMapOp, measurement: torch.Tensor
    ) -> PCMForwardData:
        return make_pcm_forward_data(
            config=self.config,
            op=op,
            measurement=measurement,
            key=self.key,
            sampling_ratio=self.sampling_ratio,
            coarse_j=self.coarse_j,
            device=self.config.train.device,
        )

    def _make_split(
        self,
        *,
        source_indices: np.ndarray,
        source_measurement: torch.Tensor,
        p_target: float,
        rng: np.random.Generator,
    ) -> PCMSSDUSplit:
        return make_pcm_ssdu_split(
            config=self.config,
            source_indices=source_indices,
            source_measurement=source_measurement,
            p_target=p_target,
            rng=rng,
            key=self.key,
            sampling_ratio=self.sampling_ratio,
            coarse_j=self.coarse_j,
            device=self.config.train.device,
        )

    def _make_validation_split(self) -> PCMSSDUSplit | None:
        p_target_val = self.config.data.p_target_val
        if p_target_val is None or p_target_val <= 0:
            return None

        val_seed = seed_from_parts(
            "pcm_val_split",
            *self._partition_seed_parts(),
            str(self.config.data.base_val_partition_seed),
        )
        return self._make_split(
            source_indices=self.full_indices,
            source_measurement=self.full_measurement,
            p_target=p_target_val,
            rng=np.random.default_rng(val_seed),
        )

    def make_logger(self, trainer: SelfSupervisedTrainer) -> ZeroShotPCMLogger:
        assert self.target_image is not None
        return ZeroShotPCMLogger(
            trainer=trainer, problem=self, target_image=self.target_image.detach().cpu()
        )

    def has_validation(self) -> bool:
        return self.val_split is not None

    def make_train_split(self) -> PCMSSDUSplit:
        return self._make_split(
            source_indices=self.train_source_indices,
            source_measurement=self.train_source_measurement,
            p_target=self.config.data.p_target_train,
            rng=self.train_split_rng,
        )

    def get_val_split(self) -> PCMSSDUSplit:
        assert self.val_split is not None
        return self.val_split

    def get_test_data(self) -> PCMForwardData:
        return self.full_data

    @staticmethod
    def _as_forward_data(data: PCMForwardData | PCMSSDUSplit) -> PCMForwardData:
        if isinstance(data, PCMSSDUSplit):
            return data.forward_data
        return data

    def run_forward(
        self,
        trainer: SelfSupervisedTrainer,
        data: PCMForwardData | PCMSSDUSplit,
        *,
        model: torch.nn.Module | None = None,
        skip_grad_checkpoint: bool,
    ) -> ForwardResult:
        active_model = trainer.model if model is None else model
        return self.forward_runner(
            model=active_model,
            data=self._as_forward_data(data),
            config=self.config,
            skip_grad_checkpoint=skip_grad_checkpoint,
        )

    def compute_lambda_maps(
        self,
        trainer: SelfSupervisedTrainer,
        data: PCMForwardData | PCMSSDUSplit,
        *,
        model: torch.nn.Module | None = None,
    ) -> torch.Tensor | None:
        active_model = trainer.model if model is None else model
        forward_data = self._as_forward_data(data)
        assert isinstance(active_model, (AdaptiveTVPCMNetwork, AdaptiveTGVPCMNetwork))
        return active_model.estimate_lambda_map(
            forward_data.model_input.input_image, forward_data.key
        )

    def compute_target_loss(
        self, recon: torch.Tensor, split: PCMSSDUSplit
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(split.target_op(recon), split.target)

    def postprocess_recon(self, recon: torch.Tensor) -> torch.Tensor:
        return recon.detach().cpu()

    def train_epoch_desc(self) -> str:
        return (
            f"PCM SSDU {self.category} sample={self.sample_id} "
            f"s={self.sampling_ratio:.2f} cJ={self.coarse_j}"
        )


class ZeroShotPCMLogger(SelfSupervisedImageLogger[ZeroShotPCMProblem]):
    """PCM-specific metrics and final artifact writing."""

    def __init__(
        self,
        *,
        trainer: SelfSupervisedTrainer,
        problem: ZeroShotPCMProblem,
        target_image: torch.Tensor,
    ) -> None:
        super().__init__(
            trainer=trainer,
            checkpoint_label="",
            problem=problem,
            contract=PCM_SELF_SUPERVISED_CONTRACT,
            condition=problem.condition_name,
            target_image=target_image,
        )
        self.key = problem.key

    def initial_save(self) -> None:
        pass

    def compute_metrics(
        self, reconstruction: torch.Tensor
    ) -> tuple[float, float, float, float]:
        return compute_metrics_2d(
            recon=reconstruction,
            target=self.target_image,
            data_range=float(
                (self.target_image.max() - self.target_image.min()).item()
            ),
        )

    def initial_reconstruction(self) -> torch.Tensor:
        return self.problem.full_zero_filled

    def log_train_model_params(self) -> None:
        if not self.should_log("train", "params", "epoch"):
            return
        assert self.model.lambda_map_network is not None
        value = self.model.lambda_map_network.scale_factors_raw[str(self.key)]
        self.log_metric(
            "lambda_map_scale_factor_raw_epoch",
            float(value.item()),
            step=self.log_index("epoch"),
        )

    def save_result(self, result: ForwardResult, name: str) -> None:
        PCMResultWriter(self.config.log.final_result).write(
            self.config.dirs.result_paths(name),
            recon=result.reconstruction,
            target=self.target_image,
            lambdas=result.extra_output,
            model_input=self.problem.full_data.model_input,
            save_reconstruction_tensor=True,
        )


def make_pcm_zero_shot_trainer(
    *,
    config: SAPMZeroShotPCMConfig,
    category: str,
    sample_id: int,
    key: int,
    sampling_ratio: float,
    coarse_j: int,
    target_image: torch.Tensor,
    forward_runner: ForwardRunner[
        SAPMZeroShotPCMConfig,
        AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork,
        PCMForwardData,
    ],
    make_param_groups: ParameterGroupBuilder,
    load_model: ModelLoader[
        SAPMZeroShotPCMConfig, AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork
    ],
    acquisition: PCMAcquisition | None = None,
) -> SelfSupervisedTrainer:
    """Construct the shared PCM zero-shot problem and trainer."""
    problem = ZeroShotPCMProblem(
        config=config,
        category=category,
        sample_id=sample_id,
        key=key,
        sampling_ratio=sampling_ratio,
        coarse_j=coarse_j,
        target_image=target_image,
        forward_runner=forward_runner,
        acquisition=acquisition,
    )
    return SelfSupervisedTrainer(
        config, make_param_groups, load_model, problem=problem, epoch_size=1
    )


def make_pcm_zero_shot_method_config(
    parameter: ZeroShotPCMMethodParameter,
    *,
    device: torch.device,
    j_order: int,
    category: str,
    sample_id: int,
    rng_seed: int,
    output_dir: Path,
    uses_normalized_op: bool,
) -> SAPMZeroShotPCMConfig:
    """Resolve one grid-case zero-shot config from the shared YAML config."""
    config = load_pcm_sapm_zero_shot_config_from_path(parameter.config_path)

    case_seed = seed_from_parts(
        "pcm_zero_shot_case", str(config.reproduce.base_seed), str(rng_seed)
    )
    reproduce = dataclasses.replace(
        config.reproduce,
        base_seed=case_seed,
        random_seed=seed_from_parts("pcm_zero_shot_random", str(case_seed)),
        torch_seed=seed_from_parts("pcm_zero_shot_torch", str(case_seed)),
        numpy_seed=seed_from_parts("pcm_zero_shot_numpy", str(case_seed)),
        rng_seed=seed_from_parts("pcm_zero_shot_rng", str(case_seed)),
    )
    data = dataclasses.replace(
        config.data, j_order=j_order, this_sample_id=sample_id, this_category=category
    )
    train = dataclasses.replace(config.train, device=device.type)

    checkpoint = dataclasses.replace(
        config.log.ckpt,
        every_n_epochs=-1,
        epochs=None,
        every_n_steps=-1,
        steps=None,
        log_best=False,
        log_cur=False,
        log_final=False,
        local=False,
        mlflow=False,
    )
    log = dataclasses.replace(
        config.log,
        results_dir=output_dir,
        log_mlflow=False,
        log_csv=False,
        log_train_loss_every_epochs=-1,
        log_train_metrics_every_epochs=-1,
        log_train_params_every_epochs=-1,
        log_val_metrics_every_epochs=-1,
        log_test_loss_every_epochs=-1,
        log_test_metrics_every_epochs=-1,
        ckpt=checkpoint,
    )
    dirs = get_run_dirs(output_dir, "training", mkdir=True)

    return dataclasses.replace(
        config,
        reproduce=reproduce,
        data=data,
        train=train,
        log=log,
        dirs=dirs,
        uses_normalized_op=uses_normalized_op,
    )


@dataclasses.dataclass
class ZeroShotPCMReconFn:
    """Run the shared PCM zero-shot trainer on one prepared grid acquisition."""

    parameter: ZeroShotPCMMethodParameter
    sampled_indices: np.ndarray
    sampling_ratio: float
    coarse_j: int
    rng_seed: int
    j_order: int
    device: torch.device
    uses_normalized_op: bool
    target_image: torch.Tensor
    category: str
    sample_id: int
    output_dir: Path

    def __call__(self, recon_input) -> AdaptiveMapReconOutput:
        """Train zero-shot PCM reconstruction and return the selected full-data result."""
        config = make_pcm_zero_shot_method_config(
            self.parameter,
            device=self.device,
            j_order=self.j_order,
            category=self.category,
            sample_id=self.sample_id,
            rng_seed=self.rng_seed,
            output_dir=self.output_dir,
            uses_normalized_op=self.uses_normalized_op,
        )
        acquisition = PCMAcquisition(
            op=recon_input.op,
            measurement=recon_input.y,
            sampled_indices=self.sampled_indices,
            seed=self.rng_seed,
        )
        trainer = make_pcm_zero_shot_trainer(
            config=config,
            category=self.category,
            sample_id=self.sample_id,
            key=0,
            sampling_ratio=self.sampling_ratio,
            coarse_j=self.coarse_j,
            target_image=self.target_image,
            forward_runner=forward_func_sapm_pcm,
            make_param_groups=make_param_groups_sapm_common,
            load_model=load_sapm_pcm_model,
            acquisition=acquisition,
        )
        trainer.setup()
        trainer.train()
        result = trainer.get_result(self.parameter.criterion)
        assert result.lambda_maps is not None
        output = AdaptiveMapReconOutput(
            reconstruction=result.reconstruction.to(self.device),
            lambda_maps=result.lambda_maps,
        )
        del result, trainer
        return output


def run_pcm_zero_shot_train_cli(
    *,
    config_cls: type[SAPMZeroShotPCMConfig],
    load_config: ExperimentConfigLoader[SAPMZeroShotPCMConfig],
    make_param_groups: ParameterGroupBuilder,
    load_model: ModelLoader[
        SAPMZeroShotPCMConfig, AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork
    ],
    forward_func: ForwardRunner[
        SAPMZeroShotPCMConfig,
        AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork,
        PCMForwardData,
    ],
) -> None:
    """Run zero-shot PCM SAPM training independently for each sample and sampling condition."""
    config = load_experiment_config_cli(config_cls, load_config)

    experiment_name = config.log.exp_code
    mlflow.set_experiment(experiment_name)

    base_run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{config.model.name}"
    if config.log.run_suffix is not None:
        base_run_name += f"_{config.log.run_suffix}"

    results_dir = config.log.results_dir / "pcm" / experiment_name
    dataset = PCMSyntheticPNGTorchDataset(config.data)
    j_order = config.data.j_order
    assert j_order is not None
    cases = make_sampling_cases(j_order=j_order, sampling=config.sampling)

    for category in tqdm(dataset.categories, desc="Categories", dynamic_ncols=True):
        for sample_id in tqdm(
            dataset.sample_ids, desc="Samples", leave=False, dynamic_ncols=True
        ):
            target = dataset.make_item(dataset.sample_png_path(category, sample_id))

            for key, (sampling_ratio, coarse_j) in enumerate(cases):
                condition = f"{category}_s{sampling_ratio:.2f}_cJ{coarse_j}"
                sample_run_name = PCM_SELF_SUPERVISED_CONTRACT.format_run_dir_name(
                    condition=condition, sample_id=sample_id
                )
                run_name = f"{base_run_name}/{sample_run_name}"
                dirs = get_run_dirs(results_dir, run_name)

                sample_data_config = dataclasses.replace(
                    config.data, this_sample_id=sample_id, this_category=category
                )
                sample_config = dataclasses.replace(
                    config, data=sample_data_config, dirs=dirs
                )

                save_typed_config(sample_config)

                trainer = make_pcm_zero_shot_trainer(
                    config=sample_config,
                    category=category,
                    sample_id=sample_id,
                    key=key,
                    sampling_ratio=sampling_ratio,
                    coarse_j=coarse_j,
                    target_image=target,
                    forward_runner=forward_func,
                    make_param_groups=make_param_groups,
                    load_model=load_model,
                )

                run_trainer_with_mlflow(
                    trainer=trainer,
                    run_name=sample_config.dirs.run_name or sample_run_name,
                )

                del trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
