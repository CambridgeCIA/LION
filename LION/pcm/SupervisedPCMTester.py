from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

from LION.config.LogConfig import (
    SAPMRenderConfig,
    ScalarImageRenderSpec,
    ScalarMapRenderConfig,
)
from LION.data_loaders.pcm.data_loader import PCMSyntheticPNGTorchDataset
from LION.experiments.pcm_experiments import make_test_cases
from LION.models.AdaptiveTGVPCMNetwork import AdaptiveTGVPCMNetwork
from LION.models.AdaptiveTVPCMNetwork import AdaptiveTVPCMNetwork
from LION.pcm.pcm_config import load_pcm_sapm_supervised_config
from LION.pcm.pcm_result_writer import PCMResultWriter
from LION.pcm.pcm_sapm import (
    forward_func_sapm_pcm,
    load_supervised_pcm_checkpoint_model,
    make_pcm_forward_data,
    resolve_supervised_pcm_checkpoint,
)
from LION.pcm.simulation import sample_and_simulate_pcm_acquisition
from LION.train.training_components import ModelInput
from LION.utils.artifact_contract import pcm_metrics_path, pcm_result_paths
from LION.utils.metrics import compute_metrics
from LION.utils.reproduce import seed_from_parts


@dataclass(kw_only=True)
class SupervisedPCMTestConfig:
    """Arguments for supervised PCM checkpoint testing/rendering."""

    root: Path
    experiment: str
    epochs: list[str]

    sample_ids: list[int] | None = None
    categories: list[str] | None = None
    sampling_ratios: list[float] | None = None
    coarse_j_values: list[int] | None = None

    save_reconstruction_tensor: bool = False
    save_lambda_maps_tensor: bool = False
    render_reconstruction: bool = True
    render_lambda_maps: bool = True


class SupervisedPCMTester:
    """Evaluate supervised TV/TGV PCM checkpoints and write visual results."""

    def __init__(self, args: SupervisedPCMTestConfig, device: torch.device) -> None:
        self.args = args
        self.device = device

        self.run_dir = args.root / args.experiment
        self.config = load_pcm_sapm_supervised_config(self.run_dir / "config.yaml")

        self.cases = make_test_cases(self.config)  # type: ignore[arg-type]
        self.dataset = PCMSyntheticPNGTorchDataset(self.config.data)

        self.categories = (
            args.categories
            if args.categories is not None
            else list(self.config.data.categories or [])
        )

        if args.sample_ids is None:
            assert self.config.data.samples is not None
            assert self.config.data.samples.ids is not None
            test_range = self.config.data.test_perm_range
            self.sample_ids = self.config.data.samples.ids[
                test_range[0] - 1 : test_range[1]
            ]
        else:
            self.sample_ids = args.sample_ids

        self.selected_cases = [
            (key, sampling_ratio, coarse_j)
            for key, (sampling_ratio, coarse_j) in enumerate(self.cases)
            if (args.sampling_ratios is None or sampling_ratio in args.sampling_ratios)
            and (args.coarse_j_values is None or coarse_j in args.coarse_j_values)
        ]

        if not self.selected_cases:
            raise ValueError("No sampling cases match the requested filters.")

    def resolve_checkpoint(self, spec: str) -> Path:
        """Resolve one checkpoint name/epoch under this run."""
        return resolve_supervised_pcm_checkpoint(self.run_dir, spec)

    def load_model(
        self,
        checkpoint_path: Path,
    ) -> tuple[AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork, int, int]:
        """Construct the model with the original keys and load one checkpoint."""
        return load_supervised_pcm_checkpoint_model(
            config=self.config,
            cases=self.cases,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

    def get_target(self, *, category: str, sample_id: int) -> torch.Tensor:
        """Load one target directly by category/sample ID."""
        path = self.dataset.sample_png_path(category, sample_id)
        return self.dataset.make_item(path)

    @torch.no_grad()
    def reconstruct(
        self,
        model: AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork,
        *,
        key: int,
        category: str,
        sample_id: int,
        sampling_ratio: float,
        coarse_j: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ModelInput]:
        """Simulate one PCM measurement and reconstruct it."""
        target = self.get_target(category=category, sample_id=sample_id)
        target_bchw = target.unsqueeze(0).unsqueeze(0).to(self.device)

        seed = seed_from_parts(
            str(self.config.reproduce.base_seed),
            category,
            str(sample_id),
            str(sampling_ratio),
            str(coarse_j),
        )
        rng = np.random.default_rng(seed)

        j_order = self.config.data.j_order
        assert j_order is not None

        acquisition = sample_and_simulate_pcm_acquisition(
            target_bchw,
            j_order=j_order,
            sampling=self.config.sampling,
            sampling_ratio=sampling_ratio,
            coarse_j=coarse_j,
            noise=self.config.noise,
            rng=rng,
            device=self.device,
            seed=seed,
        )
        data = make_pcm_forward_data(
            config=self.config,
            op=acquisition.op,
            measurement=acquisition.measurement,
            key=key,
            sampling_ratio=sampling_ratio,
            coarse_j=coarse_j,
            device=self.device,
        )
        result = forward_func_sapm_pcm(
            model=model,
            data=data,
            config=self.config,
            skip_grad_checkpoint=True,
        )

        assert result.extra_output is not None
        return (
            target_bchw.detach().cpu(),
            result.reconstruction.detach().cpu(),
            result.extra_output.detach().cpu(),
            data.model_input,
        )

    def result_output_config(self):
        """Apply the test CLI rendering switches to the saved result-output config."""
        output = dataclasses.replace(
            self.config.log.final_result,
            render_reconstruction=self.args.render_reconstruction,
            save_lambda_maps_tensor=self.args.save_lambda_maps_tensor,
        )

        if self.args.render_lambda_maps and output.sapm is None:
            n_maps = 1 if self.config.model.pdhg.regularisation == "TV" else 2
            output = dataclasses.replace(
                output,
                sapm=SAPMRenderConfig(
                    map_indices=list(range(n_maps)),
                    map_names=(
                        ["lambda_1_v"] if n_maps == 1 else ["lambda_1_v", "lambda_0_w"]
                    ),
                    lambda_map_render_configs=[
                        ScalarMapRenderConfig(
                            linear=ScalarImageRenderSpec(cmap="magma")
                        )
                        for _ in range(n_maps)
                    ],
                ),
            )
        elif not self.args.render_lambda_maps:
            output = dataclasses.replace(output, sapm=None)

        return output

    def append_metrics(
        self,
        *,
        epoch: int,
        step: int,
        category: str,
        sample_id: int,
        sampling_ratio: float,
        coarse_j: int,
        metrics: dict[str, float],
    ) -> None:
        """Append one sample result to the condition-level CSV."""
        path = pcm_metrics_path(
            self.run_dir, sampling_ratio=sampling_ratio, coarse_j=coarse_j
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()

        fieldnames = [
            "epoch",
            "step",
            "category",
            "sample_id",
            "sampling_rate",
            "coarse_j",
            "mse",
            "psnr",
            "ssim",
            "haarpsi",
            "pearson_corr",
        ]

        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "epoch": epoch,
                    "step": step,
                    "category": category,
                    "sample_id": sample_id,
                    "sampling_rate": sampling_ratio,
                    "coarse_j": coarse_j,
                    **metrics,
                }
            )

    def evaluate_checkpoint(self, checkpoint_path: Path) -> None:
        """Evaluate and render one checkpoint."""
        model, epoch, step = self.load_model(checkpoint_path)
        writer = PCMResultWriter(self.result_output_config())

        for key, sampling_ratio, coarse_j in self.selected_cases:
            for category in self.categories:
                for sample_id in tqdm(
                    self.sample_ids,
                    desc=f"epoch={epoch} step={step} {category} s={sampling_ratio:.2f} cJ={coarse_j}",
                    dynamic_ncols=True,
                ):
                    target, recon, lambda_maps, model_input = self.reconstruct(
                        model,
                        key=key,
                        category=category,
                        sample_id=sample_id,
                        sampling_ratio=sampling_ratio,
                        coarse_j=coarse_j,
                    )

                    metrics = compute_metrics(
                        recon=recon,
                        target=target,
                        data_range=float((target.max() - target.min()).item()),
                    )
                    self.append_metrics(
                        epoch=epoch,
                        step=step,
                        category=category,
                        sample_id=sample_id,
                        sampling_ratio=sampling_ratio,
                        coarse_j=coarse_j,
                        metrics=metrics,
                    )

                    paths = pcm_result_paths(
                        self.run_dir,
                        sampling_ratio=sampling_ratio,
                        coarse_j=coarse_j,
                        epoch=epoch,
                        step=step,
                        category=category,
                        sample_id=sample_id,
                    )
                    writer.write(
                        paths,
                        recon=recon,
                        target=target,
                        lambdas=lambda_maps,
                        model_input=model_input,
                        save_reconstruction_tensor=self.args.save_reconstruction_tensor,
                    )

    def run(self) -> None:
        for spec in self.args.epochs:
            self.evaluate_checkpoint(self.resolve_checkpoint(spec))


def run_supervised_pcm_test_cli() -> None:
    args = tyro.cli(SupervisedPCMTestConfig, description="Supervised SAPM PCM Testing")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    SupervisedPCMTester(args, device).run()
