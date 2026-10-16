from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from mrpro.operators import LinearOperator

from LION.models.AdaptiveTGVPCMNetwork import AdaptiveTGVPCMNetwork
from LION.models.AdaptiveTVPCMNetwork import AdaptiveTVPCMNetwork
from LION.models.model_loader import ModelLoader, ModelSetup
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.pcm_config import (
    SAPMPCMConfig,
    SAPMSelfSupervisedPCMConfig,
    SAPMSupervisedPCMConfig,
    SelfSupervisedPCMMethodParameter,
    SupervisedPCMMethodParameter,
    TrainedPCMMethodParameter,
    load_pcm_sapm_self_supervised_config_from_path,
    load_pcm_sapm_supervised_config_from_path,
)
from LION.pcm.pcm_model_input import make_pcm_model_input
from LION.pcm.reconstruct import AdaptiveMapReconOutput
from LION.pcm_tools.pcm_sampling import make_sampling_cases
from LION.train.training_components import ForwardResult, ForwardRunner, ModelInput
from LION.utils.artifact_contract import candidate_checkpoint_stems
from LION.utils.types import StateDict


class PhotocurrentMapMRProOp(LinearOperator):
    """Expose a LION PhotocurrentMapOp through the MRpro LinearOperator interface."""

    def __init__(self, op: PhotocurrentMapOp) -> None:
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.op(x),)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.op.adjoint(y),)


@dataclass(frozen=True, kw_only=True)
class PCMForwardData:
    """Inputs for one PCM SAPM forward pass."""

    forward_op: PhotocurrentMapMRProOp
    measurement: torch.Tensor
    model_input: ModelInput
    key: int
    sampling_ratio: float
    coarse_j: int


def make_pcm_forward_data(
    *,
    config: SAPMPCMConfig,
    op: PhotocurrentMapOp,
    measurement: torch.Tensor,
    key: int,
    sampling_ratio: float,
    coarse_j: int,
    device: torch.device | str,
) -> PCMForwardData:
    """Create PCM SAPM forward data from an already-prepared acquisition."""
    zero_filled = op.pseudo_inv(measurement)
    return PCMForwardData(
        forward_op=PhotocurrentMapMRProOp(op),
        measurement=measurement,
        model_input=make_pcm_model_input(
            config=config,
            op=op,
            measurement=measurement,
            zero_filled=zero_filled,
            device=device,
        ),
        key=key,
        sampling_ratio=sampling_ratio,
        coarse_j=coarse_j,
    )


class SAPMPCMModelLoader(
    ModelLoader[SAPMPCMConfig, AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork]
):
    def __call__(
        self, config: SAPMPCMConfig, setup: ModelSetup | None = None
    ) -> tuple[AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork, StateDict | None]:
        assert setup is not None
        assert setup.keys is not None

        lambda_map_network = self.make_sapm_parameter_map_network(
            config.model,
            keys=setup.keys,
            map_shape=setup.map_shape,
            expected_map_shape_ndim=3,
        )

        if config.model.pdhg.regularisation == "TV":
            model = AdaptiveTVPCMNetwork(
                config.model,
                lambda_map_network,
                setup.forward_op,
                setup.example_image,
            )
        elif config.model.pdhg.regularisation == "TGV":
            model = AdaptiveTGVPCMNetwork(
                config.model,
                lambda_map_network,
                setup.forward_op,
                setup.example_image,
            )
        else:
            raise ValueError(
                f"Unsupported regularisation: {config.model.pdhg.regularisation}"
            )

        config.model.pdhg.primal_stepsize = model.primal_stepsize
        config.model.pdhg.dual_stepsize = model.dual_stepsize
        model, optimizer_state_dict = self.load_pretrained(model, config.pretrained)
        return model, optimizer_state_dict


class SAPMPCMForwardRunner(
    ForwardRunner[
        SAPMPCMConfig, AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork, PCMForwardData
    ]
):
    def __call__(
        self,
        model: AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork,
        data: PCMForwardData,
        config: SAPMPCMConfig,
        *,
        skip_grad_checkpoint: bool,
    ) -> ForwardResult:
        lambda_map = model.estimate_lambda_map(data.model_input.input_image, data.key)

        reconstruction = model(
            input_image=None,
            initial_image=data.model_input.initial_image,
            measurement=data.measurement,
            key=data.key,
            forward_op=data.forward_op,
            lambda_reg=lambda_map,
            use_checkpoint=(config.model.use_checkpoint and not skip_grad_checkpoint),
            checkpoint_chunk_size=config.model.checkpoint_chunk_size,
            checkpoint_use_reentrant=config.model.checkpoint_use_reentrant,
        )

        return ForwardResult(reconstruction=reconstruction, extra_output=lambda_map)


load_sapm_pcm_model = SAPMPCMModelLoader()
forward_func_sapm_pcm = SAPMPCMForwardRunner()


def resolve_trained_pcm_checkpoint(run_dir: Path, spec: str) -> Path:
    """Resolve a trained PCM checkpoint name inside one training run."""
    checkpoint_dir = run_dir / "checkpoints"

    direct = checkpoint_dir / (spec if spec.endswith(".pt") else f"{spec}.pt")
    if direct.exists():
        return direct

    candidates = candidate_checkpoint_stems(Path(spec).stem)
    matches = [path for path in checkpoint_dir.glob("*.pt") if path.stem in candidates]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely resolve checkpoint {spec!r} in {checkpoint_dir}. Matches: {matches}"
        )
    return matches[0]


def load_trained_pcm_checkpoint_model(
    *,
    config: SAPMSupervisedPCMConfig | SAPMSelfSupervisedPCMConfig,
    cases: list[tuple[float, int]],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork, int, int]:
    """Construct a trained PCM model and load one training checkpoint."""
    pretrained = replace(
        config.pretrained,
        pretrained_id=checkpoint_path.stem,
        pretrained_path=checkpoint_path,
        is_checkpoint=True,
        load_optimizer_state=False,
    )
    load_config = replace(config, pretrained=pretrained)

    n_maps = 1 if load_config.model.pdhg.regularisation == "TV" else 2
    n = load_config.data.n
    model, _ = load_sapm_pcm_model(
        load_config,
        ModelSetup(
            keys=tuple(range(len(cases))),
            map_shape=(n_maps, n, n),
        ),
    )
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return model, int(checkpoint["epoch"]), int(checkpoint["global_step"])


def resolve_supervised_pcm_checkpoint(run_dir: Path, spec: str) -> Path:
    """Resolve a supervised PCM checkpoint name inside one training run."""
    return resolve_trained_pcm_checkpoint(run_dir, spec)


def load_supervised_pcm_checkpoint_model(
    *,
    config: SAPMSupervisedPCMConfig,
    cases: list[tuple[float, int]],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[AdaptiveTVPCMNetwork | AdaptiveTGVPCMNetwork, int, int]:
    """Construct a supervised PCM model and load one training checkpoint."""
    return load_trained_pcm_checkpoint_model(
        config=config,
        cases=cases,
        checkpoint_path=checkpoint_path,
        device=device,
    )


@dataclass
class TrainedPCMReconFn:
    """Run a trained supervised or dataset self-supervised PCM SAPM model."""

    parameter: TrainedPCMMethodParameter
    sampling_ratio: float
    coarse_j: int
    device: torch.device

    def __post_init__(self) -> None:
        if isinstance(self.parameter, SupervisedPCMMethodParameter):
            self.config = load_pcm_sapm_supervised_config_from_path(
                self.parameter.run_dir / "config.yaml"
            )
        elif isinstance(self.parameter, SelfSupervisedPCMMethodParameter):
            self.config = load_pcm_sapm_self_supervised_config_from_path(
                self.parameter.run_dir / "config.yaml"
            )
        else:
            raise TypeError(
                f"Unsupported trained PCM method parameter: {type(self.parameter)}"
            )

        j_order = self.config.data.j_order
        assert j_order is not None
        self.cases = make_sampling_cases(j_order=j_order, sampling=self.config.sampling)
        scale_factor_case = self.parameter.scale_factor_case or (
            self.sampling_ratio,
            self.coarse_j,
        )
        self.key = self.cases.index(scale_factor_case)
        checkpoint_path = resolve_trained_pcm_checkpoint(
            self.parameter.run_dir, self.parameter.checkpoint
        )
        self.model, self.epoch, self.step = load_trained_pcm_checkpoint_model(
            config=self.config,
            cases=self.cases,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

    @torch.no_grad()
    def __call__(self, recon_input) -> AdaptiveMapReconOutput:
        """Reconstruct the supplied acquisition with the trained model."""
        data = make_pcm_forward_data(
            config=self.config,
            op=recon_input.op,
            measurement=recon_input.y,
            key=self.key,
            sampling_ratio=self.sampling_ratio,
            coarse_j=self.coarse_j,
            device=self.device,
        )
        result = forward_func_sapm_pcm(
            model=self.model,
            data=data,
            config=self.config,
            skip_grad_checkpoint=True,
        )
        return AdaptiveMapReconOutput(
            reconstruction=result.reconstruction, lambda_maps=result.lambda_maps
        )


SupervisedPCMReconFn = TrainedPCMReconFn
