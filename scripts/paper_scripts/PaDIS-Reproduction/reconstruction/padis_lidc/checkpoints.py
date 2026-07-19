"""Checkpoint and model loading for PaDIS LIDC reconstruction."""

from __future__ import annotations

import copy
import os
import pathlib
import warnings

import torch

from LION.CTtools.ct_geometry import Geometry
from LION.models.CNNs.drunet import DRUNet
from LION.models.LIONmodel import LIONModelParameter
from LION.models.diffusion import NCSNpp
from LION.utils.parameter import LIONParameter

from padis_lidc.experiments import DEFAULT_CHECKPOINT, LION_EXPERIMENTS_PATH


def project_root() -> pathlib.Path:
    """Return the project root."""
    return pathlib.Path(__file__).resolve().parents[3]


def torch_load(path: pathlib.Path, map_location):
    """Load a PyTorch payload from load."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _checkpoint_geometry(checkpoint, *, image_scaling: float) -> Geometry:
    """Handle checkpoint geometry for the PaDIS workflow."""
    if isinstance(checkpoint, dict) and "geometry" in checkpoint:
        return Geometry.init_from_parameter(checkpoint["geometry"])
    return Geometry.default_parameters(image_scaling=image_scaling)


def _checkpoint_paper_preset(checkpoint) -> str | None:
    """Handle checkpoint paper preset for the PaDIS workflow."""
    if not isinstance(checkpoint, dict):
        return None
    training_params = checkpoint.get("training_params")
    if training_params is None:
        return None
    paper_preset = getattr(training_params, "paper_preset", None)
    if isinstance(paper_preset, str) and paper_preset:
        return paper_preset
    prior_mode = getattr(training_params, "prior_mode", None)
    if prior_mode == "whole_image":
        return "padis-paper-whole-ct-256"
    patch_sizes = getattr(training_params, "patch_sizes", None)
    if patch_sizes:
        largest_patch_size = max(int(size) for size in patch_sizes)
        if largest_patch_size == 64:
            return "padis-paper-ct-512"
        if largest_patch_size in (8, 16, 32, 56, 96):
            preset = (
                "padis-paper-ct-256"
                if largest_patch_size == 56
                else f"padis-paper-ct-p{largest_patch_size}"
            )
            if getattr(training_params, "use_position_channels", True) is False:
                preset = f"{preset}-no-position"
            return preset
    return None


def checkpoint_model_metadata(
    checkpoint_path: pathlib.Path,
    *,
    map_location,
    image_scaling: float,
    disable_position_channels: bool,
    checkpoint=None,
):
    """Handle checkpoint model metadata for the PaDIS workflow."""
    json_path = checkpoint_path.with_suffix(".json")
    if json_path.is_file():
        options = LIONParameter()
        options.load(json_path)
        if getattr(options, "model_name", "NCSNpp") != "NCSNpp":
            warnings.warn(
                f"{json_path} says model_name={options.model_name!r}; trying NCSNpp anyway."
            )
        # LIONParameter.load() intentionally reconstructs nested JSON objects as
        # generic LIONParameter instances. NCSNpp now enforces the more specific
        # LIONModelParameter type, so restore that type at this serialization
        # boundary while preserving every saved field.
        model_params = LIONModelParameter(
            **copy.deepcopy(vars(options.model_parameters))
        )
        geometry = Geometry.init_from_parameter(options.geometry)
    else:
        if checkpoint is None:
            checkpoint = torch_load(checkpoint_path, map_location=map_location)
        paper_preset = _checkpoint_paper_preset(checkpoint)
        if paper_preset is None:
            warnings.warn(
                f"No sidecar JSON found at {json_path}; using PaDIS LIDC 256 defaults."
            )
            paper_preset = "padis-paper-ct-256"
        else:
            warnings.warn(
                f"No sidecar JSON found at {json_path}; inferred {paper_preset!r} "
                "from checkpoint metadata."
            )
        model_params = NCSNpp.default_parameters(paper_preset)
        geometry = _checkpoint_geometry(checkpoint, image_scaling=image_scaling)

    if disable_position_channels:
        model_params.input_position_channels = 0
    return model_params, geometry, checkpoint


def resolve_checkpoint_path(path: pathlib.Path) -> pathlib.Path:
    """Resolve checkpoint path."""
    path = path.expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((pathlib.Path.cwd() / path, project_root() / path))

    if path.parts[:1] == ("Data",):
        data_root = os.environ.get("LION_DATA_PATH")
        if data_root is not None:
            candidates.append(
                pathlib.Path(data_root).expanduser() / pathlib.Path(*path.parts[1:])
            )

    if path == DEFAULT_CHECKPOINT:
        candidates.append(
            LION_EXPERIMENTS_PATH
            / "PaDIS/LIDC_256/padis_lidc_256_reproduction_CSD3/padis_lidc_256.pt"
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Tried:\n  {tried}")


def load_model(
    checkpoint_path: pathlib.Path,
    device: torch.device,
    use_ema: bool,
    disable_position_channels: bool,
):
    """Load model."""
    checkpoint = torch_load(checkpoint_path, map_location=device)
    model_params, geometry, checkpoint = checkpoint_model_metadata(
        checkpoint_path,
        map_location=device,
        image_scaling=0.5,
        disable_position_channels=disable_position_channels,
        checkpoint=checkpoint,
    )

    model = NCSNpp(model_params, geometry).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    ema_state = (
        checkpoint.get("ema_state_dict") if isinstance(checkpoint, dict) else None
    )
    ema_path = checkpoint_path.with_suffix(".ema.pt")
    if ema_state is None and ema_path.is_file():
        ema_checkpoint = torch_load(ema_path, map_location=device)
        ema_state = ema_checkpoint.get("ema_state_dict")
    if use_ema and ema_state is not None:
        state_dict = dict(state_dict)
        state_dict.update(ema_state)
        print("Loaded EMA weights for PaDIS reconstruction.")
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_params, geometry


def load_checkpoint_metadata(
    checkpoint_path: pathlib.Path,
    *,
    image_scaling: float,
    disable_position_channels: bool,
):
    """Load checkpoint metadata."""
    model_params, geometry, _ = checkpoint_model_metadata(
        checkpoint_path,
        map_location="cpu",
        image_scaling=image_scaling,
        disable_position_channels=disable_position_channels,
    )
    return model_params, geometry


def fallback_metadata(
    *,
    image_scaling: float,
    disable_position_channels: bool,
    paper_preset: str = "padis-paper-ct-256",
):
    """Build fallback metadata."""
    model_params = NCSNpp.default_parameters(paper_preset)
    if disable_position_channels:
        model_params.input_position_channels = 0
    return model_params, Geometry.default_parameters(image_scaling=image_scaling)


class PnPDenoiser(torch.nn.Module):
    """Batch/dataset-normalising wrapper for LION image denoisers used by PnP."""

    def __init__(self, model: torch.nn.Module, noise_level: float | None = None):
        """Initialize the instance."""
        super().__init__()
        self.model = model
        self.noise_level = noise_level

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the forward operation to the requested values."""
        squeeze_batch = image.dim() == 3
        if squeeze_batch:
            image = image.unsqueeze(0)
        normalised = (
            self.model.normalise(image) if hasattr(self.model, "normalise") else image
        )
        model_params = getattr(self.model, "model_parameters", None)
        if bool(getattr(model_params, "use_noise_level", False)):
            noise_level = 0.0 if self.noise_level is None else float(self.noise_level)
            denoised = self.model(normalised, noise_level=noise_level)
        else:
            denoised = self.model(normalised)
        if hasattr(self.model, "unnormalise"):
            denoised = self.model.unnormalise(denoised)
        return denoised.squeeze(0) if squeeze_batch else denoised


def load_pnp_denoiser(
    checkpoint_path: pathlib.Path,
    device: torch.device,
    *,
    noise_level: float | None,
) -> PnPDenoiser:
    """Load pnp denoiser."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    options = LIONParameter()
    options.load(checkpoint_path.with_suffix(".json"))
    model_name = getattr(options, "model_name", "DRUNet")
    if model_name != "DRUNet":
        raise ValueError(
            f"Only DRUNet PnP denoiser checkpoints are supported here; got {model_name!r}."
        )
    model_parameters = options.model_parameters
    if not isinstance(model_parameters, LIONModelParameter):
        model_parameters = LIONModelParameter(**model_parameters.serialize())
    model = DRUNet(model_parameters).to(device)
    payload = torch_load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise KeyError(f"{checkpoint_path} does not contain a model_state_dict.")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return PnPDenoiser(model, noise_level=noise_level).to(device).eval()
