"""Datasets, experiments, and measurements for PaDIS LIDC reconstruction."""

from __future__ import annotations

import pathlib

import numpy as np
import PIL.Image
import torch

from LION.CTtools.ct_utils import sinogram_add_noise
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI

from padis_lidc.experiments import (
    EXPERIMENT_ALIASES,
    LIDC_EXPERIMENTS,
    LIDC_NORMAL_TO_MU_OFFSET,
    LIDC_NORMAL_TO_MU_SCALE,
    PAPER_CT_EXPERIMENTS,
    PUBLIC_REPO_IMPLEMENTATION_METHODS,
    UNSUPPORTED_PADIS_GEOMETRY_MESSAGE,
    PaperCTExperiment,
)


class PNGImagePriorDataset(torch.utils.data.Dataset):
    """Image-prior dataset backed by PaDIS-style PNG slices."""

    def __init__(self, image_dir: pathlib.Path, channels: int):
        """Initialize the instance."""
        self.image_dir = pathlib.Path(image_dir).expanduser()
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"PNG image directory not found: {self.image_dir}")
        self.channels = int(channels)
        self.files = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() == ".png"
        )
        if not self.files:
            raise FileNotFoundError(f"No PNG files found in {self.image_dir}")

    def __len__(self):
        """Return the number of available items."""
        return len(self.files)

    def __getitem__(self, index):
        """Return the item at the requested index."""
        image = np.asarray(PIL.Image.open(self.files[index]), dtype=np.float32) / 255.0
        if self.channels == 1:
            if image.ndim == 3:
                image = image[..., 0]
            image = image[None, :, :]
        elif self.channels == 3:
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            image = np.transpose(image, (2, 0, 1))
        else:
            raise ValueError("PNGImagePriorDataset supports 1 or 3 channels.")
        return torch.empty(0), torch.from_numpy(image.copy()).float()


def build_dataset(args, geometry):
    """Build dataset."""
    task = "image_prior" if args.measurement_source == "normal" else "reconstruction"
    data_params = LIDC_IDRI.default_parameters(geometry=geometry, task=task)
    data_params.device = torch.device("cpu")
    if args.data_folder is not None:
        data_params.folder = args.data_folder
    return LIDC_IDRI(args.split, parameters=data_params, geometry_parameters=geometry)


def canonical_experiment_name(name: str) -> str:
    """Return the canonical experiment name."""
    return EXPERIMENT_ALIASES.get(name, name)


def validate_public_repo_method(implementation: str, method: str) -> None:
    """Validate public repo method."""
    if (
        implementation == "public_repo"
        and method not in PUBLIC_REPO_IMPLEMENTATION_METHODS
    ):
        supported = ", ".join(sorted(PUBLIC_REPO_IMPLEMENTATION_METHODS))
        raise ValueError(
            "--implementation public_repo is only supported for methods with "
            "a public PaDIS inverse-sampler analogue: "
            f"{supported}. Method {method!r} has no runnable public-repo "
            "equivalent."
        )


def experiment_spec_from_args(args) -> PaperCTExperiment | None:
    """Handle experiment spec from args for the PaDIS workflow."""
    canonical = canonical_experiment_name(args.experiment)
    return PAPER_CT_EXPERIMENTS.get(canonical)


def experiment_class_for_geometry(
    spec: PaperCTExperiment,
    geometry_tag: str,
) -> type:
    """Handle experiment class for geometry for the PaDIS workflow."""
    if geometry_tag == "lion":
        return LIDC_EXPERIMENTS[spec.lion_experiment]
    if geometry_tag in ("padis", "padis_parallel", "padis_fanbeam"):
        raise ValueError(UNSUPPORTED_PADIS_GEOMETRY_MESSAGE)
    raise ValueError(f"Unknown geometry tag: {geometry_tag}")


def build_experiment_dataset(args, checkpoint_geometry):
    """Build experiment dataset."""
    image_scaling = float(getattr(checkpoint_geometry, "image_scaling", 1.0))
    spec = experiment_spec_from_args(args)
    if spec is None:
        experiment_cls = LIDC_EXPERIMENTS[args.experiment]
    else:
        experiment_cls = experiment_class_for_geometry(spec, args.geometry)
    experiment = experiment_cls(
        dataset="LIDC-IDRI",
        datafolder=args.data_folder,
        image_scaling=image_scaling,
    )
    if args.split == "validation":
        dataset = experiment.get_validation_dataset()
    elif args.split == "test":
        dataset = experiment.get_testing_dataset()
    else:
        raise ValueError(f"Unsupported split for reconstruction: {args.split}")
    return dataset, experiment.geometry, experiment


def mu_to_lidc_normal(image: torch.Tensor) -> torch.Tensor:
    """Handle mu to lidc normal for the PaDIS workflow."""
    return ((image - LIDC_NORMAL_TO_MU_OFFSET) / LIDC_NORMAL_TO_MU_SCALE).clamp(
        0.0, 1.0
    )


def make_measurement(
    args,
    dataset,
    index,
    reconstructor,
    device,
    *,
    from_experiment,
    experiment_measurement_source,
):
    """Create measurement."""
    sample = dataset[index]
    if from_experiment:
        if experiment_measurement_source == "normal":
            target = sample[1].float().to(device)
            sinogram = reconstructor.op(target)
        else:
            sinogram, target = sample
            sinogram = sinogram.float().to(device)
            target = mu_to_lidc_normal(target.float().to(device))
        return sinogram, target

    if args.measurement_source == "normal":
        target = sample[1].float().to(device)
        sinogram = reconstructor.op(target)
    else:
        sinogram, target = sample
        sinogram = sinogram.float().to(device)
        target = mu_to_lidc_normal(target.float().to(device))

    if args.noise == "low-dose":
        if sinogram.device.type != "cuda":
            raise ValueError("LION low-dose sinogram noise currently requires CUDA.")
        sinogram = sinogram_add_noise(
            sinogram,
            I0=args.noise_i0,
            sigma=args.noise_sigma,
            cross_talk=args.noise_cross_talk,
            enable_gradients=False,
        )
    return sinogram, target
