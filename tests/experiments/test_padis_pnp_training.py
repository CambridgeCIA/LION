import warnings

import pytest
import torch

warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script` is deprecated",
    category=DeprecationWarning,
)

from PaDIS_LIDC_256 import run_prefix_for_prior_mode
from PaDIS_LIDC_PnP_denoiser import (
    build_arg_parser,
    cuda_device_index,
    validate_args,
)


def test_pnp_training_defaults_match_paper_matrix_expectations():
    args = build_arg_parser().parse_args([])

    assert args.run_name == "pnp_lidc_drunet"
    assert args.final_name == "pnp_lidc_drunet.pt"
    assert args.image_scaling == 0.5
    assert args.max_slices_per_patient == 4
    assert args.batch_size == 8
    assert args.epochs == 100
    assert args.learning_rate == 1e-4
    assert args.noise_min == 0.0
    assert args.noise_max == 0.05
    assert args.int_channels == 64
    assert args.n_blocks == 4
    assert args.validation_every == 1
    assert args.checkpoint_every == 10
    assert args.max_periodic_checkpoints == 5
    validate_args(args)


def test_lidc_256_training_prefixes_match_reconstruction_checkpoint_names():
    assert run_prefix_for_prior_mode("patch") == "padis_lidc_256"
    assert run_prefix_for_prior_mode("whole-image") == "whole_image_lidc_256"

    with pytest.raises(ValueError, match="Unsupported prior mode"):
        run_prefix_for_prior_mode("invalid")


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--batch-size", "0", "--batch-size"),
        ("--epochs", "0", "--epochs"),
        ("--learning-rate", "0", "--learning-rate"),
        ("--beta1", "1", "--beta1/--beta2"),
        ("--beta2", "-0.1", "--beta1/--beta2"),
        ("--noise-min", "-0.1", "--noise-min/--noise-max"),
        ("--max-slices-per-patient", "0", "--max-slices-per-patient"),
        ("--int-channels", "0", "--int-channels"),
        ("--n-blocks", "0", "--n-blocks"),
        ("--patch-size", "0", "--patch-size"),
        ("--patches-per-image", "0", "--patches-per-image"),
        ("--max-train-samples", "0", "--max-train-samples"),
        ("--max-validation-samples", "0", "--max-validation-samples"),
        ("--validation-every", "0", "--validation-every"),
        ("--checkpoint-every", "0", "--checkpoint-every"),
        ("--max-periodic-checkpoints", "0", "--max-periodic-checkpoints"),
        ("--max-periodic-checkpoints", "-2", "--max-periodic-checkpoints"),
        ("--num-workers", "-1", "--num-workers"),
        ("--final-name", "", "--final-name"),
    ],
)
def test_pnp_training_validation_rejects_invalid_values(flag, value, message):
    args = build_arg_parser().parse_args([flag, value])

    with pytest.raises(ValueError, match=message):
        validate_args(args)


def test_pnp_training_validation_rejects_inverted_noise_range():
    args = build_arg_parser().parse_args(["--noise-min", "0.2", "--noise-max", "0.1"])

    with pytest.raises(ValueError, match="--noise-min/--noise-max"):
        validate_args(args)


def test_pnp_training_validation_allows_full_lidc_without_slice_limit():
    args = build_arg_parser().parse_args(
        ["--full-lidc", "--max-slices-per-patient", "0"]
    )

    validate_args(args)


def test_cuda_device_index_defaults_bare_cuda_to_zero():
    assert cuda_device_index(torch.device("cuda")) == 0
    assert cuda_device_index(torch.device("cuda:2")) == 2
    with pytest.raises(ValueError, match="CUDA device"):
        cuda_device_index(torch.device("cpu"))
