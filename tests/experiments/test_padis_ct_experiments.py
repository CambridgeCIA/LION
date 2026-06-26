import numpy as np
from types import SimpleNamespace

from LION.experiments.ct_experiments import (
    PaDISFanBeam8CTRecon,
    PaDISFanBeam20CTRecon,
    PaDISFanBeam60CTRecon,
    PaDISFanBeam180CTRecon,
)
from scripts.paper_scripts.PaDIS.PaDIS_experiments import PRESETS, command_for


def test_padis_fan_beam_experiments_use_lidc_default_geometry():
    expected = (
        (PaDISFanBeam8CTRecon, 8),
        (PaDISFanBeam20CTRecon, 20),
        (PaDISFanBeam60CTRecon, 60),
        (PaDISFanBeam180CTRecon, 180),
    )

    for experiment_cls, view_count in expected:
        experiment = experiment_cls(image_scaling=0.5)
        geometry = experiment.param.geometry

        assert geometry.mode == "fan"
        assert geometry.image_shape.tolist() == [1, 256, 256]
        assert geometry.detector_shape.tolist() == [1, 900]
        assert len(geometry.angles) == view_count
        assert geometry.angles[0] == 0.0
        assert geometry.angles[-1] < 2 * np.pi
        assert np.isclose(geometry.dso, 575.0)
        assert np.isclose(geometry.dsd, 1050.0)
        assert experiment.param.view_count == view_count


def test_padis_fan_beam_experiment_is_noise_free_and_model_domain():
    experiment = PaDISFanBeam180CTRecon(image_scaling=0.5)

    assert experiment.sino_fun is None
    assert not hasattr(experiment.param, "noise_params")
    assert experiment.param.measurement_source == "normal"
    assert experiment.param.data_loader_params.task == "image_prior"


def test_paper_fan_presets_use_requested_views_and_paper_ct_hyperparameters():
    expected = {
        "paper-fan-8": ("PaDISFanBeam8CTRecon", "0.003"),
        "paper-fan-20": ("PaDISFanBeam20CTRecon", "0.002"),
        "paper-fan-20-no-pos": ("PaDISFanBeam20CTRecon", "0.002"),
        "paper-fan-20-no-pos-fdk": ("PaDISFanBeam20CTRecon", "0.002"),
        "paper-fan-60": ("PaDISFanBeam60CTRecon", "0.002"),
        "paper-fan-180": ("PaDISFanBeam180CTRecon", "0.002"),
    }

    assert set(expected).issubset(PRESETS)
    for preset_name, (experiment_name, sigma_min) in expected.items():
        preset = PRESETS[preset_name]
        arguments = list(preset.arguments)

        assert preset.implementation == "lion-paper-protocol"
        assert preset.experiment == experiment_name
        assert arguments[arguments.index("--num-steps") + 1] == "100"
        assert arguments[arguments.index("--inner-steps") + 1] == "10"
        assert arguments[arguments.index("--sigma-min") + 1] == sigma_min
        assert arguments[arguments.index("--sigma-max") + 1] == "10"
        assert arguments[arguments.index("--noise-schedule") + 1] == "geometric"
        assert arguments[arguments.index("--zeta") + 1] == "0.3"
        assert arguments[arguments.index("--dps-epsilon") + 1] == "1"
        assert arguments[arguments.index("--sampling-epsilon") + 1] == "1"
        assert (
            arguments[arguments.index("--data-consistency-gradient") + 1]
            == "paper_squared_residual"
        )
        assert arguments[arguments.index("--adjoint-data-step-schedule") + 1] == "paper"
        assert "--no-clip-output" in arguments


def test_20_view_no_position_ablation_presets():
    no_pos_args = list(PRESETS["paper-fan-20-no-pos"].arguments)
    no_pos_fdk_args = list(PRESETS["paper-fan-20-no-pos-fdk"].arguments)

    assert PRESETS["paper-fan-20-no-pos"].experiment == "PaDISFanBeam20CTRecon"
    assert PRESETS["paper-fan-20-no-pos-fdk"].experiment == "PaDISFanBeam20CTRecon"
    assert "--no-position-channels" in no_pos_args
    assert "--no-position-channels" in no_pos_fdk_args
    assert no_pos_args[no_pos_args.index("--initial-reconstruction") + 1] == "noise"
    assert (
        no_pos_fdk_args[no_pos_fdk_args.index("--initial-reconstruction") + 1] == "fdk"
    )


def test_unconditional_generation_preset_uses_generation_engine(tmp_path):
    args = SimpleNamespace(
        preset="paper-generation",
        output_root=tmp_path,
        checkpoint=None,
        split="validation",
        start_index=0,
        max_samples=3,
        seed=0,
        device="cuda",
    )
    command, output_folder = command_for(args, [])
    preset = PRESETS[args.preset]
    arguments = list(preset.arguments)

    assert preset.engine == "generation"
    assert preset.experiment is None
    assert output_folder == tmp_path / "lion-paper-protocol" / args.preset
    assert any(item.endswith("PaDIS_LIDC_generation.py") for item in command)
    assert "--experiment" not in command
    assert command[command.index("--num-samples") + 1] == "3"
    assert arguments[arguments.index("--num-steps") + 1] == "1000"
    assert arguments[arguments.index("--inner-steps") + 1] == "1"
    assert arguments[arguments.index("--sigma-min") + 1] == "0.002"
    assert arguments[arguments.index("--sigma-max") + 1] == "40"


def test_compatible_lion_presets_are_separate_and_use_closest_ct_hyperparameters():
    expected = {
        "lion-compatible-clinical": "clinicalCTRecon",
        "lion-compatible-low-dose": "LowDoseCTRecon",
        "lion-compatible-extreme-low-dose": "ExtremeLowDoseCTRecon",
        "lion-compatible-limited-angle": "LimitedAngleCTRecon",
        "lion-compatible-limited-angle-low-dose": "LimitedAngleLowDoseCTRecon",
        "lion-compatible-limited-angle-extreme-low-dose": (
            "LimitedAngleExtremeLowDoseCTRecon"
        ),
        "lion-compatible-sparse-50": "SparseAngleCTRecon",
        "lion-compatible-sparse-low-dose-50": "SparseAngleLowDoseCTRecon",
        "lion-compatible-sparse-extreme-low-dose-50": (
            "SparseAngleExtremeLowDoseCTRecon"
        ),
    }

    assert set(expected).issubset(PRESETS)
    for preset_name, experiment_name in expected.items():
        preset = PRESETS[preset_name]
        arguments = list(preset.arguments)

        assert preset.implementation == "lion-compatible"
        assert preset.experiment == experiment_name
        assert arguments[arguments.index("--num-steps") + 1] == "100"
        assert arguments[arguments.index("--inner-steps") + 1] == "10"
        assert arguments[arguments.index("--sigma-min") + 1] == "0.002"
        assert arguments[arguments.index("--sigma-max") + 1] == "10"
        assert arguments[arguments.index("--initial-reconstruction") + 1] == "noise"


def test_removed_padis_presets_do_not_reappear():
    removed = {
        "whole-paper-fan-180",
        "lion-clinical",
        "lion-low-dose",
        "lion-sparse-50",
        "lion-sparse-low-dose-50",
    }

    assert removed.isdisjoint(PRESETS)


def test_paper_fan_preset_is_separated_and_overrideable(tmp_path):
    args = SimpleNamespace(
        preset="paper-fan-180",
        output_root=tmp_path,
        checkpoint=None,
        split="validation",
        start_index=0,
        max_samples=1,
        seed=0,
        device="cuda",
    )
    command, output_folder = command_for(args, ["--sigma-max", "0.1"])

    assert PRESETS[args.preset].implementation == "lion-paper-protocol"
    assert output_folder == tmp_path / "lion-paper-protocol" / args.preset
    assert command[-2:] == ["--sigma-max", "0.1"]
    assert "PaDISFanBeam180CTRecon" in command


def test_lion_compatible_preset_has_own_namespace(tmp_path):
    args = SimpleNamespace(
        preset="lion-compatible-sparse-50",
        output_root=tmp_path,
        checkpoint=None,
        split="validation",
        start_index=0,
        max_samples=1,
        seed=0,
        device="cuda",
    )
    command, output_folder = command_for(args, [])

    assert output_folder == tmp_path / "lion-compatible" / args.preset
    assert "SparseAngleCTRecon" in command
