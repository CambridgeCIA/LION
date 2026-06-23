import numpy as np
from types import SimpleNamespace

from LION.experiments.ct_experiments import PaDISFanBeamCTRecon
from scripts.example_scripts.PaDIS_experiments import PRESETS, command_for


def test_padis_fan_beam_experiment_uses_lidc_default_geometry_with_180_views():
    experiment = PaDISFanBeamCTRecon(image_scaling=0.5)
    geometry = experiment.param.geometry

    assert geometry.mode == "fan"
    assert geometry.image_shape.tolist() == [1, 256, 256]
    assert geometry.detector_shape.tolist() == [1, 900]
    assert len(geometry.angles) == 180
    assert geometry.angles[0] == 0.0
    assert geometry.angles[-1] < 2 * np.pi
    assert np.isclose(geometry.dso, 575.0)
    assert np.isclose(geometry.dsd, 1050.0)


def test_padis_fan_beam_experiment_is_noise_free_and_model_domain():
    experiment = PaDISFanBeamCTRecon(image_scaling=0.5)

    assert experiment.sino_fun is None
    assert not hasattr(experiment.param, "noise_params")
    assert experiment.param.measurement_source == "normal"
    assert experiment.param.data_loader_params.task == "image_prior"


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
    assert "PaDISFanBeamCTRecon" in command
