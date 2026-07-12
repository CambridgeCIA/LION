"""Test padis ct experiments behaviour."""

import numpy as np
from types import SimpleNamespace

from LION.experiments.ct_experiments import (
    PaDISFanBeam8CTRecon,
    PaDISFanBeam20CTRecon,
    PaDISFanBeam60CTRecon,
    PaDISFanBeam120LimitedCTRecon,
    PaDISFanBeam180CTRecon,
)
from PaDIS_LIDC_reconstruction import (
    build_arg_parser,
    build_sampler_params,
)
from PaDIS_experiments import (
    FIGURE_PRESET_GROUPS,
    PRESETS,
    apply_generation_checkpoint_defaults,
    build_parser as build_experiments_parser,
    command_for,
    figure_command_for,
)


def test_padis_fan_beam_experiments_use_lidc_default_geometry():
    """Verify that padis fan beam experiments use lidc default geometry."""
    expected = (
        (PaDISFanBeam8CTRecon, 8, 2 * np.pi),
        (PaDISFanBeam20CTRecon, 20, 2 * np.pi),
        (PaDISFanBeam60CTRecon, 60, 2 * np.pi),
        (PaDISFanBeam120LimitedCTRecon, 20, 2 * np.pi / 3),
        (PaDISFanBeam180CTRecon, 20, 2 * np.pi / 3),
    )

    for experiment_cls, view_count, angle_span in expected:
        experiment = experiment_cls(image_scaling=0.5)
        geometry = experiment.param.geometry

        assert geometry.mode == "fan"
        assert geometry.image_shape.tolist() == [1, 256, 256]
        assert geometry.detector_shape.tolist() == [1, 900]
        assert len(geometry.angles) == view_count
        assert geometry.angles[0] == 0.0
        assert geometry.angles[-1] < angle_span
        assert np.isclose(experiment.param.angle_span, angle_span)
        assert np.isclose(geometry.dso, 575.0)
        assert np.isclose(geometry.dsd, 1050.0)
        assert experiment.param.view_count == view_count


def test_padis_fanbeam_extra_row_is_120_degree_limited_angle():
    """Verify that padis fanbeam extra row is 120 degree limited angle."""
    experiment = PaDISFanBeam120LimitedCTRecon(image_scaling=0.5)
    geometry = experiment.param.geometry

    assert len(geometry.angles) == 20
    assert np.isclose(geometry.angles[0], 0.0)
    assert np.isclose(geometry.angles[1] - geometry.angles[0], np.pi / 30)
    assert np.isclose(geometry.angles[-1], 19 * np.pi / 30)
    assert np.isclose(experiment.param.angle_span, 2 * np.pi / 3)


def test_padis_fan_beam_experiment_is_noise_free_and_model_domain():
    """Verify that padis fan beam experiment is noise free and model domain."""
    experiment = PaDISFanBeam180CTRecon(image_scaling=0.5)

    assert experiment.sino_fun is None
    assert not hasattr(experiment.param, "noise_params")
    assert experiment.param.measurement_source == "normal"
    assert experiment.param.data_loader_params.task == "image_prior"


def test_paper_fan_presets_use_requested_views_and_paper_ct_hyperparameters():
    """Verify that paper fan presets use requested views and paper ct hyperparameters."""
    expected = {
        "paper-fan-8": ("PaDISFanBeam8CTRecon", "0.003"),
        "paper-fan-20": ("PaDISFanBeam20CTRecon", "0.002"),
        "paper-fan-20-no-pos": ("PaDISFanBeam20CTRecon", "0.002"),
        "paper-fan-20-no-pos-fdk": ("PaDISFanBeam20CTRecon", "0.002"),
        "paper-fan-60": ("PaDISFanBeam60CTRecon", "0.002"),
        "paper-fan-180": ("PaDISFanBeam120LimitedCTRecon", "0.002"),
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


def test_public_repo_branch_uses_paper_sigma_schedule_with_public_mechanics():
    """Verify that public repo branch uses paper sigma schedule with public mechanics."""
    expected = {
        "ct_20": 0.002,
        "ct_8": 0.003,
        "ct_60": 0.002,
        "ct_20_limited_angle_120": 0.002,
        "ct_512_60": 0.002,
    }
    parser = build_arg_parser()

    for experiment_name, sigma_min in expected.items():
        args = parser.parse_args(
            ["--implementation", "public_repo", "--experiment", experiment_name]
        )
        params = build_sampler_params(args, None, measurement_source="normal")

        assert params.num_steps == 100
        assert params.inner_steps == 10
        assert params.sigma_min == sigma_min
        assert params.sigma_max == 10.0
        assert params.noise_schedule == "geometric"
        assert params.initial_reconstruction == "fdk"
        assert params.data_consistency_gradient == "norm"
        assert params.adjoint_data_step_schedule == "public_repo"


def test_public_repo_branch_can_use_literal_readme_sigma_schedule():
    """Verify that public repo branch can use literal readme sigma schedule."""
    args = build_arg_parser().parse_args(
        [
            "--implementation",
            "public_repo",
            "--experiment",
            "ct_20",
            "--public-repo-sigma-schedule",
            "readme",
        ]
    )
    params = build_sampler_params(args, None, measurement_source="normal")

    assert params.sigma_min == 0.003
    assert params.sigma_max == 10.0
    assert params.noise_schedule == "edm"
    assert params.initial_reconstruction == "fdk"
    assert params.data_consistency_gradient == "norm"


def test_method_flags_set_expected_sampler_modes():
    """Verify that method flags set expected sampler modes."""
    parser = build_arg_parser()

    average_args = parser.parse_args(
        [
            "--implementation",
            "paper",
            "--experiment",
            "ct_20",
            "--method",
            "patch_average",
        ]
    )
    average_params = build_sampler_params(
        average_args, None, measurement_source="normal"
    )
    assert average_params.prior_mode == "patch"
    assert average_params.patch_assembly == "fixed_average"
    assert average_params.fixed_overlap_layout == "lion_clipped"
    assert average_params.fixed_overlap_checkpoint_denoiser is True

    stitch_args = parser.parse_args(
        [
            "--implementation",
            "paper",
            "--experiment",
            "ct_20",
            "--method",
            "patch_stitch",
        ]
    )
    stitch_params = build_sampler_params(stitch_args, None, measurement_source="normal")
    assert stitch_params.patch_assembly == "fixed_stitch"
    assert stitch_params.fixed_overlap_layout == "lion_clipped"
    assert stitch_params.fixed_overlap_checkpoint_denoiser is True

    public_average_args = parser.parse_args(
        [
            "--implementation",
            "public_repo",
            "--experiment",
            "ct_20",
            "--method",
            "patch_average",
        ]
    )
    public_average_params = build_sampler_params(
        public_average_args, None, measurement_source="normal"
    )
    assert public_average_params.patch_assembly == "fixed_average"
    assert public_average_params.fixed_overlap_layout == "public_overlap"

    public_stitch_args = parser.parse_args(
        [
            "--implementation",
            "public_repo",
            "--experiment",
            "ct_20",
            "--method",
            "patch_stitch",
        ]
    )
    public_stitch_params = build_sampler_params(
        public_stitch_args, None, measurement_source="normal"
    )
    assert public_stitch_params.patch_assembly == "fixed_stitch"
    assert public_stitch_params.fixed_overlap_layout == "public_tile"

    no_checkpoint_average_args = parser.parse_args(
        [
            "--implementation",
            "paper",
            "--experiment",
            "ct_20",
            "--method",
            "patch_average",
            "--no-fixed-overlap-checkpoint-denoiser",
        ]
    )
    no_checkpoint_average_params = build_sampler_params(
        no_checkpoint_average_args,
        None,
        measurement_source="normal",
    )
    assert no_checkpoint_average_params.fixed_overlap_checkpoint_denoiser is False

    ddnm_args = parser.parse_args(
        ["--implementation", "paper", "--experiment", "ct_20", "--method", "ve_ddnm"]
    )
    ddnm_params = build_sampler_params(ddnm_args, None, measurement_source="normal")
    assert ddnm_params.langevin_ddnm is True
    assert ddnm_params.num_steps == 1000
    assert ddnm_params.inner_steps == 1
    assert ddnm_params.ve_ddnm_nfe_layout == "paper_1000x1"
    assert ddnm_params.ddnm_pseudoinverse_clip is True
    assert ddnm_params.ddnm_projected_pseudoinverse_clip is True
    assert ddnm_params.ddnm_corrected_clip is False
    assert ddnm_params.sampling_epsilon == 0.1

    stable_ddnm_args = parser.parse_args(
        [
            "--implementation",
            "lion_quality",
            "--experiment",
            "ct_20",
            "--method",
            "ve_ddnm",
        ]
    )
    stable_ddnm_params = build_sampler_params(
        stable_ddnm_args,
        None,
        measurement_source="normal",
    )
    assert stable_ddnm_params.num_steps == 1000
    assert stable_ddnm_params.inner_steps == 1
    assert stable_ddnm_params.ve_ddnm_nfe_layout == "paper_1000x1"
    assert stable_ddnm_params.initial_reconstruction == "noise"
    assert stable_ddnm_params.initial_fdk_filter_type is None
    assert stable_ddnm_params.initial_fdk_frequency_scaling == 1.0
    assert stable_ddnm_params.initial_fdk_padded is True
    assert stable_ddnm_params.clip_initial is False
    assert stable_ddnm_params.clip_output is False
    assert stable_ddnm_params.sampling_epsilon == 0.1
    assert stable_ddnm_params.ddnm_corrected_clip is True

    public_inner_ddnm_args = parser.parse_args(
        [
            "--implementation",
            "paper",
            "--experiment",
            "ct_20",
            "--method",
            "ve_ddnm",
            "--ve-ddnm-nfe-layout",
            "public_inner",
        ]
    )
    public_inner_ddnm_params = build_sampler_params(
        public_inner_ddnm_args,
        None,
        measurement_source="normal",
    )
    assert public_inner_ddnm_params.num_steps == 100
    assert public_inner_ddnm_params.inner_steps == 10
    assert public_inner_ddnm_params.ve_ddnm_nfe_layout == "public_inner"

    public_repo_ddnm_args = parser.parse_args(
        [
            "--implementation",
            "public_repo",
            "--experiment",
            "ct_20",
            "--method",
            "ve_ddnm",
        ]
    )
    public_repo_ddnm_params = build_sampler_params(
        public_repo_ddnm_args,
        None,
        measurement_source="normal",
    )
    assert public_repo_ddnm_params.num_steps == 100
    assert public_repo_ddnm_params.inner_steps == 10
    assert public_repo_ddnm_params.ve_ddnm_nfe_layout == "public_inner"

    strict_ddnm_args = parser.parse_args(
        [
            "--implementation",
            "paper",
            "--experiment",
            "ct_20",
            "--method",
            "ve_ddnm",
            "--no-ddnm-projected-pseudoinverse-clip",
        ]
    )
    strict_ddnm_params = build_sampler_params(
        strict_ddnm_args,
        None,
        measurement_source="normal",
    )
    assert strict_ddnm_params.ddnm_projected_pseudoinverse_clip is False

    clipped_ddnm_args = parser.parse_args(
        [
            "--implementation",
            "paper",
            "--experiment",
            "ct_20",
            "--method",
            "ve_ddnm",
            "--ddnm-corrected-clip",
        ]
    )
    clipped_ddnm_params = build_sampler_params(
        clipped_ddnm_args,
        None,
        measurement_source="normal",
    )
    assert clipped_ddnm_params.ddnm_corrected_clip is True

    pc_args = parser.parse_args(
        [
            "--implementation",
            "paper",
            "--experiment",
            "ct_20",
            "--method",
            "predictor_corrector",
        ]
    )
    pc_params = build_sampler_params(pc_args, None, measurement_source="normal")
    assert pc_params.pc_snr == 0.08
    assert pc_params.pc_corrector_step_rule == "paper_linear"
    assert pc_params.pc_corrector_denoise_sigma == "next"
    assert pc_params.pc_reuse_predictor_layout is False

    public_pc_args = parser.parse_args(
        [
            "--implementation",
            "public_repo",
            "--experiment",
            "ct_20",
            "--method",
            "predictor_corrector",
        ]
    )
    public_pc_params = build_sampler_params(
        public_pc_args, None, measurement_source="normal"
    )
    assert public_pc_params.pc_corrector_step_rule == "paper_linear"
    assert public_pc_params.pc_corrector_denoise_sigma == "current"
    assert public_pc_params.pc_reuse_predictor_layout is True
    assert public_pc_params.data_consistency_scale == 0.0405
    assert public_pc_params.adjoint_data_consistency_scale == 0.1022

    whole_args = parser.parse_args(
        [
            "--implementation",
            "paper",
            "--experiment",
            "ct_20",
            "--method",
            "whole_image_diffusion",
        ]
    )
    whole_params = build_sampler_params(whole_args, None, measurement_source="normal")
    assert whole_params.prior_mode == "whole_image"


def test_reconstruction_parser_defaults_to_paper_ct_test_count():
    """Verify that reconstruction parser defaults to paper ct test count."""
    args = build_arg_parser().parse_args([])

    assert args.split == "test"
    assert args.max_samples == 25


def test_20_view_no_position_ablation_presets():
    """Verify that 20 view no position ablation presets."""
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
    """Verify that unconditional generation preset uses generation engine."""
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
    assert arguments[arguments.index("--num-steps") + 1] == "300"
    assert arguments[arguments.index("--inner-steps") + 1] == "1"
    assert arguments[arguments.index("--sigma-min") + 1] == "0.002"
    assert arguments[arguments.index("--sigma-max") + 1] == "10"
    assert arguments[arguments.index("--noise-schedule") + 1] == "geometric"


def test_generation_presets_use_separately_tuned_prior_settings():
    """Verify that generation presets use separately tuned prior settings."""
    patch_arguments = list(PRESETS["paper-generation"].arguments)
    whole_arguments = list(PRESETS["paper-generation-whole"].arguments)

    assert patch_arguments[patch_arguments.index("--generation-epsilon") + 1] == "0.8"
    assert (
        patch_arguments[patch_arguments.index("--langevin-noise-scale") + 1] == "0.85"
    )
    assert whole_arguments[whole_arguments.index("--generation-epsilon") + 1] == "0.75"
    assert (
        whole_arguments[whole_arguments.index("--langevin-noise-scale") + 1] == "0.75"
    )


def test_generation_figure_presets_cover_patch_assembly_methods():
    """Verify that generation figure presets cover patch assembly methods."""
    assert FIGURE_PRESET_GROUPS["paper-generation-figures"] == (
        "paper-generation-whole",
        "paper-generation-naive-patch",
        "paper-generation",
        "paper-generation-langevin-300nfe",
        "paper-generation-patch-stitch",
        "paper-generation-patch-average",
    )
    stitch_args = list(PRESETS["paper-generation-patch-stitch"].arguments)
    average_args = list(PRESETS["paper-generation-patch-average"].arguments)

    assert stitch_args[stitch_args.index("--patch-assembly") + 1] == "fixed_stitch"
    assert stitch_args[stitch_args.index("--fixed-overlap-layout") + 1] == "public_tile"
    assert average_args[average_args.index("--patch-assembly") + 1] == "fixed_average"
    assert (
        average_args[average_args.index("--fixed-overlap-layout") + 1]
        == "public_overlap"
    )
    assert "--fixed-overlap-checkpoint-denoiser" in stitch_args
    assert "--fixed-overlap-checkpoint-denoiser" in average_args


def test_generation_group_can_default_checkpoints_from_gcp_training_root(tmp_path):
    """Verify that generation group can default checkpoints from gcp training root."""
    parser = build_experiments_parser()
    args, passthrough = parser.parse_known_args(
        [
            "run-group",
            "paper-generation-figures",
            "--training-root-preset",
            "gcp",
            "--run-root",
            str(tmp_path / "runs"),
            "--gcp-run-name",
            "PaDIS-Reproduction-GCP-test",
            "--output-root",
            str(tmp_path / "generation"),
            "--dry-run",
        ]
    )

    assert passthrough == []
    apply_generation_checkpoint_defaults(args)

    training_root = (
        tmp_path / "runs" / "final_real_runs" / "PaDIS-Reproduction-GCP-test"
    ).resolve()
    assert args.patch_checkpoint == (
        training_root / "patch_lidc_default" / "padis_lidc_256.pt"
    )
    assert args.whole_checkpoint == (
        training_root / "whole_lidc_default" / "whole_image_lidc_256_min_val.pt"
    )

    child_args = SimpleNamespace(**vars(args))
    child_args.preset = "paper-generation-whole"
    child_args.checkpoint = child_args.whole_checkpoint
    command, _ = command_for(child_args, [])

    assert command[command.index("--checkpoint") + 1].endswith(
        "whole_lidc_default/whole_image_lidc_256_min_val.pt"
    )


def test_experiments_entrypoint_can_render_paper_figures(tmp_path):
    """Verify that experiments entrypoint can render paper figures."""
    parser = build_experiments_parser()
    args, passthrough = parser.parse_known_args(
        [
            "make-figures",
            "--reconstruction-root",
            str(tmp_path / "recon"),
            "--generation-root",
            str(tmp_path / "generation"),
            "--output-folder",
            str(tmp_path / "figures"),
            "--figures",
            "figureA1_ct20_additional,figureA2_ct8_additional",
            "--sample-index",
            "4",
            "--allow-missing",
            "--dry-run",
        ]
    )

    assert passthrough == []
    command = figure_command_for(args)

    assert any(item.endswith("PaDIS_make_paper_figures.py") for item in command)
    assert command[command.index("--reconstruction-root") + 1] == str(
        tmp_path / "recon"
    )
    assert command[command.index("--generation-root") + 1] == str(
        tmp_path / "generation"
    )
    assert command[command.index("--output-folder") + 1] == str(tmp_path / "figures")
    assert command[command.index("--figures") + 1] == (
        "figureA1_ct20_additional,figureA2_ct8_additional"
    )
    assert command[command.index("--sample-index") + 1] == "4"
    assert "--allow-missing" in command


def test_compatible_lion_presets_are_separate_and_use_closest_ct_hyperparameters():
    """Verify that compatible lion presets are separate and use closest ct hyperparameters."""
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
    """Verify that removed padis presets do not reappear."""
    removed = {
        "whole-paper-fan-180",
        "lion-clinical",
        "lion-low-dose",
        "lion-sparse-50",
        "lion-sparse-low-dose-50",
    }

    assert removed.isdisjoint(PRESETS)


def test_paper_fan_preset_is_separated_and_overrideable(tmp_path):
    """Verify that paper fan preset is separated and overrideable."""
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
    assert "PaDISFanBeam120LimitedCTRecon" in command


def test_lion_compatible_preset_has_own_namespace(tmp_path):
    """Verify that lion compatible preset has own namespace."""
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
