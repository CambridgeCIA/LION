"""Test padis pipeline entrypoint behaviour."""

from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = (
    ROOT / "scripts/paper_scripts/PaDIS-Reproduction/pipeline/PaDIS_run_pipeline.sh"
)


def _dry_run(backend: str):
    """Create dry run test support data."""
    return subprocess.run(
        ["bash", str(PIPELINE), "--backend", backend, "--dry-run"],
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_unified_pipeline_dispatches_to_gcp_runner():
    """Verify that unified pipeline dispatches to gcp runner."""
    result = _dry_run("gcp")

    assert result.returncode == 0, result.stderr
    assert "PaDIS pipeline backend: gcp" in result.stdout
    assert "gcp/run_PaDIS_GCP_spot_training.sh" in result.stdout


def test_unified_pipeline_dispatches_to_slurm_submitter():
    """Verify that unified pipeline dispatches to slurm submitter."""
    result = _dry_run("slurm")

    assert result.returncode == 0, result.stderr
    assert "PaDIS pipeline backend: slurm" in result.stdout
    assert "slurm/submit_PaDIS_A100_pipeline.sh" in result.stdout


def test_unified_pipeline_requires_a_backend():
    """Verify that unified pipeline requires a backend."""
    result = subprocess.run(
        ["bash", str(PIPELINE), "--dry-run"],
        cwd=ROOT,
        env={
            key: value
            for key, value in os.environ.items()
            if key != "PADIS_PIPELINE_BACKEND"
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Select a backend" in result.stderr


def test_unified_pipeline_finishes_with_generation_tables_and_figures():
    """Verify that unified pipeline finishes with generation tables and figures."""
    pipeline_text = PIPELINE.read_text()
    finaliser = (
        ROOT
        / "scripts/paper_scripts/PaDIS-Reproduction/pipeline/PaDIS_finalise_pipeline.sh"
    )
    finaliser_text = finaliser.read_text()

    assert "PaDIS_finalise_pipeline.sh" in pipeline_text
    assert "core/PaDIS_experiments.py" in finaliser_text
    assert 'run "$preset"' in finaliser_text
    assert "PADIS_GENERATION_EPSILON:-1.0" not in finaliser_text
    assert "--langevin-noise-scale" in finaliser_text
    assert "PaDIS_make_tables.py" in finaliser_text
    assert "PaDIS_make_paper_figures.py" in finaliser_text


def test_production_generation_paths_use_authoritative_named_presets():
    """Ensure production wrappers preserve per-preset generation tuning."""

    gcp_runner = (
        ROOT / "scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp/"
        "run_PaDIS_GCP_manual_reconstruction.sh"
    ).read_text()
    finaliser = (
        ROOT / "scripts/paper_scripts/PaDIS-Reproduction/pipeline/"
        "PaDIS_finalise_pipeline.sh"
    ).read_text()

    for wrapper in (gcp_runner, finaliser):
        assert "core/PaDIS_experiments.py" in wrapper
        assert 'run "$preset"' in wrapper
        assert "PADIS_GENERATION_EPSILON:-1.0" not in wrapper

    assert 'PADIS_GENERATION_EPSILON="${PADIS_GENERATION_EPSILON:-}"' in gcp_runner
    assert (
        'PADIS_GENERATION_NOISE_SCALE="${PADIS_GENERATION_NOISE_SCALE:-}"' in gcp_runner
    )
