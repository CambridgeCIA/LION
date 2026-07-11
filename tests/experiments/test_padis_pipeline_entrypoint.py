from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "scripts/paper_scripts/PaDIS/PaDIS_run_pipeline.sh"


def _dry_run(backend: str):
    return subprocess.run(
        ["bash", str(PIPELINE), "--backend", backend, "--dry-run"],
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_unified_pipeline_dispatches_to_gcp_runner():
    result = _dry_run("gcp")

    assert result.returncode == 0, result.stderr
    assert "PaDIS pipeline backend: gcp" in result.stdout
    assert "gcp/run_PaDIS_GCP_spot_training.sh" in result.stdout


def test_unified_pipeline_dispatches_to_slurm_submitter():
    result = _dry_run("slurm")

    assert result.returncode == 0, result.stderr
    assert "PaDIS pipeline backend: slurm" in result.stdout
    assert "slurm/submit_PaDIS_A100_pipeline.sh" in result.stdout


def test_unified_pipeline_requires_a_backend():
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
    pipeline_text = PIPELINE.read_text()
    finaliser = ROOT / "scripts/paper_scripts/PaDIS/PaDIS_finalise_pipeline.sh"
    finaliser_text = finaliser.read_text()

    assert "PaDIS_finalise_pipeline.sh" in pipeline_text
    assert "PaDIS_LIDC_generation.py" in finaliser_text
    assert "PaDIS_make_tables.py" in finaliser_text
    assert "PaDIS_make_paper_figures.py" in finaliser_text
