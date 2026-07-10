import json

from scripts.paper_scripts.PaDIS.PaDIS_reconcile_reconstruction_manifest import (
    completed_output_exists,
)


def _job(tmp_path, *, expected_sampler=None, expected_method_settings=None):
    output = tmp_path / "output"
    job = {
        "command": [
            "python",
            "driver.py",
            "--output-folder",
            str(output),
            "--experiment",
            "ct_20",
            "--split",
            "test",
            "--method",
            "padis_dps",
            "--algorithm",
            "dps_langevin",
            "--max-samples",
            "2",
        ],
        "expected_sampler": expected_sampler or {},
        "expected_method_settings": expected_method_settings or {},
    }
    metrics_path = output / "ct_20/test/padis_dps/dps_langevin/metrics.json"
    metrics_path.parent.mkdir(parents=True)
    return job, metrics_path


def _write_metrics(path, *, sampler=None, method_settings=None):
    path.write_text(
        json.dumps(
            {
                "sampler": sampler or {},
                "method_settings": method_settings or {},
                "metrics": [{"index": 0}, {"index": 1}],
            }
        )
    )


def test_completed_output_reused_when_recorded_settings_match(tmp_path):
    job, path = _job(
        tmp_path,
        expected_sampler={"zeta": 4.5, "patch_checkpoint_denoiser": True},
        expected_method_settings={"iterations": 20},
    )
    _write_metrics(
        path,
        sampler={"zeta": 4.5, "patch_checkpoint_denoiser": True, "extra": 1},
        method_settings={"iterations": 20},
    )

    assert completed_output_exists(job)


def test_completed_output_invalidated_when_sampler_hyperparameter_changes(tmp_path):
    job, path = _job(tmp_path, expected_sampler={"zeta": 4.5})
    _write_metrics(path, sampler={"zeta": 4.25})

    assert not completed_output_exists(job)


def test_completed_output_can_be_reused_outside_selected_validation_scope(tmp_path):
    job, path = _job(tmp_path, expected_sampler={"zeta": 4.5})
    _write_metrics(path, sampler={"zeta": 4.25})

    assert completed_output_exists(job, validate_settings=False)


def test_completed_output_invalidated_when_expected_setting_is_missing(tmp_path):
    job, path = _job(
        tmp_path,
        expected_sampler={"patch_batch_size": 8},
    )
    _write_metrics(path, sampler={"zeta": 4.5})

    assert not completed_output_exists(job)


def test_completed_output_invalidated_when_method_setting_changes(tmp_path):
    job, path = _job(
        tmp_path,
        expected_method_settings={"iterations": 20},
    )
    _write_metrics(path, method_settings={"iterations": 10})

    assert not completed_output_exists(job)
