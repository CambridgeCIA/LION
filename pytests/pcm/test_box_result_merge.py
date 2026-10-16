"""Tests for safely adding scalar TV to an earlier PCM box-plot run."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from LION.pcm.box_result_merge import merge_scalar_tv_box_results
from LION.pcm.results import RESULT_COLUMNS


def _make_run(
    root: Path, method: str, *, noise_value: float = 0.012, altered_zf: bool = False
) -> None:
    root.mkdir(parents=True)
    settings = {
        "data": {
            "root": "/synthetic-png",
            "categories": ["CIGS"],
            "samples": {"ids": [288, 470]},
        },
        "scaling": {"is_out_of_distribution": False},
        "sampling": {"sampling_ratios": [0.2], "coarse_j_values": [6]},
        "uses_normalized_op": True,
        "noise": {"beam_sigma": noise_value},
        "base_seed": 42,
        "runtime": {"num_trials": 1, "num_trials_skip": 0},
        "methods": {
            method: {"lambda_reg": 0.001} if method == "scalar_tv" else {"iters": 50}
        },
    }
    (root / "config.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    for method_id in ("zero_filled", method):
        dest = root / method_id / "CIGS.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for sample_id in (288, 470):
                score = 0.75 if method_id == "zero_filled" else 0.9
                if altered_zf and method_id == "zero_filled":
                    score += 0.05
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "trial_num": 1,
                        "sampling_rate": 0.2,
                        "coarse_j": 6,
                        "rng_seed": sample_id + 42,
                        "mse": 0.01,
                        "psnr": 20.0,
                        "ssim": 0.8,
                        "haarpsi": score,
                        "pearson_corr": 0.9,
                        "duration": 0.123,
                    }
                )


def test_merge_keeps_original_runs_and_copies_paired_csvs(tmp_path: Path) -> None:
    original = tmp_path / "original"
    tv = tmp_path / "tv"
    combined = tmp_path / "combined"
    _make_run(original, "spgl1")
    _make_run(tv, "scalar_tv")

    assert merge_scalar_tv_box_results(original, tv, combined) == combined
    assert (combined / "spgl1" / "CIGS.csv").exists()
    assert (combined / "zero_filled" / "CIGS.csv").exists()
    assert (combined / "scalar_tv" / "CIGS.csv").exists()
    assert not (original / "scalar_tv").exists()
    assert not (tv / "spgl1").exists()
    with (combined / "config.yaml").open(encoding="utf-8") as f:
        assert set(yaml.safe_load(f)["methods"]) == {"spgl1", "scalar_tv"}


def test_merge_rejects_different_noise_parameters(tmp_path: Path) -> None:
    _make_run(tmp_path / "original", "spgl1")
    _make_run(tmp_path / "tv", "scalar_tv", noise_value=0.014)
    output = tmp_path / "combined"
    with pytest.raises(ValueError, match="noise"):
        merge_scalar_tv_box_results(tmp_path / "original", tmp_path / "tv", output)
    assert not output.exists()


def test_merge_rejects_different_acquisition_realizations(tmp_path: Path) -> None:
    _make_run(tmp_path / "original", "spgl1")
    _make_run(tmp_path / "tv", "scalar_tv", altered_zf=True)
    output = tmp_path / "combined"
    with pytest.raises(ValueError, match="Zero-filled haarpsi"):
        merge_scalar_tv_box_results(tmp_path / "original", tmp_path / "tv", output)
    assert not output.exists()


def test_merge_rejects_missing_scalar_tv_trials(tmp_path: Path) -> None:
    _make_run(tmp_path / "original", "spgl1")
    _make_run(tmp_path / "tv", "scalar_tv")
    csv_path = tmp_path / "tv" / "scalar_tv" / "CIGS.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow(rows[0])
    with pytest.raises(ValueError, match="Unmatched"):
        merge_scalar_tv_box_results(
            tmp_path / "original", tmp_path / "tv", tmp_path / "combined"
        )


def test_merge_does_not_overwrite_output(tmp_path: Path) -> None:
    _make_run(tmp_path / "original", "spgl1")
    _make_run(tmp_path / "tv", "scalar_tv")
    output = tmp_path / "combined"
    output.mkdir()
    with pytest.raises(FileExistsError):
        merge_scalar_tv_box_results(tmp_path / "original", tmp_path / "tv", output)
