"""Experiment definitions and data loaders (no real data needed)."""

import os
import tempfile

# LION.utils.paths requires LION_DATA_PATH at import time
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import LION.experiments.ct_experiments as ct_experiments  # noqa: E402
from LION.data_loaders.deteCT import deteCT  # noqa: E402

EXPERIMENTS = [
    cls
    for cls in vars(ct_experiments).values()
    if isinstance(cls, type)
    and issubclass(cls, ct_experiments.Experiment)
    and cls is not ct_experiments.Experiment
]


def test_experiment_names_are_unique():
    names = [cls.default_parameters().name for cls in EXPERIMENTS]
    assert len(EXPERIMENTS) == 9
    assert len(set(names)) == len(names), names


@pytest.mark.parametrize("cls", EXPERIMENTS, ids=lambda c: c.__name__)
def test_noise_function_is_always_defined(cls):
    params = cls.default_parameters()
    assert cls(params).sino_fun is not None
    del params.noise_params
    assert cls(params).sino_fun is None


# --------------------------------------------------------------------------
# 2DeteCT: splits and length, from a fake data record (no sinograms needed)


def fake_detect(tmp_path, mode, n_samples=10, query=""):
    """n_samples samples of 3 slices each; slice 1 of every sample is detector 2."""
    rows = []
    for sample in range(n_samples):
        for k in range(3):
            slice_index = 3 * sample + k
            rows.append(
                dict(
                    slice_identifier=f"slice{slice_index:05d}",
                    sample_index=sample,
                    first_slice=3 * sample,
                    last_slice=3 * sample + 2,
                    detector=2 if k == 1 else 1,
                )
            )
    pd.DataFrame(rows).to_csv(tmp_path / "default_data_records.csv", index=False)
    params = deteCT.default_parameters()
    params.path_to_dataset = tmp_path
    params.query = query
    return deteCT(mode, parameters=params)


@pytest.mark.filterwarnings("ignore:You are not using any detector query")
@pytest.mark.parametrize("n_samples", [10, 17, 100])
def test_detect_splits_use_every_sample_once(tmp_path, n_samples):
    splits = {
        mode: set(
            fake_detect(tmp_path, mode, n_samples).slice_dataframe["sample_index"]
        )
        for mode in ("train", "validation", "test")
    }
    sizes = [len(splits[m]) for m in ("train", "validation", "test")]
    # train and validation as before (floor of n * proportion); test gets the rest
    end_train, end_validation = int(0.8 * n_samples), int(0.9 * n_samples + 1e-9)
    assert sizes == [end_train, end_validation - end_train, n_samples - end_validation]
    assert set.union(*splits.values()) == set(range(n_samples))


@pytest.mark.parametrize("query", ["", "detector==1"])
@pytest.mark.filterwarnings("ignore:You are not using any detector query")
def test_detect_length_is_the_number_of_slices(tmp_path, query):
    dataset = fake_detect(tmp_path, "test", query=query)
    assert len(dataset) == len(dataset.slice_dataframe)
    assert len(dataset) == (3 if query == "" else 2)  # 1 test sample, 3 slices
