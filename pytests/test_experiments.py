"""Experiment definitions and data loaders (no real data needed)."""

import os
import tempfile

# LION.utils.paths requires LION_DATA_PATH at import time
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import LION.experiments.ct_experiments as ct_experiments  # noqa: E402
import LION.data_loaders.LIDC_IDRI as lidc  # noqa: E402
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


@pytest.mark.parametrize("saved_before", [False, True])
def test_experiments_choose_the_noise_model(monkeypatch, saved_before):
    """New experiments use the current noise model; parameters saved before
    legacy_noise_model existed reproduce the old one."""
    params = ct_experiments.LowDoseCTRecon.default_parameters()
    assert params.noise_params.legacy_noise_model is False
    if saved_before:
        del params.noise_params.legacy_noise_model
    calls = []
    monkeypatch.setattr(
        ct_experiments.ct, "sinogram_add_noise", lambda sino, **kw: calls.append(kw)
    )
    ct_experiments.LowDoseCTRecon(params).sino_fun("sinogram")
    assert calls[0]["legacy"] is saved_before
    assert calls[0]["I0"] == params.noise_params.I0
    assert calls[0]["mass_attenuation"] is None  # relative noise model by default
    assert calls[0]["min_counts"] is None  # LION's old rule for signals <= 0


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


# --------------------------------------------------------------------------
# LIDC-IDRI: slices taken per patient

NON_NODULE, NODULE = list(range(100)), list(range(100, 110))


@pytest.mark.parametrize(
    "non_nodule, nodule, max_slices, expected_total, expected_nodule",
    [
        (NON_NODULE, NODULE, 5, 5, 2),  # both kinds available: the requested split
        (NON_NODULE, [], 5, 5, 0),  # no nodule slices: non-nodule ones only
        ([7], NODULE, 5, 5, 4),  # too few non-nodule slices: more nodule ones
        ([7], [100], 5, 2, 1),  # fewer slices than asked for: all of them
        (NON_NODULE, NODULE, -1, 110, 10),  # -1: every slice
    ],
)
def test_lidc_slice_selection(
    non_nodule, nodule, max_slices, expected_total, expected_nodule
):
    selected = lidc.select_slices(non_nodule, nodule, max_slices, 0.5)
    assert len(selected) == expected_total
    assert sum(s in nodule for s in selected) == expected_nodule
    assert selected == sorted(set(selected))  # sorted, no repeats


def test_lidc_legacy_selection_skipped_patients_without_nodules():
    assert lidc.select_slices(NON_NODULE, [], 5, 0.5, legacy=True) == []
    # with both kinds available, legacy and current agree
    assert lidc.select_slices(NON_NODULE, NODULE, 5, 0.5, legacy=True) == (
        lidc.select_slices(NON_NODULE, NODULE, 5, 0.5)
    )


@pytest.mark.parametrize("saved_before", [False, True])
def test_lidc_uses_legacy_selection_for_old_parameters(saved_before):
    params = lidc.LIDC_IDRI.default_parameters(task="segmentation")
    assert params.legacy_slice_selection is False
    if saved_before:
        del params.legacy_slice_selection
    dataset = lidc.LIDC_IDRI.__new__(lidc.LIDC_IDRI)  # no data needed
    dataset.params = params
    selected = dataset.get_slices_to_load(["p"], {"p": NON_NODULE}, {"p": []}, 5, 0.5)
    assert len(selected["p"]) == (0 if saved_before else 5)
