"""Experiment definitions and data loaders (no real data needed)."""

import os
import tempfile

# LION.utils.paths requires LION_DATA_PATH at import time
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np  # noqa: E402
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
    """New experiments use the physical noise model; parameters saved before
    noise_model existed (25 Oct 2026) use the relative one."""
    params = ct_experiments.LowDoseCTRecon.default_parameters()
    assert params.noise_params.noise_model == "physical"
    assert params.noise_params.I0 == ct_experiments.LOW_DOSE_I0
    if saved_before:  # what the parameter files of that time contain
        for name in ("noise_model", "detector", "mass_attenuation", "min_counts"):
            delattr(params.noise_params, name)
        params.noise_params.I0 = 3500
        params.noise_params.legacy_noise_model = False  # ignored now
    calls = []
    monkeypatch.setattr(
        ct_experiments.ct, "sinogram_add_noise", lambda sino, **kw: calls.append(kw)
    )
    ct_experiments.LowDoseCTRecon(params).sino_fun("sinogram")
    (kw,) = calls
    assert kw["I0"] == params.noise_params.I0
    assert "legacy" not in kw
    assert kw["model"] == ("relative" if saved_before else "physical")
    assert (kw["detector"], kw["min_counts"]) == ("energy_integrating", 1.0)
    assert kw["mass_attenuation"] == ct_experiments.ct.WATER_MASS_ATTENUATION[80]


def test_photon_counting_experiment(monkeypatch):
    params = ct_experiments.LowDoseCTRecon.default_parameters()
    params.noise_params = ct_experiments.Experiment.noise_parameters(
        5000, detector="photon_counting"
    )
    calls = []
    monkeypatch.setattr(
        ct_experiments.ct, "sinogram_add_noise", lambda sino, **kw: calls.append(kw)
    )
    ct_experiments.LowDoseCTRecon(params).sino_fun("sinogram")
    assert calls[0]["detector"] == "photon_counting" and calls[0]["I0"] == 5000


def test_experiment_doses():
    doses = {
        cls: cls.default_parameters().noise_params.I0
        for cls in (
            ct_experiments.clinicalCTRecon,
            ct_experiments.LowDoseCTRecon,
            ct_experiments.ExtremeLowDoseCTRecon,
            ct_experiments.SparseAngleExtremeLowDoseCTRecon,
        )
    }
    assert list(doses.values()) == [20000, 5000, 1000, 1000]


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
def test_detect_noise_I0_per_detector(tmp_path):
    """noise_params.I0 is one number for every slice (as before October 2026)
    or a dict per detector (the default of deteCT.noise_parameters)."""
    data = fake_detect(tmp_path, "train")
    detectors = data.slice_dataframe["detector"].values
    data.noise_params = deteCT.noise_parameters()  # per detector
    for index, detector in enumerate(detectors):
        assert data.noise_I0(index) == {1: 290, 2: 100}[detector]
    data.noise_params = deteCT.noise_parameters(I0={"1": 300, "2": 120})  # from JSON
    assert [data.noise_I0(i) for i in range(3)] == [
        {1: 300, 2: 120}[d] for d in detectors[:3]
    ]
    data.noise_params = deteCT.noise_parameters(I0=200)  # one for all
    assert {data.noise_I0(i) for i in range(len(detectors))} == {200}


@pytest.mark.filterwarnings("ignore:You are not using any detector query")
def test_detect_loader_adds_noise_with_the_slice_I0(tmp_path, monkeypatch):
    data = fake_detect(tmp_path, "train")  # slices 0-2: the first sample
    geometry = deteCT.get_default_geometry()
    n_angles, n_det = len(geometry.angles), geometry.detector_shape[-1]
    for k in range(3):
        folder = tmp_path / f"slice{k:05d}" / "mode2"
        folder.mkdir(parents=True)
        np.save(folder / "sinogram.npy", np.full((n_angles, n_det), 0.5, np.float32))
        np.save(folder / "flat.npy", np.ones((1, n_det), np.float32))
        np.save(folder / "dark.npy", np.zeros((1, n_det), np.float32))
    data.add_noise = True
    data.noise_params = deteCT.noise_parameters()
    seen = []
    monkeypatch.setattr(
        data, "add_sinogram_noise", lambda sino, I0, cross_talk: seen.append(I0) or sino
    )
    for k in range(3):
        data.__load_and_preprocess_sinogram__(k, "mode2")
    assert seen == [290, 100, 290]  # slice 1 of the sample is detector 2


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
