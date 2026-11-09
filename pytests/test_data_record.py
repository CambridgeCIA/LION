# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""The record of a model's data (LION.data_loaders.data_record): datasets and
experiments described so they can be made again and checked
(LIONDataset.instantiate, Experiment.instantiate), paths inside
LION_DATA_PATH, noise functions as partials, the seeded random annotation."""

import functools
import json
import os
import shutil
import tempfile
import warnings

# the datasets' default folders are in LION_DATA_PATH
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import LION.data_loaders.LIDC_IDRI as lidc  # noqa: E402
import LION.experiments.ct_experiments as ct_experiments  # noqa: E402
import LION.utils.paths as paths  # noqa: E402
from LION.operators.ct.ct_geometry import Geometry
from LION.operators.ct.ct_utils import sinogram_add_noise  # noqa: E402
from LION.data_loaders.LIONDataset import LIONDataset  # noqa: E402
from LION.data_loaders.data_record import (  # noqa: E402
    data_record,
    is_data_record,
)
from LION.data_loaders.deteCT import deteCT  # noqa: E402
from LION.models.CNNs.dncnn import DnCNN  # noqa: E402
from LION.utils.parameter import LIONParameter  # noqa: E402
from LION.utils.spec import build, encode, to_spec  # noqa: E402
from pytests.test_experiments import fake_detect  # noqa: E402
from pytests.test_lidc_resampling import fake_lidc  # noqa: E402


def through_json(value):
    """As written to a file and read back: a record comes back as a
    LIONParameter (whose specs are dicts), anything else as plain JSON."""
    if isinstance(value, LIONParameter):
        return LIONParameter(**json.loads(json.dumps(value.serialize())))
    return json.loads(json.dumps(value))


# --------------------------------------------------------------------------
# LIDC-IDRI


def stand_in_projection(image):
    return (
        (image.float().mean() + 2.0) * torch.ones(1, 90, 128, device=image.device) * 100
    )


@pytest.fixture
def no_tomosipo(monkeypatch):
    monkeypatch.setattr(Geometry, "make_operator", lambda self: stand_in_projection)


@pytest.fixture
def lidc_folder(tmp_path):
    folder = tmp_path / "LIDC-IDRI"
    folder.mkdir()
    fake_lidc(folder, range(1, 11))
    for k, path in enumerate(sorted(folder.glob("LIDC-IDRI-*/slice_*.npy"))):
        np.save(path, np.full((512, 512), -500 + k, np.int16))
    return folder


def lidc_dataset(folder, mode="test", task="reconstruction"):
    operator_params = (
        lidc.LIDC_IDRI.default_geometry() if task == "reconstruction" else None
    )
    params = lidc.LIDC_IDRI.default_parameters(
        operator_params=operator_params, task=task
    )
    params.folder = folder
    return lidc.LIDC_IDRI(mode, parameters=params)


def test_lidc_dataset_record_rebuilds_the_same_dataset(lidc_folder, no_tomosipo):
    data = lidc_dataset(lidc_folder, mode="train")
    record = through_json(data_record(data))
    assert not hasattr(record, "experiment")
    assert record.samples.count == len(data)
    for cls in (LIONDataset, lidc.LIDC_IDRI):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # same samples, no noise: no warning
            rebuilt = cls.instantiate(record)
        assert type(rebuilt) is lidc.LIDC_IDRI
        assert rebuilt.mode == "train"
        assert rebuilt.sample_ids() == record.samples.ids

    path = paths.LION_DATA_PATH / "processed" / "LIDC-IDRI"
    written = encode(path)
    assert written == {
        "_target_": "LION.utils.paths.data_path",
        "_args_": ["processed/LIDC-IDRI"],
    }
    assert build(through_json(written)) == path

    data = lidc_dataset(lidc_folder)
    data.set_sinogram_transform(functools.partial(sinogram_add_noise, I0=500), seed=3)
    record = through_json(data_record(data))
    assert record.sinogram_transform_seed == 3
    noise = build(record.sinogram_transform)
    assert noise.func is sinogram_add_noise and noise.keywords == {"I0": 500}


# --------------------------------------------------------------------------
# Experiments


def test_experiment_record(lidc_folder, no_tomosipo):
    experiment = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder)
    record = through_json(experiment.data_record("test"))
    assert is_data_record(record)
    assert record.split == "test"
    assert record.experiment["_target_"].endswith("ct_experiments.LowDoseCTRecon")
    assert record.sinogram_transform_seed == 0
    assert record.sinogram_transform["I0"] == ct_experiments.LOW_DOSE_I0
    # made again from the record alone: the same samples and noise
    again = ct_experiments.Experiment.instantiate(record).get_dataset(record.split)
    assert type(again.experiment) is ct_experiments.LowDoseCTRecon
    assert again.sample_ids() == record.samples.ids
    original = experiment.get_testing_dataset()
    assert torch.equal(again[0][0], original[0][0])


# --------------------------------------------------------------------------
# File contents


def test_file_hash_checks_the_data(lidc_folder, tmp_path, no_tomosipo):
    record = data_record(lidc_dataset(lidc_folder), hash_files=True)
    assert record.files.count == 2
    copy = tmp_path / "copy"
    shutil.copytree(lidc_folder, copy)  # another folder, same contents
    again = data_record(lidc_dataset(copy), hash_files=True)
    assert again.files.serialize() == record.files.serialize()
    np.save(copy / "LIDC-IDRI-0010" / "slice_1.npy", np.ones((512, 512), np.int16))
    changed = data_record(lidc_dataset(copy), hash_files=True)
    assert changed.files.sha256 != record.files.sha256
    assert changed.samples.serialize() == record.samples.serialize()  # same names

    record = through_json(data_record(lidc_dataset(lidc_folder), hash_files=True))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LIONDataset.instantiate(record, check_files=True)
    np.save(
        lidc_folder / "LIDC-IDRI-0010" / "slice_1.npy", np.ones((512, 512), np.int16)
    )
    with pytest.warns(UserWarning, match="contents of its files differ"):
        LIONDataset.instantiate(record, check_files=True)
    del record.files
    with pytest.raises(ValueError, match="hash_files=True"):
        LIONDataset.instantiate(record, check_files=True)


# --------------------------------------------------------------------------
# Saved with a model


def test_record_saved_with_a_model(lidc_folder, tmp_path, no_tomosipo):
    record = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder).data_record("test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DnCNN().save(tmp_path / "model", dataset=record)
    _, options, _ = DnCNN.load(tmp_path / "model", supress_warnings=True)
    assert isinstance(options.dataset_params, LIONParameter)
    assert options.dataset_params.serialize() == record.serialize()  # not built
    assert isinstance(options.dataset_params.experiment, dict)  # a spec


# --------------------------------------------------------------------------
# 2DeteCT


def same_geometry(a, b):
    a, b = vars(a), vars(b)
    return a.keys() == b.keys() and all(
        np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a
    )


@pytest.mark.filterwarnings("ignore:You are not using any detector query")
def test_detect_record(tmp_path):
    data = fake_detect(tmp_path, "train")
    assert data.parameters.path_to_dataset == tmp_path
    # no geometry given: 2DeteCT's, and the record says so
    default = deteCT.get_default_geometry()
    assert same_geometry(data.operator_params, default)
    record = through_json(data_record(data))
    assert record.samples.ids == list(data.slice_dataframe["slice_identifier"])
    rebuilt = deteCT.instantiate(record)
    assert type(rebuilt) is deteCT and rebuilt.mode == "train"
    assert same_geometry(rebuilt.operator_params, default)
    assert rebuilt.sample_ids() == record.samples.ids

    # default_parameters() has the geometry a dataset uses by default.
    assert same_geometry(
        deteCT.default_parameters().operator_params, deteCT.get_default_geometry()
    )
    assert same_geometry(
        lidc.LIDC_IDRI.default_parameters().operator_params,
        lidc.LIDC_IDRI.default_geometry(),
    )
    # no sinograms, no geometry
    assert (
        lidc.LIDC_IDRI.default_parameters(task="segmentation").operator_params is None
    )


# --------------------------------------------------------------------------
# instantiate(): datasets and experiments made again from what was saved


def saved_model(folder, record):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DnCNN().save(folder / "model", dataset=record)
    return folder / "model"


def test_dataset_from_a_model_file(lidc_folder, tmp_path, no_tomosipo):
    experiment = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder)
    path = saved_model(tmp_path, experiment.data_record("test"))
    for source in (
        path,
        str(path) + ".pt",
        json.loads(path.with_suffix(".json").read_text()),
    ):
        with pytest.warns(UserWarning, match="without the noise"):
            data = LIONDataset.instantiate(source)
        assert type(data) is lidc.LIDC_IDRI and data.mode == "test"
        assert data.sinogram_transform is None  # the dataset alone
        assert data.sample_ids() == experiment.get_testing_dataset().sample_ids()

    experiment = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder)
    path = saved_model(tmp_path, experiment.data_record("test"))
    again = ct_experiments.LowDoseCTRecon.instantiate(path).get_dataset("test")
    assert torch.equal(again[1][0], experiment.get_testing_dataset()[1][0])

    experiment = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder)
    made = ct_experiments.Experiment.instantiate(through_json(to_spec(experiment)))
    assert type(made) is ct_experiments.LowDoseCTRecon
    assert made.param.noise_params.I0 == experiment.param.noise_params.I0


def test_a_subset_is_recorded_and_made_again(lidc_folder, no_tomosipo):
    data = lidc_dataset(lidc_folder, mode="train")
    subset = torch.utils.data.Subset(data, [2, 0])
    record = through_json(data_record(subset))
    assert record.subset.indices == [2, 0]
    ids = data.sample_ids()
    assert record.samples.ids == [ids[2], ids[0]]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the subset's samples match
        made = LIONDataset.instantiate(record)
    assert isinstance(made, torch.utils.data.Subset) and made.indices == [2, 0]
    assert type(made.dataset) is lidc.LIDC_IDRI


# --------------------------------------------------------------------------
# Groups of samples (patients), for checking leakage


def test_lidc_groups_are_the_patients(lidc_folder, no_tomosipo, tmp_path):
    """The groups of a dataset's samples (LIDC-IDRI: its patients) are kept,
    sorted, for checking that no test patient was trained on; a subset's are
    those of its samples, through nested subsets too."""
    data = lidc_dataset(lidc_folder, mode="train")
    patients = [i.split("/")[0] for i in data.sample_ids()]
    assert data.sample_groups() == patients
    assert through_json(data_record(data)).samples.groups == sorted(set(patients))
    subset = torch.utils.data.Subset(data, [3, 2])
    assert data_record(subset).samples.groups == sorted({patients[3], patients[2]})
    nested = torch.utils.data.Subset(subset, [1])
    assert data_record(nested).samples.groups == [patients[2]]
    assert data_record(nested).samples.ids == [data.sample_ids()[2]]

    # 2DeteCT's splits are by physical sample: its records keep them.
    training = data_record(fake_detect(tmp_path, "train"))
    test = data_record(fake_detect(tmp_path, "test"))
    assert training.samples.groups == [f"sample_{i}" for i in range(8)]
    assert test.samples.groups == ["sample_9"]
