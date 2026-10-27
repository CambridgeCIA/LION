"""The record of a model's data (LION.data_loaders.data_record): datasets and
experiments described so they can be made again and checked, paths inside
LION_DATA_PATH, noise functions as partials, the seeded random annotation."""

import functools
import json
import os
import pathlib
import shutil
import tempfile
import warnings

# LION.utils.paths requires LION_DATA_PATH at import time
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import LION.data_loaders.LIDC_IDRI as lidc  # noqa: E402
import LION.experiments.ct_experiments as ct_experiments  # noqa: E402
import LION.utils.paths as paths  # noqa: E402
from LION.CTtools.ct_utils import sinogram_add_noise  # noqa: E402
from LION.data_loaders.data_record import data_record, is_data_record  # noqa: E402
from LION.data_loaders.deteCT import deteCT  # noqa: E402
from LION.models.CNNs.dncnn import DnCNN  # noqa: E402
from LION.utils.device import default_device  # noqa: E402
from LION.utils.spec import SpecError, build, encode, to_spec  # noqa: E402
from pytests.test_experiments import fake_detect  # noqa: E402
from pytests.test_lidc_resampling import fake_lidc  # noqa: E402


def through_json(value):
    return json.loads(json.dumps(value))


# --------------------------------------------------------------------------
# Specs: paths inside LION_DATA_PATH, partials, unreadable objects


def test_paths_inside_lion_data_path_are_relative():
    path = paths.LION_DATA_PATH / "processed" / "LIDC-IDRI"
    written = encode(path)
    assert written == {
        "_target_": "LION.utils.paths.data_path",
        "_args_": ["processed/LIDC-IDRI"],
    }
    assert build(through_json(written)) == path


def test_other_paths_are_written_as_given():
    path = pathlib.Path("/somewhere/else/data")
    assert build(through_json(encode(path))) == path


def test_noise_function_as_a_partial():
    noise = functools.partial(sinogram_add_noise, I0=500, sigma=3)
    rebuilt = build(through_json(encode(noise)))
    assert rebuilt.func is sinogram_add_noise
    assert rebuilt.keywords == {"I0": 500, "sigma": 3}


class TakesAnything:
    def __init__(self, *args, **kwargs):
        self.args = args


def test_object_whose_arguments_cannot_be_read_raises():
    """It used to be written as its class alone, silently."""
    with pytest.raises(SpecError, match=r"\*args or \*\*kwargs"):
        to_spec(TakesAnything(1, 2))


# --------------------------------------------------------------------------
# LIDC-IDRI


def stand_in_projection(image):
    return (image.float().mean() + 2.0) * torch.ones(1, 90, 128) * 100


@pytest.fixture
def no_tomosipo(monkeypatch):
    monkeypatch.setattr(lidc.ct, "make_operator", lambda geometry: stand_in_projection)


@pytest.fixture
def lidc_folder(tmp_path):
    folder = tmp_path / "LIDC-IDRI"
    folder.mkdir()
    fake_lidc(folder, range(1, 11))
    for k, path in enumerate(sorted(folder.glob("LIDC-IDRI-*/slice_*.npy"))):
        np.save(path, np.full((512, 512), -500 + k, np.int16))
    return folder


def lidc_dataset(folder, mode="test", task="reconstruction"):
    geometry = lidc.LIDC_IDRI.default_geometry() if task == "reconstruction" else None
    params = lidc.LIDC_IDRI.default_parameters(geometry=geometry, task=task)
    params.folder = folder
    return lidc.LIDC_IDRI(mode, parameters=params)


def test_lidc_keeps_its_arguments(lidc_folder, no_tomosipo):
    data = lidc_dataset(lidc_folder)
    assert data.mode == "test"
    assert data.parameters.folder == lidc_folder
    assert data.geometry_parameters is None


def test_lidc_uses_the_selected_device(lidc_folder, no_tomosipo):
    assert "device" not in vars(lidc.LIDC_IDRI.default_parameters(task="segmentation"))
    params = lidc.LIDC_IDRI.default_parameters(task="segmentation")
    params.folder = lidc_folder
    params.device = "cuda:7"  # as in parameter files saved before: not used
    data = lidc.LIDC_IDRI("test", parameters=params)
    assert data.device == default_device()


def test_lidc_sample_ids(lidc_folder, no_tomosipo):
    data = lidc_dataset(lidc_folder)
    assert data.sample_ids() == ["LIDC-IDRI-0010/slice_0", "LIDC-IDRI-0010/slice_1"]
    assert [p.name for p in data.sample_files()] == ["slice_0.npy", "slice_1.npy"]


def test_lidc_dataset_record_rebuilds_the_same_dataset(lidc_folder, no_tomosipo):
    data = lidc_dataset(lidc_folder, mode="train")
    record = through_json(data_record(data))
    assert "experiment" not in record
    assert record["samples"]["count"] == len(data)
    rebuilt = build(record["dataset"])
    assert type(rebuilt) is lidc.LIDC_IDRI
    assert rebuilt.sample_ids() == record["samples"]["ids"]


# --------------------------------------------------------------------------
# A user's own pipeline: LION's noise with their parameters, or their own


def test_users_own_noise_is_recorded(lidc_folder, no_tomosipo):
    data = lidc_dataset(lidc_folder)
    data.set_sinogram_transform(functools.partial(sinogram_add_noise, I0=500), seed=3)
    record = through_json(data_record(data))
    assert record["sinogram_transform_seed"] == 3
    noise = build(record["sinogram_transform"])
    assert noise.func is sinogram_add_noise and noise.keywords == {"I0": 500}


def test_a_lambda_is_named_with_a_warning(lidc_folder, no_tomosipo):
    data = lidc_dataset(lidc_folder)
    data.set_sinogram_transform(lambda s: s + 1)
    with pytest.warns(UserWarning, match="can't be written as a spec"):
        record = data_record(data)
    assert "lambda" in record["sinogram_transform"]["unsaved"]


# --------------------------------------------------------------------------
# Experiments


def test_experiment_record(lidc_folder, no_tomosipo):
    experiment = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder)
    record = through_json(experiment.data_record("test"))
    assert is_data_record(record)
    assert record["split"] == "test"
    assert record["experiment"]["_target_"].endswith("ct_experiments.LowDoseCTRecon")
    assert record["sinogram_transform_seed"] == 0
    assert record["sinogram_transform"]["I0"] == ct_experiments.LOW_DOSE_I0
    # made again from the record alone: the same samples and noise
    again = build(record["experiment"]).get_dataset(record["split"])
    assert again.sample_ids() == record["samples"]["ids"]
    original = experiment.get_testing_dataset()
    assert torch.equal(again[0][0], original[0][0])


def test_datasets_remember_their_experiment(lidc_folder, no_tomosipo):
    experiment = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder)
    data = experiment.get_training_dataset()
    assert data.experiment is experiment and data.split == "train"
    assert data_record(data)["split"] == "train"
    with pytest.raises(ValueError, match="split"):
        experiment.get_dataset("training")


def test_experiment_noise_is_unchanged(lidc_folder, no_tomosipo):
    """The partial gives the same noise as the lambda it replaces."""
    experiment = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder)
    noise = experiment.param.noise_params
    sinogram = torch.full((1, 90, 128), 200.0)
    torch.manual_seed(0)
    by_partial = experiment.sino_fun(sinogram)
    torch.manual_seed(0)
    by_hand = sinogram_add_noise(
        sinogram,
        I0=noise.I0,
        sigma=noise.sigma,
        sigma_blur=noise.sigma_blur,
        model=noise.noise_model,
        detector=noise.detector,
        mass_attenuation=noise.mass_attenuation,
        min_counts=noise.min_counts,
    )
    assert torch.equal(by_partial, by_hand)


# --------------------------------------------------------------------------
# File contents


def test_file_hash_checks_the_data(lidc_folder, tmp_path, no_tomosipo):
    record = data_record(lidc_dataset(lidc_folder), hash_files=True)
    assert record["files"]["count"] == 2
    copy = tmp_path / "copy"
    shutil.copytree(lidc_folder, copy)  # another folder, same contents
    assert data_record(lidc_dataset(copy), hash_files=True)["files"] == record["files"]
    np.save(copy / "LIDC-IDRI-0010" / "slice_1.npy", np.ones((512, 512), np.int16))
    changed = data_record(lidc_dataset(copy), hash_files=True)
    assert changed["files"]["sha256"] != record["files"]["sha256"]
    assert changed["samples"] == record["samples"]  # same names


def test_no_file_hash_unless_asked(lidc_folder, no_tomosipo):
    assert "files" not in data_record(lidc_dataset(lidc_folder))


# --------------------------------------------------------------------------
# Saved with a model


def test_record_saved_with_a_model(lidc_folder, tmp_path, no_tomosipo):
    record = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder).data_record("test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DnCNN().save(tmp_path / "model", dataset=record)
    _, options, _ = DnCNN.load(tmp_path / "model", supress_warnings=True)
    assert options.dataset_params == through_json(record)  # read as it is, not built


# --------------------------------------------------------------------------
# The random annotation (segmentation), seeded


@pytest.fixture
def annotated_folder(lidc_folder):
    """Patient 1's slice 1 has a nodule with three different annotations."""
    masks = json.loads((lidc_folder / "patients_masks.json").read_text())
    masks["LIDC-IDRI-0001"] = {"1": {"0": ["1", "2", "3"]}}
    (lidc_folder / "patients_masks.json").write_text(json.dumps(masks))
    for k in (1, 2, 3):
        mask = np.zeros((512, 512), bool)
        mask[k * 10 : k * 10 + 5] = True
        np.save(
            lidc_folder / "LIDC-IDRI-0001" / f"mask_1_nodule_0_annotation_{k}.npy", mask
        )
    return lidc_folder


def read_annotation(data, times):
    index = data.sample_ids().index("LIDC-IDRI-0001/slice_1")
    return {
        tuple(data[index][1][1].any(dim=1).nonzero().flatten().tolist())
        for _ in range(times)
    }


def test_random_annotation_is_seeded(annotated_folder):
    params = lidc.LIDC_IDRI.default_parameters(task="segmentation")
    params.folder = annotated_folder
    params.annotation = "random"
    data = lidc.LIDC_IDRI("train", parameters=params)
    assert len(read_annotation(data, 20)) == 1  # one annotation for the nodule
    data.parameters.annotation_seed = None  # as parameter files saved before
    assert len(read_annotation(data, 40)) > 1


# --------------------------------------------------------------------------
# 2DeteCT


@pytest.mark.filterwarnings("ignore:You are not using any detector query")
def test_detect_record(tmp_path):
    data = fake_detect(tmp_path, "train")
    assert data.parameters.path_to_dataset == tmp_path
    assert data.geometry_params is None
    record = through_json(data_record(data))
    assert record["samples"]["ids"] == list(data.slice_dataframe["slice_identifier"])
    rebuilt = build(record["dataset"])
    assert rebuilt.sample_ids() == record["samples"]["ids"]
