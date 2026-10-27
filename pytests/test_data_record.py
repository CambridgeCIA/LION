"""The record of a model's data (LION.data_loaders.data_record): datasets and
experiments described so they can be made again and checked
(LIONDataset.instantiate, Experiment.instantiate), paths inside
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
from LION.data_loaders.LIONDataset import LIONDataset  # noqa: E402
from LION.data_loaders.data_record import (  # noqa: E402
    data_record,
    is_data_record,
    read_record,
)
from LION.data_loaders.deteCT import deteCT  # noqa: E402
from LION.models.CNNs.dncnn import DnCNN  # noqa: E402
from LION.utils.device import default_device  # noqa: E402
from LION.utils.parameter import LIONParameter  # noqa: E402
from LION.utils.spec import SpecError, build, encode, to_spec  # noqa: E402
from pytests.test_experiments import fake_detect  # noqa: E402
from pytests.test_lidc_resampling import fake_lidc  # noqa: E402


def through_json(value):
    """As written to a file and read back: a record comes back as a
    LIONParameter (whose specs are dicts), anything else as plain JSON."""
    if isinstance(value, LIONParameter):
        return LIONParameter(**json.loads(json.dumps(value.serialize())))
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
    return (
        (image.float().mean() + 2.0) * torch.ones(1, 90, 128, device=image.device) * 100
    )


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


def test_making_a_split_leaves_the_experiment_unchanged(lidc_folder, no_tomosipo):
    """An experiment's datasets share its data_loader_params: making one split
    doesn't change the experiment, or the record of another split."""
    experiment = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder)
    before = json.dumps(to_spec(experiment), sort_keys=True, default=str)
    training = experiment.get_dataset("train")
    training_record = json.dumps(data_record(training).serialize(), default=str)
    experiment.get_dataset("test")
    assert json.dumps(to_spec(experiment), sort_keys=True, default=str) == before
    assert json.dumps(data_record(training).serialize(), default=str) == (
        training_record
    )
    assert training.mode == "train"


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
    assert not hasattr(record, "experiment")
    assert record.samples.count == len(data)
    for cls in (LIONDataset, lidc.LIDC_IDRI):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # same samples, no noise: no warning
            rebuilt = cls.instantiate(record)
        assert type(rebuilt) is lidc.LIDC_IDRI
        assert rebuilt.mode == "train"
        assert rebuilt.sample_ids() == record.samples.ids


# --------------------------------------------------------------------------
# A user's own pipeline: LION's noise with their parameters, or their own


def test_users_own_noise_is_recorded(lidc_folder, no_tomosipo):
    data = lidc_dataset(lidc_folder)
    data.set_sinogram_transform(functools.partial(sinogram_add_noise, I0=500), seed=3)
    record = through_json(data_record(data))
    assert record.sinogram_transform_seed == 3
    noise = build(record.sinogram_transform)
    assert noise.func is sinogram_add_noise and noise.keywords == {"I0": 500}


def test_a_lambda_is_named_with_a_warning(lidc_folder, no_tomosipo):
    data = lidc_dataset(lidc_folder)
    data.set_sinogram_transform(lambda s: s + 1)
    with pytest.warns(UserWarning, match="can't be written as a spec"):
        record = data_record(data)
    assert "lambda" in record.sinogram_transform.unsaved


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


def test_datasets_remember_their_experiment(lidc_folder, no_tomosipo):
    experiment = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder)
    data = experiment.get_training_dataset()
    assert data.experiment is experiment and data.split == "train"
    assert data_record(data).split == "train"
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
    assert record.files.count == 2
    copy = tmp_path / "copy"
    shutil.copytree(lidc_folder, copy)  # another folder, same contents
    again = data_record(lidc_dataset(copy), hash_files=True)
    assert again.files.serialize() == record.files.serialize()
    np.save(copy / "LIDC-IDRI-0010" / "slice_1.npy", np.ones((512, 512), np.int16))
    changed = data_record(lidc_dataset(copy), hash_files=True)
    assert changed.files.sha256 != record.files.sha256
    assert changed.samples.serialize() == record.samples.serialize()  # same names


def test_no_file_hash_unless_asked(lidc_folder, no_tomosipo):
    assert not hasattr(data_record(lidc_dataset(lidc_folder)), "files")


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
    assert record.samples.ids == list(data.slice_dataframe["slice_identifier"])
    rebuilt = deteCT.instantiate(record)
    assert type(rebuilt) is deteCT and rebuilt.mode == "train"
    assert rebuilt.sample_ids() == record.samples.ids


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


def test_experiment_from_a_model_file(lidc_folder, tmp_path, no_tomosipo):
    experiment = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder)
    path = saved_model(tmp_path, experiment.data_record("test"))
    again = ct_experiments.LowDoseCTRecon.instantiate(path).get_dataset("test")
    assert torch.equal(again[1][0], experiment.get_testing_dataset()[1][0])


def test_experiment_from_its_spec(lidc_folder, no_tomosipo):
    experiment = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder)
    made = ct_experiments.Experiment.instantiate(through_json(to_spec(experiment)))
    assert type(made) is ct_experiments.LowDoseCTRecon
    assert made.param.noise_params.I0 == experiment.param.noise_params.I0


def test_the_noise_warning_says_how_to_get_it(lidc_folder, no_tomosipo):
    """Data set up without an experiment: the user's own noise."""
    data = lidc_dataset(lidc_folder)
    data.set_sinogram_transform(functools.partial(sinogram_add_noise, I0=500), seed=3)
    record = through_json(data_record(data))
    with pytest.warns(UserWarning, match=r"sinogram_add_noise, seed 3.*set_sinogram"):
        again = LIONDataset.instantiate(record)
    again.set_sinogram_transform(
        build(record.sinogram_transform), seed=record.sinogram_transform_seed
    )
    assert torch.equal(again[0][0], data[0][0])
    with pytest.raises(ValueError, match="not made by an Experiment"):
        ct_experiments.Experiment.instantiate(record)


def test_other_samples_warn(lidc_folder, no_tomosipo):
    record = through_json(data_record(lidc_dataset(lidc_folder)))
    record.samples.ids = record.samples.ids[:1] + ["LIDC-IDRI-0099/slice_3"]
    with pytest.warns(
        UserWarning, match="1 of the record's are missing, 1 are not in it"
    ):
        LIONDataset.instantiate(record)
    record.samples.ids = list(reversed(lidc_dataset(lidc_folder).sample_ids()))
    with pytest.warns(UserWarning, match="in another order"):
        LIONDataset.instantiate(record)


def test_check_files(lidc_folder, no_tomosipo):
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


def test_a_dataset_spec_alone(lidc_folder, no_tomosipo):
    data = lidc_dataset(lidc_folder)
    made = lidc.LIDC_IDRI.instantiate(through_json(to_spec(data)))
    assert made.sample_ids() == data.sample_ids()


def test_what_cannot_be_instantiated(lidc_folder, no_tomosipo):
    experiment = ct_experiments.LowDoseCTRecon(datafolder=lidc_folder)
    with pytest.raises(SpecError, match="not a LIONDataset"):
        LIONDataset.instantiate(through_json(to_spec(experiment)))
    with pytest.raises(ValueError, match="no data record"):
        LIONDataset.instantiate({"lion_format": 1})
    record = through_json(data_record(lidc_dataset(lidc_folder)))
    record.dataset = LIONParameter(unsaved="<MyDataset>", reason="a lambda")
    with pytest.raises(ValueError, match="only names its dataset"):
        LIONDataset.instantiate(record)


def test_read_record(lidc_folder, tmp_path, no_tomosipo):
    record = data_record(lidc_dataset(lidc_folder))
    expected = record.serialize()
    assert read_record(record).serialize() == expected
    assert read_record(saved_model(tmp_path, record)).serialize() == expected
    (tmp_path / "record.json").write_text(json.dumps(expected))
    assert read_record(tmp_path / "record.json").serialize() == expected
    assert record.samples.count == 2 and isinstance(record.dataset, dict)


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


def test_lidc_groups_are_the_patients(lidc_folder, no_tomosipo):
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


def test_detect_groups_are_the_physical_samples(tmp_path):
    """2DeteCT's splits are by physical sample: its records keep them."""
    training = data_record(fake_detect(tmp_path, "train"))
    test = data_record(fake_detect(tmp_path, "test"))
    assert training.samples.groups == [f"sample_{i}" for i in range(8)]
    assert test.samples.groups == ["sample_9"]
