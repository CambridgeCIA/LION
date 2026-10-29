"""Reproducible random numbers (LION.utils.rng) and the simulated noise of the
data loaders: one noisy measurement per slice when a seed is given."""

import os
import tempfile

# the datasets' default folders are in LION_DATA_PATH
os.environ.setdefault("LION_DATA_PATH", tempfile.gettempdir())

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import LION.data_loaders.LIDC_IDRI as lidc  # noqa: E402
import LION.experiments.ct_experiments as ct_experiments  # noqa: E402
from LION.CTtools.ct_utils import sinogram_add_noise  # noqa: E402
from LION.data_loaders.deteCT import deteCT  # noqa: E402
from LION.utils.rng import seed_from_parts, seeded  # noqa: E402
from pytests.test_experiments import fake_detect  # noqa: E402
from pytests.test_lidc_resampling import fake_lidc  # noqa: E402

# --------------------------------------------------------------------------
# seed_from_parts and seeded


def test_seed_from_parts_is_stable():
    """These values fix every seeded noisy dataset: if they change, results do."""
    assert seed_from_parts(0, "LIDC-IDRI", "LIDC-IDRI-0001", 3) == 2950840417
    assert seed_from_parts("0", "LIDC-IDRI", "LIDC-IDRI-0001", "3") == 2950840417
    assert seed_from_parts(1, "LIDC-IDRI", "LIDC-IDRI-0001", 3) != 2950840417


def test_seeded_gives_the_same_numbers():
    with seeded(0, "a", 1):
        first = torch.rand(5)
    with seeded(0, "a", 1):
        second = torch.rand(5)
    with seeded(0, "a", 2):
        other = torch.rand(5)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_seeded_restores_the_global_random_state():
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    with seeded(0, "a"):
        torch.rand(100)
    assert torch.equal(torch.rand(3), expected)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_seeded_on_cuda():
    device = torch.device("cuda")
    with seeded(0, "a", device=device):
        first = torch.rand(5, device=device)
    with seeded(0, "a", device=device):
        second = torch.rand(5, device=device)
    assert torch.equal(first, second)


# --------------------------------------------------------------------------
# LIDC-IDRI: one noisy measurement per slice


def noise(sinogram):
    return sinogram_add_noise(sinogram, I0=2100)


def stand_in_projection(image):
    """A clean sinogram that depends on the image, without tomosipo."""
    return (
        (image.float().mean() + 2.0) * torch.ones(1, 90, 128, device=image.device) * 100
    )


@pytest.fixture
def no_tomosipo(monkeypatch):
    monkeypatch.setattr(lidc.ct, "make_operator", lambda geometry: stand_in_projection)


@pytest.fixture
def lidc_data(tmp_path, no_tomosipo):
    """A LIDC-IDRI loader on fake slices, each slice different."""
    fake_lidc(tmp_path, range(1, 11))
    for k, path in enumerate(sorted(tmp_path.glob("LIDC-IDRI-*/slice_*.npy"))):
        np.save(path, np.full((512, 512), -500 + k, np.int16))
    params = lidc.LIDC_IDRI.default_parameters(
        geometry=lidc.LIDC_IDRI.default_geometry(), task="reconstruction"
    )
    params.folder = tmp_path
    return lidc.LIDC_IDRI("train", parameters=params)


def read_all(data, seed=1, order=None, workers=0):
    """The noisy sinograms of an epoch, by index."""
    if seed is not None:
        torch.manual_seed(seed)  # as a training script does, once
    if workers:
        loader = DataLoader(data, batch_size=1, num_workers=workers)
        return {i: sino[0] for i, (sino, _) in enumerate(loader)}
    order = range(len(data)) if order is None else order
    return {i: data[i][0] for i in order}


def same(a, b):
    return all(torch.equal(a[i], b[i]) for i in a)


def test_seeded_slices_have_one_noisy_measurement(lidc_data):
    # on the CPU: DataLoader workers can't use CUDA once the main process has
    # (LION's loaders return data on their device)
    lidc_data.device = torch.device("cpu")
    lidc_data.set_sinogram_transform(noise, seed=0)
    first = read_all(lidc_data)
    assert same(first, read_all(lidc_data, seed=None))  # the next epoch
    assert same(first, read_all(lidc_data, seed=2))  # another run's seed
    assert same(first, read_all(lidc_data, order=reversed(range(len(lidc_data)))))
    assert same(first, read_all(lidc_data, workers=2))
    # and the noise is there, and different for each slice
    clean = lidc_data.compute_clean_sinogram(lidc_data[0][1])
    assert not torch.equal(first[0], clean)
    assert not torch.equal(
        first[0] - clean, first[1] - lidc_data.compute_clean_sinogram(lidc_data[1][1])
    )


def test_without_seed_the_noise_is_fresh_each_time(lidc_data):
    lidc_data.set_sinogram_transform(noise)
    first = read_all(lidc_data)
    assert not same(first, read_all(lidc_data, seed=None))


def test_training_randomness_stays_fresh(lidc_data):
    """A solver's own random numbers (e.g. noise added in training) are new
    every time, even for the same, seeded, sample."""
    lidc_data.set_sinogram_transform(noise, seed=0)
    torch.manual_seed(3)
    first_sample, _ = lidc_data[0]
    first_extra = torch.randn(4)
    second_sample, _ = lidc_data[0]
    second_extra = torch.randn(4)
    assert torch.equal(first_sample, second_sample)
    assert not torch.equal(first_extra, second_extra)


def test_noise_follows_the_slice_not_its_index(lidc_data):
    """A slice keeps its noise when the split or the slice selection moves it."""
    lidc_data.set_sinogram_transform(noise, seed=0)
    noisy = lidc_data[1][0]
    lidc_data.slice_index_to_patient_id_list = lidc_data.slice_index_to_patient_id_list[
        1:
    ]
    for patient in lidc_data.patient_id_to_first_index_dict:
        lidc_data.patient_id_to_first_index_dict[patient] -= 1
    assert torch.equal(lidc_data[0][0], noisy)


# --------------------------------------------------------------------------
# The experiments give the loader their noise seed


@pytest.mark.parametrize(
    "cls",
    [ct_experiments.LowDoseCTRecon, ct_experiments.SparseAngleExtremeLowDoseCTRecon],
)
def test_experiments_seed_the_noise(cls, tmp_path, no_tomosipo):
    fake_lidc(tmp_path, range(1, 11))
    params = cls.default_parameters()
    assert params.noise_params.seed == 0
    experiment = cls(experiment_params=params, datafolder=tmp_path)
    assert experiment.get_training_dataset().sinogram_transform_seed == 0


def test_experiment_parameters_without_seed_keep_fresh_noise(tmp_path, no_tomosipo):
    """Parameter files saved before the seed existed."""
    fake_lidc(tmp_path, range(1, 11))
    params = ct_experiments.LowDoseCTRecon.default_parameters()
    del params.noise_params.seed
    experiment = ct_experiments.LowDoseCTRecon(
        experiment_params=params, datafolder=tmp_path
    )
    assert experiment.get_training_dataset().sinogram_transform_seed is None


# --------------------------------------------------------------------------
# 2DeteCT added noise


@pytest.fixture
def detect_data(tmp_path):
    data = fake_detect(tmp_path, "train")
    geometry = deteCT.get_default_geometry()
    n_angles, n_det = len(geometry.angles), geometry.detector_shape[-1]
    for k in range(3):
        folder = tmp_path / f"slice{k:05d}" / "mode2"
        folder.mkdir(parents=True)
        np.save(folder / "sinogram.npy", np.full((n_angles, n_det), 0.5, np.float32))
        np.save(folder / "flat.npy", np.ones((1, n_det), np.float32))
        np.save(folder / "dark.npy", np.zeros((1, n_det), np.float32))
    data.add_noise = True
    return data


@pytest.mark.filterwarnings("ignore:You are not using any detector query")
def test_detect_added_noise_is_seeded(detect_data):
    detect_data.noise_params = deteCT.noise_parameters()
    assert detect_data.noise_params.seed == 0
    read = lambda k: detect_data.__load_and_preprocess_sinogram__(k, "mode2")
    torch.manual_seed(1)
    first = [read(k) for k in range(3)]
    torch.manual_seed(2)
    assert all(torch.equal(a, read(k)) for k, a in enumerate(first))
    assert not torch.equal(first[0], first[2])  # same input, other slice


@pytest.mark.filterwarnings("ignore:You are not using any detector query")
def test_detect_added_noise_without_seed_is_fresh(detect_data):
    detect_data.noise_params = deteCT.noise_parameters()
    del detect_data.noise_params.seed  # a parameter file saved before
    read = lambda: detect_data.__load_and_preprocess_sinogram__(0, "mode2")
    assert not torch.equal(read(), read())
