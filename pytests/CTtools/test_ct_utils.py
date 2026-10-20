"""Tests for LION.CTtools.ct_utils that run on CPU."""

import numpy as np
import pytest
import torch

from LION.CTtools.ct_utils import sinogram_add_noise


@pytest.fixture
def clean_sino():
    torch.manual_seed(0)
    return 2.0 * torch.rand(1, 30, 40)


def test_sinogram_add_noise_on_cpu_tensor(clean_sino):
    noisy = sinogram_add_noise(clean_sino, I0=1000, sigma=5)
    assert noisy.device == clean_sino.device
    assert noisy.shape == clean_sino.shape
    assert torch.isfinite(noisy).all() and (noisy >= 0).all()
    assert not torch.equal(noisy, clean_sino)


def test_sinogram_add_noise_numpy_in_numpy_out(clean_sino):
    noisy = sinogram_add_noise(clean_sino.numpy(), I0=1000, sigma=5)
    assert isinstance(noisy, np.ndarray)
    assert noisy.shape == tuple(clean_sino.shape)


def test_less_noise_with_more_photons(clean_sino):
    torch.manual_seed(1)
    low = sinogram_add_noise(clean_sino, I0=1e3, sigma=0)
    high = sinogram_add_noise(clean_sino, I0=1e6, sigma=0)
    assert (high - clean_sino).abs().mean() < (low - clean_sino).abs().mean()
