# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Tests for LION.metrics."""

import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio as skim_psnr
from skimage.metrics import structural_similarity as skim_ssim

from LION.metrics.haarpsi import HAARPsi
from LION.metrics.psnr import PSNR
from LION.metrics.ssim import SSIM


@pytest.fixture
def pair():
    torch.manual_seed(0)
    target = torch.rand(3, 1, 32, 32)
    x = (target + 0.05 * torch.randn_like(target)).clamp(0, 1)
    return x, target


def _ref_psnr(x, t):
    x, t = x.squeeze().numpy(), t.squeeze().numpy()
    return skim_psnr(t, x, data_range=t.max() - t.min())


def _ref_ssim(x, t):
    x, t = x.squeeze().numpy(), t.squeeze().numpy()
    return skim_ssim(x, t, data_range=t.max() - t.min())


def test_batched_matches_whole_image_reference(pair):
    for batch_size in [1, 3]:
        for metric, ref in [(PSNR(), _ref_psnr), (SSIM(), _ref_ssim)]:
            x, target = pair[0][:batch_size], pair[1][:batch_size]
            vals = metric(x, target)
            assert vals.shape == (batch_size,)
            for i in range(batch_size):
                assert vals[i].item() == pytest.approx(ref(x[i], target[i]), rel=1e-5)

    for metric, ref in [(PSNR(), _ref_psnr), (SSIM(), _ref_ssim)]:
        x, target = pair[0][0], pair[1][0]
        assert metric(x, target, batched=False).item() == pytest.approx(
            ref(x, target), rel=1e-5
        )


@pytest.mark.parametrize("metric", [PSNR(), SSIM()])
def test_data_range(pair, metric):
    """data_range=None is the range of each target; a given one is used as is."""
    x, target = pair
    ranges = target.amax(dim=(1, 2, 3)) - target.amin(dim=(1, 2, 3))
    for i in range(len(x)):
        torch.testing.assert_close(
            metric(x[i : i + 1], target[i : i + 1], data_range=ranges[i].item()),
            metric(x[i : i + 1], target[i : i + 1]),
        )
    assert metric(x, target, data_range=2.0)[0] != metric(x, target)[0]


# --------------------------------------------------------------------------
# HAARPsi: same interface as PSNR and SSIM, values as piq's implementation


@pytest.fixture
def smooth_pair():
    torch.manual_seed(0)
    target = torch.nn.functional.avg_pool2d(torch.rand(3, 3, 64, 64), 2)
    x = (target + 0.05 * torch.randn_like(target)).clamp(0, 1)
    return x, target


@pytest.mark.parametrize("channels", [1, 3])
def test_haarpsi_matches_piq(smooth_pair, channels):
    piq = pytest.importorskip("piq")
    x, target = (v[:, :channels] for v in smooth_pair)
    torch.testing.assert_close(
        HAARPsi()(x, target, data_range=1.0),
        piq.haarpsi(x, target, data_range=1.0, reduction="none"),
        rtol=1e-3,
        atol=1e-4,
    )
