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


@pytest.mark.parametrize("metric, ref", [(PSNR(), _ref_psnr), (SSIM(), _ref_ssim)])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_batched_matches_whole_image_reference(pair, metric, ref, batch_size):
    x, target = pair[0][:batch_size], pair[1][:batch_size]
    vals = metric(x, target)
    assert vals.shape == (batch_size,)
    for i in range(batch_size):
        assert vals[i].item() == pytest.approx(ref(x[i], target[i]), rel=1e-5)


@pytest.mark.parametrize("metric, ref", [(PSNR(), _ref_psnr), (SSIM(), _ref_ssim)])
def test_unbatched(pair, metric, ref):
    x, target = pair[0][0], pair[1][0]
    assert metric(x, target, batched=False).item() == pytest.approx(
        ref(x, target), rel=1e-5
    )


def test_reduce_mean(pair):
    x, target = pair
    torch.testing.assert_close(
        PSNR()(x, target, reduce="mean"), PSNR()(x, target).mean()
    )


def test_haarpsi_can_be_called(pair):
    x, target = pair
    metric = HAARPsi()
    same = metric(target, target)
    noisy = metric(target, x)
    score = lambda r: (r[0] if isinstance(r, tuple) else r).mean()
    assert score(same) == pytest.approx(1.0, abs=1e-4)
    assert score(noisy) < score(same)
