"""Gradient-step denoisers need autograd in forward(); they must work when
called under torch.no_grad() (validation, testing, PnP loops)."""

import pytest
import torch

from LION.models.PnP.gradient_step_denoiser import GSD
from LION.models.PnP.gs_drunet import GSDRUNet


def small_gsd():
    params = GSD.default_parameters()
    params.backbone_params.int_channels = 4
    params.backbone_params.n_blocks = 1
    return GSD(params)


def small_gsdrunet():
    params = GSDRUNet.default_parameters()
    params.int_channels = 4
    params.n_blocks = 1
    return GSDRUNet(params)


@pytest.mark.parametrize("make_model", [small_gsd, small_gsdrunet])
def test_same_output_with_and_without_no_grad(make_model):
    torch.manual_seed(0)
    model = make_model().eval()
    x = torch.rand(2, 1, 16, 16)
    expected = model(x)
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, expected.detach())
    assert not out.requires_grad  # nothing leaks out of no_grad


@pytest.mark.parametrize("make_model", [small_gsd, small_gsdrunet])
def test_training_still_backpropagates_to_weights(make_model):
    model = make_model().train()
    loss = model(torch.rand(2, 1, 16, 16)).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
