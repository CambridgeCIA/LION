"""UNet blocks: conv-norm-relu twice; old single-norm checkpoints still load."""

import pytest
import torch

from LION.models.CNNs.UNets.Unet import UNet


def _norms(block):
    return [
        name for name, m in block.named_children() if isinstance(m, torch.nn.GroupNorm)
    ]


def small_params(**overrides):
    params = UNet.default_parameters()
    params.baseDim = 4
    params.noGrp = 2
    for k, v in overrides.items():
        setattr(params, k, v)
    return params


def test_default_blocks_have_two_norms():
    model = UNet(small_params())
    for block in (model.encoder1, model.base, model.decoder1):
        assert len(_norms(block)) == 2
    assert model(torch.rand(2, 1, 32, 32)).shape == (2, 1, 32, 32)


def test_parameters_without_the_option_keep_old_architecture():
    params = small_params()
    del params.norm_after_second_conv  # as in parameter files saved before it existed
    model = UNet(params)
    assert _norms(model.encoder1) == ["enc1gn1"]


def test_old_state_dict_loads_strictly():
    old = small_params(norm_after_second_conv=False)
    state = UNet(old).state_dict()
    assert not any("gn2" in k for k in state)
    UNet(old).load_state_dict(state, strict=True)
