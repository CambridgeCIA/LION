import importlib

import torch


class FakeOperator:
    domain_shape = (1, 2, 2)


def test_tv_min_passes_each_sinogram_item_to_backend(monkeypatch):
    tv_module = importlib.import_module("LION.classical_algorithms.tv_min")
    seen = []

    def fake_tv_min2d(
        op,
        y,
        lam,
        num_iterations=500,
        L=None,
        non_negativity=False,
        progress_bar=False,
        callbacks=(),
    ):
        del op, lam, num_iterations, L, non_negativity, progress_bar, callbacks
        seen.append(y.clone())
        return torch.full(FakeOperator.domain_shape, float(len(seen)))

    monkeypatch.setattr(tv_module, "ts_tv_min", fake_tv_min2d)
    sinogram = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)

    recon = tv_module.tv_min(sinogram, FakeOperator(), lam=0.001)

    assert recon.shape == (2, 1, 2, 2)
    assert torch.equal(seen[0], sinogram[0])
    assert torch.equal(seen[1], sinogram[1])
    assert torch.all(recon[0] == 1)
    assert torch.all(recon[1] == 2)
