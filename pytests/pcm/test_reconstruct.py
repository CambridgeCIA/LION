import os
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import LION.pcm.reconstruct as reconstruct
import torch


class RecordingDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None
        self.last_sigma = None

    def forward(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        self.last_input = x
        self.last_sigma = sigma
        return x + 1.0


def test_pnp_admm_recon_fn_passes_denoiser_input_unchanged(monkeypatch):
    denoiser_input = torch.randn(3, 1, 4, 5)
    denoiser = RecordingDenoiser()
    config = SimpleNamespace(sigma=0.05)

    class FakePnPADMM:
        def __init__(self, physics, denoiser):
            self.denoiser = denoiser

        def admm_algorithm(self, measurement, config, prog_bar=None):
            return self.denoiser(denoiser_input, sigma=config.sigma)

    monkeypatch.setattr(reconstruct, "PnPADMM", FakePnPADMM)

    recon_fn = reconstruct.PnPADMMReconFn(
        config=config,
        denoiser=denoiser,
    )

    recon_input = reconstruct.ReconInput(
        op=object(),
        y=torch.randn(3, 1, 8),
        x0=torch.randn(3, 1, 4, 5),
        i0=None,
    )

    output = recon_fn(recon_input)

    assert denoiser.last_input is denoiser_input
    assert denoiser.last_input.shape == (3, 1, 4, 5)
    assert denoiser.last_sigma == 0.05
    torch.testing.assert_close(output, denoiser_input + 1.0)


def test_gspnp_recon_fn_passes_denoiser_directly(monkeypatch):
    denoiser = RecordingDenoiser()
    config = SimpleNamespace(uses_deepinv=False)

    received = {}

    class FakeGSPnP:
        def __init__(self, physics, denoiser):
            received["physics"] = physics
            received["denoiser"] = denoiser

        def reconstruct_sample(self, measurement, **kwargs):
            received["measurement"] = measurement
            received["config"] = kwargs["config"]
            return torch.ones(3, 1, 4, 5)

    monkeypatch.setattr(reconstruct, "GSPnP", FakeGSPnP)

    op = object()
    measurement = torch.randn(3, 1, 8)

    recon_fn = reconstruct.GSPnPReconFn(
        config=config,
        denoiser=denoiser,
    )

    output = recon_fn(
        reconstruct.ReconInput(
            op=op,
            y=measurement,
            x0=torch.randn(3, 1, 4, 5),
            i0=None,
        )
    )

    assert received["physics"] is op
    assert received["denoiser"] is denoiser
    assert received["measurement"] is measurement
    assert received["config"] is config
    assert output.shape == (3, 1, 4, 5)


def test_gspnp_deepinv_recon_fn_passes_denoiser_directly(monkeypatch):
    denoiser = RecordingDenoiser()
    config = SimpleNamespace(uses_deepinv=True)

    received = {}

    class FakePhysics:
        def __init__(self, op):
            self.op = op

    class FakeGSPnPDeepInv:
        def __init__(self, physics, denoiser):
            received["physics"] = physics
            received["denoiser"] = denoiser

        def reconstruct_sample(self, measurement, **kwargs):
            received["measurement"] = measurement
            received["config"] = kwargs["config"]
            return torch.ones(3, 1, 4, 5)

    monkeypatch.setattr(reconstruct, "DeepInvLinearPhysics", FakePhysics)
    monkeypatch.setattr(reconstruct, "GSPnPDeepInv", FakeGSPnPDeepInv)

    op = object()
    measurement = torch.randn(3, 1, 8)

    recon_fn = reconstruct.GSPnPReconFn(
        config=config,
        denoiser=denoiser,
    )

    output = recon_fn(
        reconstruct.ReconInput(
            op=op,
            y=measurement,
            x0=torch.randn(3, 1, 4, 5),
            i0=None,
        )
    )

    assert isinstance(received["physics"], FakePhysics)
    assert received["physics"].op is op
    assert received["denoiser"] is denoiser
    assert received["measurement"] is measurement
    assert received["config"] is config
    assert output.shape == (3, 1, 4, 5)
