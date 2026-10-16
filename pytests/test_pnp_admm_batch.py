import numpy as np
import torch
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm_tools.pcm_prox import PCMProxParameter
from LION.reconstructors.BasePnP import DenoiserParameter
from LION.reconstructors.PnPADMM import PnPADMM, PnPADMMParameter


def identity_denoiser(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return x


def test_pnp_admm_batch_matches_stacked_single_reconstructions():
    torch.manual_seed(0)

    op = PhotocurrentMapOp(j_order=2, s_indices=np.array([0, 1, 4, 7, 10, 13]))
    config = PnPADMMParameter(
        denoiser=DenoiserParameter(name="drunet"),
        iters=4,
        prox=PCMProxParameter(rho=0.1),
        sigma=0.05,
    )
    pnp = PnPADMM(physics=op, denoiser=identity_denoiser)

    y = torch.randn(3, *op.range_shape, dtype=torch.float64)

    actual = pnp.admm_algorithm(y, config)
    expected = torch.stack(
        [pnp.admm_algorithm(y[i], config) for i in range(y.shape[0])]
    )

    assert actual.shape == (3, *op.domain_shape)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
