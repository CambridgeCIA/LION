"""Tests for repeated PCM acquisition averaging."""

from __future__ import annotations

from dataclasses import replace

import LION.pcm.simulation as simulation
import numpy as np
import torch
from LION.pcm.noise import PCMNoiseSimulation


def test_repeat_acquisitions_one_preserves_single_acquisition(monkeypatch) -> None:
    """One repeated acquisition is exactly the existing single-acquisition behavior."""
    calls = iter([1.0])

    monkeypatch.setattr(
        simulation,
        "generate_noise",
        lambda **kwargs: (
            np.full(kwargs["num_measurements"], next(calls)),
            np.zeros(kwargs["num_measurements"]),
            np.zeros(kwargs["num_measurements"]),
        ),
    )

    def fake_add_noise(*, y_clean_plus, y_clean_minus, e_abs_trace, e_rel, rng):
        n = y_clean_plus.numel()
        noise = e_abs_trace.reshape(n, 2)
        return y_clean_plus + noise[:, 0], y_clean_minus + noise[:, 1], torch.empty(0)

    monkeypatch.setattr(simulation, "add_noise", fake_add_noise)

    noise = PCMNoiseSimulation.npl_default(
        current_to_model_scale=1.0,
        j_order=1,
        n_repeat_acquisitions=1,
    )
    target = torch.tensor([[[[0.0, 1.0], [0.5, 0.25]]]], dtype=torch.float64)

    acquisition = simulation.simulate_pcm_acquisition(
        target,
        j_order=1,
        sampled_indices=np.arange(4),
        noise=noise,
        rng=np.random.default_rng(0),
        device="cpu",
    )

    clean = simulation.simulate_pcm_acquisition(
        target,
        j_order=1,
        sampled_indices=np.arange(4),
        noise=None,
        rng=np.random.default_rng(0),
        device="cpu",
    )

    torch.testing.assert_close(acquisition.measurement, clean.measurement)


def test_repeat_acquisitions_are_averaged(monkeypatch) -> None:
    """Repeated noisy acquisitions are averaged before reconstruction."""
    values = iter([0.0, 2.0, 4.0])

    def fake_generate_noise(**kwargs):
        value = next(values)
        absolute = np.tile([value, 0.0], kwargs["num_measurements"] // 2)
        return (
            absolute,
            np.zeros(kwargs["num_measurements"]),
            np.zeros(kwargs["num_measurements"]),
        )

    monkeypatch.setattr(simulation, "generate_noise", fake_generate_noise)

    def fake_add_noise(*, y_clean_plus, y_clean_minus, e_abs_trace, e_rel, rng):
        n = y_clean_plus.numel()
        noise = e_abs_trace.reshape(n, 2)
        return y_clean_plus + noise[:, 0], y_clean_minus + noise[:, 1], torch.empty(0)

    monkeypatch.setattr(simulation, "add_noise", fake_add_noise)

    noise = PCMNoiseSimulation.npl_default(
        current_to_model_scale=1.0,
        j_order=1,
        n_repeat_acquisitions=3,
    )
    target = torch.tensor([[[[0.0, 1.0], [0.5, 0.25]]]], dtype=torch.float64)

    acquisition = simulation.simulate_pcm_acquisition(
        target,
        j_order=1,
        sampled_indices=np.arange(4),
        noise=noise,
        rng=np.random.default_rng(0),
        device="cpu",
    )

    clean = simulation.simulate_pcm_acquisition(
        target,
        j_order=1,
        sampled_indices=np.arange(4),
        noise=None,
        rng=np.random.default_rng(0),
        device="cpu",
    )

    expected_added_noise = torch.full_like(clean.measurement, 2.0)
    torch.testing.assert_close(
        acquisition.measurement,
        clean.measurement + expected_added_noise,
    )
