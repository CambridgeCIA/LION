from types import SimpleNamespace

import LION.pcm.simulation as simulation
import numpy as np
import torch
from LION.pcm.noise import PCMNoiseParameter
from LION.pcm_tools.noise_generate import generate_noise


def test_sample_and_simulate_samples_indices_once(monkeypatch):
    sampled_indices = np.array([0, 2], dtype=np.int64)
    num_sampling_calls = 0
    captured_indices = None

    def fake_sample_indices(**kwargs):
        nonlocal num_sampling_calls
        num_sampling_calls += 1
        return sampled_indices

    sentinel = object()

    def fake_simulate_pcm_acquisition(target_image, **kwargs):
        nonlocal captured_indices
        captured_indices = kwargs["sampled_indices"]
        return sentinel

    monkeypatch.setattr(simulation, "sample_indices", fake_sample_indices)
    monkeypatch.setattr(
        simulation, "simulate_pcm_acquisition", fake_simulate_pcm_acquisition
    )

    noise = SimpleNamespace(n_repeat_acquisitions=5)
    sampling = SimpleNamespace(randomising_scheme="multilevel")

    result = simulation.sample_and_simulate_pcm_acquisition(
        torch.ones(1, 1, 2, 2),
        j_order=1,
        sampling=sampling,
        sampling_ratio=0.5,
        coarse_j=0,
        noise=noise,
        rng=np.random.default_rng(0),
        device="cpu",
    )

    assert result is sentinel
    assert num_sampling_calls == 1
    np.testing.assert_array_equal(captured_indices, sampled_indices)


def test_repeat_acquisitions_draw_noise_each_time_and_average(monkeypatch):
    sampled_indices = np.array([0, 1], dtype=np.int64)
    operator_indices = []
    num_noise_draws = 0

    class FakePhotocurrentMapOp:
        normalization_factor = 1.0

        def __init__(self, *, j_order, s_indices, normalized, device):
            operator_indices.append(np.array(s_indices, copy=True))

    y_plus_clean = torch.tensor([[[1.0, 2.0]]])
    y_minus_clean = torch.zeros_like(y_plus_clean)

    def fake_sample_split_measurements(target_image, op):
        return y_plus_clean, y_minus_clean

    def fake_generate_noise(*, num_measurements, i_bias, noise_params, rng):
        nonlocal num_noise_draws
        num_noise_draws += 1
        value = float(num_noise_draws)
        return (
            np.full(num_measurements, value),
            np.zeros(num_measurements),
            np.zeros(num_measurements),
        )

    def fake_add_noise(*, y_clean_plus, y_clean_minus, e_abs_trace, e_rel, rng):
        noise_value = e_abs_trace[0]
        return (
            y_clean_plus + noise_value,
            y_clean_minus,
            torch.empty(0),
        )

    monkeypatch.setattr(simulation, "PhotocurrentMapOp", FakePhotocurrentMapOp)
    monkeypatch.setattr(
        simulation, "_sample_split_measurements", fake_sample_split_measurements
    )
    monkeypatch.setattr(simulation, "generate_noise", fake_generate_noise)
    monkeypatch.setattr(simulation, "add_noise", fake_add_noise)

    noise = SimpleNamespace(
        n_repeat_acquisitions=3,
        signal_to_bias_ratio=0.3,
        noise_params=object(),
    )

    acquisition = simulation.simulate_pcm_acquisition(
        torch.ones(1, 1, 2, 2),
        j_order=1,
        sampled_indices=sampled_indices,
        noise=noise,
        rng=np.random.default_rng(0),
        device="cpu",
    )

    assert num_noise_draws == 3
    assert len(operator_indices) == 1
    np.testing.assert_array_equal(operator_indices[0], sampled_indices)

    # The three fake noise realizations are +1, +2, +3, so their mean is +2.
    expected_measurement = y_plus_clean + 2.0
    torch.testing.assert_close(acquisition.measurement, expected_measurement)


def test_generate_noise_draws_different_realizations_from_same_rng():
    noise_params = PCMNoiseParameter(
        beam_ar=None,
        beam_ma=None,
        beam_sigma=None,
        bias_sigma=None,
        bias_gradient_sigma=None,
        dark_sigma=1.0,
    )
    rng = np.random.default_rng(0)

    first_dark_noise, _, _ = generate_noise(
        num_measurements=8,
        i_bias=1.0,
        noise_params=noise_params,
        rng=rng,
    )
    second_dark_noise, _, _ = generate_noise(
        num_measurements=8,
        i_bias=1.0,
        noise_params=noise_params,
        rng=rng,
    )

    assert not np.array_equal(first_dark_noise, second_dark_noise)
