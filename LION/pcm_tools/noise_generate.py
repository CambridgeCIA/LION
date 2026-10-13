"""PCM noise generation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from statsmodels.tsa.arima_process import arma_generate_sample

from LION.pcm.noise import PCMNoiseParameter


def generate_dark_noise(
    num_measurements: int, dark_sigma: float | None, rng: np.random.Generator
) -> np.ndarray:
    r"""Generate dark noise.

    For measurement :math:`k`, the dark noise is

    .. math::

        e_k^{\mathrm{dark}} \sim \mathcal{N}(0, \sigma_{\mathrm{dark}}^2).

    If ``dark_sigma`` is ``None``, then :math:`e_k^{\mathrm{dark}} = 0`.

    Args:
        num_measurements: Number of measurements.
        dark_sigma: Standard deviation of the dark noise in canonical measurement units.
        rng: Random number generator.

    Returns:
        Absolute dark noise array.
    """
    if dark_sigma is None:
        return np.zeros(num_measurements)
    return rng.standard_normal(num_measurements) * dark_sigma


def generate_bias_noise(
    num_measurements: int,
    *,
    i_bias: float,  # I_0
    bias_gradient_sigma: float | None,
    bias_sigma: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Generate bias noise.

    First, draw one random bias gradient

    .. math::

        g = \sigma_g \eta,  \qquad  \eta \sim \mathcal{N}(0, 1),

    where :math:`\sigma_g` is ``bias_gradient_sigma``, :math:`\mathcal{N}(0, 1)` is the standard normal distribution.

    For measurement :math:`k = 1, \ldots, M`, the bias component is

    .. math::

        e_k^{\mathrm{bias}} = I_{\mathrm{bias}}(1 + k g) + I_{\mathrm{bias}} \sigma_b \varepsilon_k,
        \qquad  \varepsilon_k \sim \mathcal{N}(0, 1),

    where :math:`I_{\mathrm{bias}}` is ``i_bias`` and :math:`\sigma_b` is ``bias_sigma``.
    Note that some formulations use :math:`I_{0}` instead of :math:`I_{\mathrm{bias}}`.

    Thus, the first term gives a linearly drifting bias shared across the
    measurement trace, while the second term gives independent random fluctuations around that drift.

    If either ``bias_gradient_sigma`` or ``bias_sigma`` is ``None``,
    the returned bias component is zero.

    Args:
        num_measurements: Number of measurements.
        i_bias: Bias photocurrent.
        bias_gradient_sigma: Standard deviation of the fractional linear bias drift per measurement.
        bias_sigma: Standard deviation of the bias noise proportional to the signal.
        rng: Random number generator.

    Returns:
        Absolute bias noise array.
    """
    if bias_gradient_sigma is None or bias_sigma is None:
        return np.zeros(num_measurements)

    gradient = rng.standard_normal() * bias_gradient_sigma
    indices = np.arange(1, num_measurements + 1, dtype=np.float64)

    bias_noise = (
        i_bias * (1.0 + indices * gradient)
        + i_bias * rng.standard_normal(num_measurements) * bias_sigma
    )

    return bias_noise


def generate_beam_noise(
    num_measurements: int,
    *,
    beam_ar: Sequence[float] | None,
    beam_ma: Sequence[float] | None,
    beam_sigma: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Generate relative beam noise using an integrated ARMA model.

    Let :math:`a_1, \ldots, a_p` be the autoregressive coefficients in ``beam_ar`` and
    :math:`b_1, \ldots, b_q` the moving-average coefficients in ``beam_ma``. The ARMA process :math:`z_k` satisfies

    .. math::

        z_k = \sum_{i=1}^{p} a_i z_{k-i} + \varepsilon_k + \sum_{j=1}^{q} b_j \varepsilon_{k-j},

    with white-noise innovations

    .. math::

        \varepsilon_k \sim \mathcal{N}(0, \sigma_{\mathrm{beam}}^2),

    where :math:`\sigma_{\mathrm{beam}}` is ``beam_sigma``.

    The returned relative beam noise is the cumulative sum

    .. math::

        e_k^{\mathrm{rel}} = \sum_{\ell=1}^{k} z_\ell.

    Hence, the code first generates an ARMA trace and then integrates it using a cumulative sum.

    If any of ``beam_ar``, ``beam_ma``, or ``beam_sigma`` is ``None``, the returned relative beam noise is zero.

    Args:
        num_measurements: Number of measurements.
        beam_ar: Autoregressive coefficients of the beam noise model.
        beam_ma: Moving average coefficients of the beam noise model.
        beam_sigma: Standard deviation of the white noise input to the ARMA model.
        rng: Random number generator.

    Returns:
        Relative beam noise array.
    """
    if beam_ar is None or beam_ma is None or beam_sigma is None:
        return np.zeros(num_measurements)

    ar_poly = np.r_[1.0, -np.asarray(beam_ar, dtype=np.float64)]
    ma_poly = np.r_[1.0, np.asarray(beam_ma, dtype=np.float64)]

    arma_part = arma_generate_sample(
        ar=ar_poly,
        ma=ma_poly,
        nsample=num_measurements,
        scale=beam_sigma,  # type: ignore
        distrvs=rng.standard_normal,
    )
    rel_beam_noise = np.cumsum(arma_part)

    return rel_beam_noise


def generate_noise(
    num_measurements: int,
    i_bias: float,
    noise_params: PCMNoiseParameter,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Generate the components of the synthetic PCM noise model.

    The function generates three noise components,

    .. math::

        \mathbf{e}^{\mathrm{dark}},
        \qquad
        \mathbf{e}^{\mathrm{bias}},
        \qquad
        \mathbf{e}^{\mathrm{rel}},

    where

    .. math::

        e_k^{\mathrm{dark}}
        =
        \sigma_{\mathrm{dark}} \varepsilon_k,

    .. math::

        e_k^{\mathrm{bias}}
        =
        I_{\mathrm{bias}}(1 + k g)
        +
        I_{\mathrm{bias}} \sigma_b \xi_k,

    and :math:`\mathbf{e}^{\mathrm{rel}}` is obtained by cumulatively
    summing an ARMA process. Here
    :math:`\varepsilon_k, \xi_k \sim \mathcal{N}(0, 1)` and

    .. math::

        g \sim \mathcal{N}(0, \sigma_g^2).

    Args:
        num_measurements: Number of measurements.
        i_bias: Bias photocurrent.
        noise_params: Noise model parameters.
        rng: Random number generator.

    Returns:
        Dark noise, bias noise, and relative beam noise arrays.
    """
    # ar_poly = np.r_[1.0, -np.asarray(noise_params.beam_ar, dtype=np.float64)]
    # ma_poly = np.r_[1.0, np.asarray(noise_params.beam_ma, dtype=np.float64)]

    # arma_part = arma_generate_sample(
    #     ar=ar_poly,
    #     ma=ma_poly,
    #     nsample=num_measurements,
    #     scale=noise_params.beam_sigma,  # type: ignore
    #     distrvs=rng.standard_normal,
    # )
    # rel_beam_noise = np.cumsum(arma_part)

    # gradient = rng.standard_normal() * noise_params.bias_gradient_sigma
    # idx = np.arange(1, num_measurements + 1, dtype=np.float64)

    # bias_noise = (
    #     i_bias * (1.0 + idx * gradient)
    #     + i_bias * rng.standard_normal(num_measurements) * noise_params.bias_sigma
    # )

    # dark_noise = rng.standard_normal(num_measurements) * noise_params.dark_sigma

    rel_beam_noise = generate_beam_noise(
        num_measurements,
        beam_ar=noise_params.beam_ar,
        beam_ma=noise_params.beam_ma,
        beam_sigma=noise_params.beam_sigma,
        rng=rng,
    )
    bias_noise = generate_bias_noise(
        num_measurements,
        i_bias=i_bias,
        bias_gradient_sigma=noise_params.bias_gradient_sigma,
        bias_sigma=noise_params.bias_sigma,
        rng=rng,
    )
    dark_noise = generate_dark_noise(
        num_measurements, dark_sigma=noise_params.dark_sigma, rng=rng
    )

    return dark_noise, bias_noise, rel_beam_noise


def add_noise(
    *,
    y_clean_plus: torch.Tensor,
    y_clean_minus: torch.Tensor,
    e_abs_trace: torch.Tensor,
    e_rel: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Add synthetic noise to split-Hadamard measurements.

    Let the clean positive and negative measurements be

    .. math::

        y_k^+,  \qquad  y_k^-,  \qquad  k = 1, \ldots, n.

    They are interleaved into a single trace

    .. math::

        \mathbf{s} = (y_1^+, y_1^-, y_2^+, y_2^-, \ldots, y_n^+, y_n^-)^\top.

    Within each positive-negative pair, a random permutation :math:`\pi` either keeps or swaps the two noise samples.
    Thus, for each pair,

    .. math::

        \bigl(\pi(2k-1), \pi(2k)\bigr) \in \{(2k-1, 2k), (2k, 2k-1)\}.

    The noise added to trace sample :math:`j` is

    .. math::

        n_j = e^{\mathrm{abs}}_{\pi(j)} + s_j e^{\mathrm{rel}}_{\pi(j)}.

    Therefore the noisy trace is

    .. math::

        \widetilde{s}_j = s_j + n_j = s_j\left(1 + e^{\mathrm{rel}}_{\pi(j)}\right) + e^{\mathrm{abs}}_{\pi(j)}.

    Finally, the interleaved trace is split back into positive and negative
    measurements,

    .. math::

        \widetilde{y}_k^+ = \widetilde{s}_{2k-1},  \qquad  \widetilde{y}_k^- = \widetilde{s}_{2k}.

    Args:
        y_clean_plus: Positive-pattern noiseless measurements.
        y_clean_minus: Negative-pattern noiseless measurements.
        e_abs_trace: Absolute noise component.
        e_rel: Relative noise component.
        rng: Random number generator.

    Returns:
        Noisy positive measurements, noisy negative measurements, and interleaved noisy measurement trace.
    """
    n = int(y_clean_plus.numel())
    device = e_abs_trace.device

    i_swap = generate_pairwise_swap_indices(num_pairs=n, rng=rng, device=device)

    interleaved_signal = torch.stack([y_clean_plus, y_clean_minus], dim=1).reshape(-1)

    i_noise = e_abs_trace[i_swap] + interleaved_signal * e_rel[i_swap]
    noisy_interleaved_trace = interleaved_signal + i_noise

    i_noise_matrix = i_noise.reshape(n, 2)

    y_plus_noise = y_clean_plus + i_noise_matrix[:, 0]
    y_minus_noise = y_clean_minus + i_noise_matrix[:, 1]

    return y_plus_noise, y_minus_noise, noisy_interleaved_trace


def generate_pairwise_swap_indices(
    num_pairs: int, rng: np.random.Generator, device: torch.device
) -> torch.Tensor:
    """Generate a random pairwise permutation of interleaved indices.

    For each pair

    .. math::

        (2k, 2k + 1),

    independently return either

    .. math::

        (2k, 2k + 1)

    or

    .. math::

        (2k + 1, 2k).

    Args:
        num_pairs: Number of index pairs.
        rng: Random number generator.
        device: Device of the returned tensor.

    Returns:
        Pairwise-permuted indices of length ``2 * num_pairs``.
    """
    # index list [0, ..., 2*num_pairs - 1]
    i_swap = torch.arange(2 * num_pairs, dtype=torch.long, device=device)
    # random binary list of length num_pairs, where 0 means "don't swap" and 1 means "swap"
    do_swap = torch.as_tensor(
        rng.integers(0, 2, size=num_pairs), dtype=torch.long, device=device
    )
    # Pairs corresponding to 0s remain unchanged,
    # pairs corresponding to 1s are swapped.
    i_swap[0::2] += do_swap
    i_swap[1::2] -= do_swap
    return i_swap
