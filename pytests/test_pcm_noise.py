from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from LION.data_loaders.pcm.data_utils import read_pcm_measurement_data_from_txt
from LION.experiments.pcm_experiments import PCMNoiseParameter
from LION.pcm_tools.hadamard_matrix import hadamard_matrix, hadamard_row_plus_minus
from LION.pcm_tools.measure import measure_split_hadamard_transform
from LION.pcm_tools.noise_generate import generate_noise
from LION.pcm_tools.noise_metrics import compute_metrics_hybrid
from LION.pcm_tools.noise_reconstruct import PCMReconstructor
from LION.pcm_tools.noise_utils import generate_order_sorting_matrix
from LION.utils.math import is_power_of_two, next_power_of_two
from LION.utils.shape import pad_to_powers_of_two_torch, torch_reshape_fortran_2d


def explicit_hadamard(n: int) -> np.ndarray:
    rows = np.arange(n, dtype=np.uint32)[:, None]
    cols = np.arange(n, dtype=np.uint32)[None, :]
    values = np.bitwise_and(rows, cols)

    parity = np.zeros((n, n), dtype=np.int8)
    for bit in range(int(np.log2(n))):
        parity ^= ((values >> np.uint32(bit)) & np.uint32(1)).astype(np.int8)

    return (1 - 2 * parity).astype(np.float64)


def matlab_vec(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float64).reshape(-1, order="F")


def test_is_power_of_two() -> None:
    assert is_power_of_two(1)
    assert is_power_of_two(2)
    assert is_power_of_two(8)
    assert not is_power_of_two(0)
    assert not is_power_of_two(3)
    assert not is_power_of_two(10)


def test_next_power_of_two() -> None:
    assert next_power_of_two(1) == 1
    assert next_power_of_two(2) == 2
    assert next_power_of_two(3) == 4
    assert next_power_of_two(9) == 16

    with pytest.raises(ValueError):
        next_power_of_two(0)


def test_hadamard_rows_are_plus_minus_one() -> None:
    rows = hadamard_matrix(order=2)

    assert set(np.unique(rows).tolist()) == {-1.0, 1.0}


def test_hadamard_rows_match_reference_matrix() -> None:
    expected = np.array(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=np.float64,
    )

    got = hadamard_matrix(order=2)

    np.testing.assert_array_equal(got, expected)


def test_single_hadamard_row_match_reference_matrix() -> None:
    expected = np.array(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ]
    )

    row_index = 2
    order = 2  # matrix is 2^2 x 2^2 = 4 x 4

    got = hadamard_row_plus_minus(row_index=row_index, order=order)

    np.testing.assert_array_equal(got, expected[row_index])


def test_generate_order_sorting_matrix_small_cases() -> None:
    got_2_by_2 = generate_order_sorting_matrix(o_col=1, o_row=1)
    expected_2_by_2 = np.array([0, 2, 1, 3], dtype=np.int64)

    got_2_by_4 = generate_order_sorting_matrix(o_col=2, o_row=1)
    expected_2_by_4 = np.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=np.int64)

    np.testing.assert_array_equal(got_2_by_2, expected_2_by_2)
    np.testing.assert_array_equal(got_2_by_4, expected_2_by_4)


def test_generate_order_sorting_matrix_is_permutation() -> None:
    # 4-by-8?
    iy = generate_order_sorting_matrix(o_col=3, o_row=2)

    np.testing.assert_array_equal(np.sort(iy), np.arange(32))


def test_measure_split_hadamard_transform_matches_matlab_code() -> None:
    x = torch.tensor(
        [1.0, 2.0, 3.0, 4.0], dtype=torch.float32
    )  # vectorized 2-by-2 image
    h = torch.tensor(
        explicit_hadamard(n=4), dtype=torch.float32
    )  # 4-by-4 Hadamard matrix {-1, 1}

    # Convert to {0, 1} matrices
    h_plus = (1.0 + h) / 2.0
    h_minus = (1.0 - h) / 2.0

    expected_plus = h_plus @ x
    expected_minus = h_minus @ x

    got_plus, got_minus = measure_split_hadamard_transform(x)

    assert torch.allclose(got_plus, expected_plus, rtol=0.0, atol=1e-12)
    assert torch.allclose(got_minus, expected_minus, rtol=0.0, atol=1e-12)


def test_measure_split_hadamard_transform_rejects_bad_block_size() -> None:
    x = torch.tensor([1, 2, 3], dtype=torch.float32)

    with pytest.raises(ValueError):
        # Length is not power of two
        measure_split_hadamard_transform(x)


def test_pad_map_to_powers_of_two() -> None:
    image = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    padded = pad_to_powers_of_two_torch(image)

    assert padded.shape == (2, 4)
    assert torch.allclose(padded[:, :3], image)
    assert torch.allclose(padded[:, 3], torch.zeros(2, dtype=torch.float32))


def test_reconstruct_uses_matlab_column_major_order() -> None:
    image = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    x = matlab_vec(image)
    h = explicit_hadamard(4)
    y_h = torch.tensor(h @ x)
    iy = torch.arange(4, dtype=torch.int64)

    reconstructor = PCMReconstructor(
        iy=iy, n_row=2, n_col=2, n_rows_orig=2, n_cols_orig=2
    )

    got = reconstructor.reconstruct_from_sorted_measurements(
        y_sorted=y_h, transpose_after_reshape=False
    )

    assert torch.allclose(got, torch.tensor(image), rtol=0.0, atol=1e-12)


def test_reconstruct_real_measurement_transpose_matches_matlab_order() -> None:
    image = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    x = matlab_vec(image.T)
    h = explicit_hadamard(4)
    y_h = torch.tensor(h @ x)
    iy = np.arange(4, dtype=np.int64)

    reconstructor = PCMReconstructor(
        iy=torch.tensor(iy), n_row=2, n_col=2, n_rows_orig=2, n_cols_orig=2
    )

    got = reconstructor.reconstruct_from_sorted_measurements(
        y_sorted=y_h, transpose_after_reshape=True
    )

    assert torch.allclose(got, torch.tensor(image), rtol=0.0, atol=1e-12)


def test_generate_synthetic_noise_zero_noise_limit() -> None:
    rng = np.random.default_rng(0)
    noise_params = PCMNoiseParameter(
        beam_ar=(-0.9221, -0.1709, -0.0769, -0.0354),
        beam_ma=(0.0377, -0.3669),
        beam_sigma=0.0,
        bias_sigma=0.0,
        bias_gradient_sigma=0.0,
        dark_sigma=0.0,
    )
    i_bias = 2.5
    e_dark, e_bias, e_rel = generate_noise(
        num_measurements=16, i_bias=i_bias, noise_params=noise_params, rng=rng
    )

    np.testing.assert_allclose(e_dark, np.zeros(16), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(e_bias, i_bias * np.ones(16), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(e_rel, np.zeros(16), rtol=0.0, atol=1e-12)


def test_generate_synthetic_noise_reproducible_for_same_seed() -> None:
    noise_params = PCMNoiseParameter(
        beam_ar=(-0.9221, -0.1709, -0.0769, -0.0354),
        beam_ma=(0.0377, -0.3669),
        beam_sigma=1.2e-4,
        bias_sigma=2.9e-5,
        bias_gradient_sigma=1.6e-7,
        dark_sigma=3.6e-7,
    )

    e_dark_1, e_bias_1, e_rel_1 = generate_noise(
        num_measurements=16,
        i_bias=2.5,
        noise_params=noise_params,
        rng=np.random.default_rng(123),
    )
    e_dark_2, e_bias_2, e_rel_2 = generate_noise(
        num_measurements=16,
        i_bias=2.5,
        noise_params=noise_params,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_allclose(e_dark_1, e_dark_2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(e_bias_1, e_bias_2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(e_rel_1, e_rel_2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.isfinite(e_dark_1), np.ones_like(e_dark_1, dtype=bool)
    )
    np.testing.assert_allclose(
        np.isfinite(e_bias_1), np.ones_like(e_bias_1, dtype=bool)
    )
    np.testing.assert_allclose(np.isfinite(e_rel_1), np.ones_like(e_rel_1, dtype=bool))


def test_read_pcm_data(tmp_path: Path) -> None:
    path = tmp_path / "mock_pcm.txt"
    path.write_text(
        "header 1\n" "header 2\n" "header 3\n" "1\t0.1\n" "-1\t0.2\n" "2\t0.3\n"
    )

    got = read_pcm_measurement_data_from_txt(path)

    assert list(got.columns) == ["PatternIndex", "MeasuredCurrentA"]
    np.testing.assert_allclose(
        got.to_numpy(dtype=float),
        np.array(
            [
                [1.0, 0.1],
                [-1.0, 0.2],
                [2.0, 0.3],
            ]
        ),
    )


def test_compute_metrics_identical_image() -> None:
    image = torch.tensor(
        [
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
            [-2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
            [-3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
            [-4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
            [-5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0],
            [-6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0],
            [-7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0],
        ]
    )

    ssim, psnr = compute_metrics_hybrid(
        image=image, reference=image, pixel_min=image.min().item()
    )

    assert ssim == pytest.approx(1.0)
    assert math.isinf(psnr)


def test_torch_reshape_fortran_2d() -> None:
    x = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    n_row = 2
    n_col = 3

    got = torch_reshape_fortran_2d(x, n_row=n_row, n_col=n_col)

    expected = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.float32)

    assert got.shape == (n_row, n_col)
    assert torch.allclose(got, expected)
