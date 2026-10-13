import numpy as np
import torch
from LION.pcm_tools.noise_generate import generate_pairwise_swap_indices


def test_generate_pairwise_swap_indices_seeded():
    rng = np.random.default_rng(42)

    i_swap = generate_pairwise_swap_indices(
        num_pairs=5, rng=rng, device=torch.device("cpu")
    )

    expected = torch.tensor([0, 1, 3, 2, 5, 4, 6, 7, 8, 9])

    torch.testing.assert_close(i_swap, expected)


def test_generate_pairwise_swap_indices_only_swaps_within_pairs():
    num_pairs = 100
    rng = np.random.default_rng(0)

    i_swap = generate_pairwise_swap_indices(
        num_pairs=num_pairs, rng=rng, device=torch.device("cpu")
    )

    pairs = i_swap.reshape(num_pairs, 2)

    for k, pair in enumerate(pairs):
        expected = torch.tensor([2 * k, 2 * k + 1])
        assert torch.equal(torch.sort(pair).values, expected)


def test_generate_pairwise_swap_indices_is_permutation():
    num_pairs = 100
    rng = np.random.default_rng(0)

    i_swap = generate_pairwise_swap_indices(
        num_pairs=num_pairs, rng=rng, device=torch.device("cpu")
    )

    expected = torch.arange(2 * num_pairs)

    assert torch.equal(torch.sort(i_swap).values, expected)
