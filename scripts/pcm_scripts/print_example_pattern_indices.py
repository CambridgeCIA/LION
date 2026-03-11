import numpy as np

from LION.operators.multilevel_sample import multilevel_sample


def print_example_pattern_indices() -> None:
    indices = multilevel_sample(
        J=3,  # 2^3 x 2^3  =  8 x 8 image  => 64 pixels
        num_samples=40,  # total number of samples to select
        coarse_J=2,  # 2^2 x 2^2  =  16 coarse samples
        alpha=1.0,  # decay exponent
        rng=np.random.default_rng(42),  # random number generator with a fixed seed
    )

    print(indices)  # print the selected indices as a list

    # J=3 so keep the first 3 levels (0, 1, 2)
    level_0_indices = [0]
    level_1_indices = [1, 2, 3]
    level_2_indices = list(range(4, 16))  # indices from 4 to 15

    # Break down the indices in the randomized part into sub-lists for readability
    # indices between 16 and 31
    random_indices_1 = [18, 19, 21, 22, 24, 28]
    # indices between 32 and 47
    random_indices_2 = [33, 34, 35, 36, 37, 38, 41, 43, 44, 46, 47]
    # indices between 48 and 63
    random_indices_3 = [51, 52, 53, 54, 60, 61, 63]

    expected_indices = np.array(
        level_0_indices
        + level_1_indices
        + level_2_indices
        + random_indices_1
        + random_indices_2
        + random_indices_3
    )

    assert np.array_equal(
        indices, expected_indices
    ), "The selected indices do not match the expected pattern."


if __name__ == "__main__":
    print_example_pattern_indices()
