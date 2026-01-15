from pprint import pprint

import numpy as np

from LION.operators.multilevel_sample import multilevel_sample


indices = multilevel_sample(
    J=3,  # 2^3 x 2^3  =  8 x 8 image  => 64 pixels
    num_samples=40,  # total number of samples to select
    coarse_J=2,  # 2^2 x 2^2  =  16 coarse samples
    alpha=1.0,  # decay exponent
    rng=np.random.default_rng(42),  # random number generator with a fixed seed
)

pprint(indices.tolist())  # print the selected indices as a list
