"""
Configuration parameters for score-based CT reconstruction tasks.

Defines:
  - Sparse-view parallel and fan-beam CT geometries.
  - Simulated projection noise levels (clean, low, med, high).
  - VESDE parameters (sigma_min, sigma_max).
  - Model paths and seed setups.

Author: Tianzhen Peng
"""

from LION.CTtools.ct_geometry import Geometry
import numpy as np
import os
from LION.models.score_inverse.sde import VESDE

sad = 23 # sparse angle density
sparse_geometry = Geometry(
    image_shape=(1, 512, 512), 
    image_size=(1.0, 512.0, 512.0),
    detector_shape=(1, 725),
    detector_size=(1.0, 725.0),
    angles=np.linspace(0, np.pi, sad, endpoint=False),
    mode="parallel" 
)

sparse_geometry_fan = Geometry(
    image_shape=[1, 512, 512],
    image_size=[300 / 512, 300, 300],
    detector_shape=[1, 900],
    detector_size=[1, 900],
    dso=575,
    dsd=1050,
    mode="fan",
    angles=np.linspace(0, 2 * np.pi, sad, endpoint=False),
)

noise_presets = {
    'clean': {'I0': 0.0, 'sigma': 0.0, 'cross_talk': 0.0},
    'low': {'I0': 100000.0, 'sigma': 2.0, 'cross_talk': 0.0},
    'med': {'I0': 10000.0, 'sigma': 5.0, 'cross_talk': 0.0},
    'high': {'I0': 1000.0, 'sigma': 10.0, 'cross_talk': 0.0}
}

# read the location of this file
file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(file_dir, "data")

num_images = 16
seed_dataset = 0

sigma_min = 0.01
sigma_max = 220.0
sde = VESDE(sigma_min=sigma_min, sigma_max=sigma_max)

# CP_14="/home/tp534/rds/hpc-work/trained_models/ncsnpp/checkpoint_epoch_14_step_178229.pth"
# CP_24="/home/tp534/rds/hpc-work/trained_models/ncsnpp/checkpoint_epoch_24_step_297049.pth"
# CP_49="/home/tp534/rds/hpc-work/trained_models/ncsnpp/checkpoint_epoch_49_step_594099.pth"
# CP_99="/home/tp534/rds/hpc-work/trained_models/ncsnpp/checkpoint_epoch_99_step_1188199.pth"
# CP_199="/home/tp534/rds/hpc-work/trained_models/ncsnpp/checkpoint_epoch_199_step_2376399.pth"