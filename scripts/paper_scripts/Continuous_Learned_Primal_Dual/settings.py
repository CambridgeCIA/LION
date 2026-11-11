# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Christina Runkel, Ander Biguri
# =============================================================================
"""What the cLPD paper (Runkel, Biguri and Schönlieb, 2024) trained and
tested, shared by ``train.py`` and ``evaluate.py``: as the code that trained
the paper's models (June 2024), which the paper's text doesn't always match.
"""

import torch

from LION.experiments import ct_experiments
from LION.models.iterative_unrolled.cLPD import cLPD
from LION.models.iterative_unrolled.LPD import LPD
from LION.operators.ct.ct_geometry import Geometry
from LION.utils.paths import results_path

# The paper's six problems (its Table 1). Its "clinical dose" was LION's low
# dose then (I0 = 3500), its "extremely low dose" I0 = 1000.
# name: (LION experiment, geometry of June 2024, I0)
EXPERIMENTS = {
    "full_low": (  # 1. full angle, "clinical"
        ct_experiments.LowDoseCTRecon,
        Geometry.default_parameters,
        3500,
    ),
    "full_extreme_low": (  # 2.a full angle, extremely low dose
        ct_experiments.ExtremeLowDoseCTRecon,
        Geometry.default_parameters,
        1000,
    ),
    "sparse_low": (  # 2.b sparse angle, "clinical"
        ct_experiments.SparseAngleLowDoseCTRecon,
        Geometry.sparse_view_parameters,
        3500,
    ),
    "sparse_extreme_low": (  # 2.c sparse angle, extremely low dose
        ct_experiments.SparseAngleExtremeLowDoseCTRecon,
        Geometry.sparse_view_parameters,
        1000,
    ),
    "limited_low": (  # 3.a limited angle, "clinical"
        ct_experiments.LimitedAngleLowDoseCTRecon,
        Geometry.sparse_angle_parameters,
        3500,
    ),
    "limited_extreme_low": (  # 3.b limited angle, extremely low dose
        ct_experiments.LimitedAngleExtremeLowDoseCTRecon,
        Geometry.sparse_angle_parameters,
        1000,
    ),
}

# The two models, with the parameters they were trained with
MODELS = {
    "cLPD": (cLPD, cLPD.continous_LPD_paper),
    "LPD": (LPD, LPD.continous_LPD_paper),
}

# The training: Adam (learning rate 1e-4, betas 0.9 and 0.99), batch size 2,
# 100 epochs, the mean squared error to the target, validated every epoch
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.99)
LOSS = torch.nn.MSELoss


def experiment(name, data_folder=None):
    """The paper's problem ``name`` (a key of ``EXPERIMENTS``), as LION's
    experiment with the settings of June 2024:

    - the geometry: LION's default fan beam then (512x512 pixels over 300 mm,
      a 900-pixel detector), with 360, 50 (sparse) or 60 over 60 degrees
      (limited) angles;
    - the noise: the relative model (each sinogram scaled by its maximum),
      with I0 photons, electronic noise of 5 photons and 5% cross-talk, and
      starved rays at 1e-6 (``min_counts=None``); fresh noise each time a
      slice is read (``seed=None``);
    - the data: 5 slices per patient, half with nodules, chosen and split
      (80/10/10 by patient) as LION did then (``legacy_*``).

    ``data_folder``: a processed LIDC-IDRI folder other than LION's (e.g. one
    pre-processed before October 2026, at each scan's own pixel spacing)."""
    experiment_class, geometry, I0 = EXPERIMENTS[name]
    parameters = experiment_class.default_parameters()
    parameters.name += " (cLPD paper, settings of June 2024)"
    parameters.operator_params = geometry()
    noise = parameters.noise_params
    noise.noise_model = "relative"
    noise.I0 = I0
    noise.sigma = 5
    noise.min_counts = None
    noise.seed = None
    data = parameters.data_loader_params
    data.operator_params = parameters.operator_params
    data.max_num_slices_per_patient = 5
    data.pcg_slices_nodule = 0.5
    data.training_proportion = 0.8
    data.validation_proportion = 0.1
    data.legacy_slice_selection = True
    data.legacy_patient_list = True
    return experiment_class(parameters, datafolder=data_folder)


def folder(experiment_name):
    """Where the models trained on one problem, and their evaluation, go."""
    return results_path(f"paper_scripts/cLPD/{experiment_name}")
