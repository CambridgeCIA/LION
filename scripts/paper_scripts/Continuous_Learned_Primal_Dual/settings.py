# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Christina Runkel, Ander Biguri
# =============================================================================
"""What the cLPD paper (Runkel, Biguri and Schönlieb, 2024) trains and tests,
shared by ``train.py`` and ``evaluate.py``."""

import torch

from LION.experiments import ct_experiments
from LION.models.iterative_unrolled.cLPD import cLPD
from LION.models.iterative_unrolled.LPD import LPD
from LION.utils.paths import results_path

# The paper's six problems (its Table 1), as LION's experiments
EXPERIMENTS = {
    "full_clinical": ct_experiments.clinicalCTRecon,  # 1
    "full_extreme_low_dose": ct_experiments.ExtremeLowDoseCTRecon,  # 2.a
    "sparse_clinical": ct_experiments.SparseAngleCTRecon,  # 2.b
    "sparse_extreme_low_dose": ct_experiments.SparseAngleExtremeLowDoseCTRecon,  # 2.c
    "limited_clinical": ct_experiments.LimitedAngleCTRecon,  # 3.a
    "limited_extreme_low_dose": ct_experiments.LimitedAngleExtremeLowDoseCTRecon,  # 3.b
}

# The two models, each with the paper's parameters
MODELS = {
    "cLPD": (cLPD, cLPD.continous_LPD_paper),
    "LPD": (LPD, LPD.continous_LPD_paper),
}

# The paper's training: Adam, learning rate 1e-4, batch size 2, 100 epochs,
# the mean squared error to the target
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.99)
LOSS = torch.nn.MSELoss


def folder(experiment_name):
    """Where the models trained on one problem, and their evaluation, go."""
    return results_path(f"paper_scripts/cLPD/{experiment_name}")
