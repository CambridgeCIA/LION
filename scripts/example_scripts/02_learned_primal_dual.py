# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""An unrolled method: Learned Primal-Dual (Adler and Öktem, 2018).

The same supervised training as 01_supervised_training.py, with a model that
takes the measurements and uses the operator inside: LPD alternates learned
updates in the measurement (dual) and image (primal) domains, with the CT
projector and backprojector between them. Training goes through the operator,
so it needs one torch can differentiate (the CT operator's ``autograd()``).

The model's parameters are changed from the paper's here, as an example:
``LPD.default_parameters()`` lists them all. The other unrolled models (LG,
ItNet, cLPD) are trained the same way.

Needs: a GPU, the processed LIDC-IDRI data in ``LION_DATA_PATH``, and
``LION_RESULTS_PATH``.
"""

# %% Imports
import torch
from torch.utils.data import DataLoader, Subset

from LION.experiments import ct_experiments
from LION.metrics.psnr import PSNR
from LION.models.iterative_unrolled.LPD import LPD
from LION.optimizers.SupervisedSolver import SupervisedSolver
from LION.reconstructors.ForwardReconstructor import ForwardReconstructor
from LION.utils.paths import results_path

# %% Settings
EPOCHS = 5
TRAINING_SLICES = 200
VALIDATION_SLICES = 20
# LION uses the GPU if there is one (torch.cuda.set_device(k) picks another);
# device is where this script moves samples to use them
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
verbose = 2  # 0: nothing; 1: one progress bar; 2: also each epoch's bars and messages

save_folder = results_path("examples/lpd")
save_folder.mkdir(parents=True, exist_ok=True)

# %% Data
experiment = ct_experiments.LowDoseCTRecon()
training = experiment.get_dataset("train")
validation = experiment.get_dataset("validation")
if TRAINING_SLICES is not None:
    training = Subset(training, range(min(TRAINING_SLICES, len(training))))
    validation = Subset(validation, range(min(VALIDATION_SLICES, len(validation))))

# %% The model
parameters = LPD.default_parameters()
print(parameters)  # every setting, with the paper's values
parameters.n_iters = 5  # fewer iterations than the paper's 10: faster
parameters.learned_step = True  # learn each iteration's step size
parameters.step_positive = True  # ...and keep it positive
model = LPD(experiment.operator_params, parameters)
print(model.cite("MLA"))

# %% Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
solver = SupervisedSolver(model, optimizer, torch.nn.MSELoss(), verbose=verbose)
solver.set_training(DataLoader(training, batch_size=1, shuffle=True))
solver.set_validation(
    DataLoader(validation, batch_size=1),
    validation_freq=1,
    validation_fn=PSNR(),
    validation_fname="LPD_best",
)
solver.set_saving(save_folder, "LPD")
solver.set_checkpointing("LPD_check_*.pt", checkpoint_freq=1)
solver.train(EPOCHS)
solver.save_final_results()
solver.clean_checkpoints()

# %% Use it
# eval() sets how layers behave (e.g. dropout, batch normalisation), not
# whether gradients are computed: that is torch.no_grad(). A reconstructor
# runs the model as its solver did (normalisation, initial reconstruction),
# on one sample or a batch, without gradients.
model.eval()
lpd = ForwardReconstructor(experiment.operator_params, model)
sinogram, image = validation[0]
reconstruction = lpd.reconstruct(sinogram.to(device)).cpu()
fdk = lpd.op.default_reconstruction(sinogram.to(device)).cpu()  # the baseline
psnr = PSNR()
print(
    f"PSNR: LPD {psnr(reconstruction, image, batched=False):.2f} dB, "
    f"FDK {psnr(fdk, image, batched=False):.2f} dB"
)
