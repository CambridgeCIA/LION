from dataclasses import dataclass
from typing import Tuple
import pathlib
import torch
from torch.utils.data import DataLoader, Subset
from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
from LION.models.CNNs.UNets.Unet import UNet
from LION.optimizers.EquivariantSolver import EquivariantSolver
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from LION.metrics.ssim import SSIM


# %%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/ab2860/trained_models/")
# Creates the folders if they does not exist
savefolder.mkdir(parents=True, exist_ok=True)
final_result_fname = "Equivariance.pt"
checkpoint_fname = "Equivariance_check_*.pt"
validation_fname = "Equivariance_min_val.pt"
#
# %% Define experiment

experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")

# %% Dataset
lidc_dataset = experiment.get_training_dataset()

# smaller dataset for example. Remove this for full dataset
lidc_dataset = Subset(lidc_dataset, [i for i in range(100)])


# %% Define DataLoader

batch_size = 2
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_test = DataLoader(experiment.get_testing_dataset(), 3, True)
lidc_val = DataLoader(experiment.get_validation_dataset(), 3, True)

# %% Model
model = UNet().to(device)

# %% Optimizer
@dataclass
class TrainParams(LIONParameter):
    optimizer: str
    epochs: int
    learning_rate: float
    betas: Tuple[float, float]
    loss: str


train_param = TrainParams("adam", 20, 1e-4, (0.9, 0.99), "MSELoss")

# loss fn
loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# %% Train
# create solver
solver = EquivariantSolver(
    model,
    optimiser,
    loss_fn,
    experiment.geometry,
    verbose=False,
    device=device,
)

print(solver.cite("MLA"))
# set data
solver.set_training(lidc_dataloader)
solver.set_saving(savefolder, final_result_fname)
# SSIM is a score: the solver keeps the model with the highest value
solver.set_validation(lidc_val, 1, SSIM(), validation_fname)
# solver.set_normalization(True)
solver.set_testing(lidc_test, SSIM())

# set checkpointing procedure
solver.set_checkpointing(checkpoint_fname, 10, save_folder=savefolder)

# train
solver.train(train_param.epochs)
# delete checkpoints if finished
solver.clean_checkpoints()
# save final result
solver.save_final_results(final_result_fname, savefolder)
