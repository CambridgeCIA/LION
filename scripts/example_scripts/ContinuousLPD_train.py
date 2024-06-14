# %% This example shows how to train Continuous LPD for full angle, noisy measurements.


# %% Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI

from LION.models.ContinuousLPD import ContinuousLPD
from LION.utils.parameter import Parameter
from ts_algorithms import fdk


import LION.experiments.ct_experiments as ct_experiments


# %%
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# %%
# arguments for argparser
parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=str)
parser.add_argument("--dose", type=str)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--second_order", type=str2bool)
parser.add_argument("--instance_norm", type=str2bool)

# %%
args = parser.parse_args()
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
if args.dose == "low":
    savefolder = pathlib.Path(
        "/home/cr661/rds/hpc-work/store/LION/trained_models/low_dose/"
    )
elif args.dose == "extreme_low":
    savefolder = pathlib.Path(
        "/home/cr661/rds/hpc-work/store/LION/trained_models/extreme_low_dose/"
    )
else:
    raise ValueError("Dose not recognised")
datafolder = pathlib.Path("/home/cr661/rds/hpc-work/store/LION/data/LIDC-IDRI/")
final_result_fname = savefolder.joinpath(
    f"ContinuousLPD_final_iterBS2smallerLR_no_adjoint_in{args.instance_norm}_{args.dose}_{args.geometry}.pt"
)
checkpoint_fname = savefolder.joinpath(
    f"ContinuousLPD_checkBS2smallerLR_no_adjoint_in{args.instance_norm}_{args.dose}_{args.geometry}*.pt"
)
validation_fname = savefolder.joinpath(
    f"ContinuousLPD_min_valBS2smallerLR_no_adjoint_in{args.instance_norm}_{args.dose}_{args.geometry}.pt"
)
#
# %% Define experiment
if args.geometry == "full":
    if args.dose == "low":
        experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)
    elif args.dose == "extreme_low":
        experiment = ct_experiments.ExtremeLowDoseCTRecon(datafolder=datafolder)
    else:
        raise ValueError("Dose not recognised")
elif args.geometry == "limited_angle":
    if args.dose == "low":
        experiment = ct_experiments.LimitedAngleLowDoseCTRecon(datafolder=datafolder)
    elif args.dose == "extreme_low":
        experiment = ct_experiments.LimitedAngleExtremeLowDoseCTRecon(
            datafolder=datafolder
        )
    else:
        raise ValueError("Dose not recognised")
elif args.geometry == "sparse_angle":
    if args.dose == "low":
        experiment = ct_experiments.SparseAngleLowDoseCTRecon(datafolder=datafolder)
    elif args.dose == "extreme_low":
        experiment = ct_experiments.SparseAngleExtremeLowDoseCTRecon(
            datafolder=datafolder
        )
    else:
        raise ValueError("Dose not recognised")
else:
    raise ValueError("Geometry not recognised")

# %% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

# %% Define DataLoader
# Use the same amount of training
batch_size = 2
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)

# %% Model
# Default model is already from the paper.
default_parameters = ContinuousLPD.default_parameters()
# This makes the LPD calculate the step size for the backprojection, which in my experience results in much much better pefromace
# as its all in the correct scale.
default_parameters.step_size = None
default_parameters.learned_step = True
default_parameters.step_positive = True
print(f"Training ContinuousLPD with second order: {args.second_order}")
model = ContinuousLPD(
    geometry_parameters=experiment.geo,
    model_parameters=default_parameters,
    second_order=args.second_order,
    instance_norm=args.instance_norm,
).to(device)


# %% Optimizer
train_param = Parameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 100
train_param.learning_rate = args.lr
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"
optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# learning parameter update
steps = len(lidc_dataloader)
model.train()
min_valid_loss = np.inf
total_loss = np.zeros(train_param.epochs)
start_epoch = 0

# %% Check if there is a checkpoint saved, and if so, start from there.

# If there is a file with the final results, don't run again
if model.final_file_exists(savefolder.joinpath(final_result_fname)):
    print("final model exists! You already reahced final iter")
    exit()

model, optimiser, start_epoch, total_loss, _ = ContinuousLPD.load_checkpoint_if_exists(
    checkpoint_fname, model, optimiser, total_loss
)
print(f"Starting iteration at epoch {start_epoch}")

# %% train
for epoch in range(start_epoch, train_param.epochs):
    train_loss = 0.0
    model.train()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    for index, (sinogram, target_reconstruction) in tqdm(enumerate(lidc_dataloader)):
        optimiser.zero_grad()
        reconstruction = model(sinogram)
        loss = loss_fcn(reconstruction, target_reconstruction)

        loss.backward()

        train_loss += loss.item()

        optimiser.step()
        # scheduler.step()
    total_loss[epoch] = train_loss
    # Validation
    valid_loss = 0.0
    model.eval()
    for index, (sinogram, target_reconstruction) in tqdm(enumerate(lidc_validation)):
        reconstruction = model(sinogram)
        loss = loss_fcn(target_reconstruction, reconstruction)
        valid_loss += loss.item()

    print(
        f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(lidc_dataloader)} \t\t Validation Loss: {valid_loss / len(lidc_validation)}"
    )

    if min_valid_loss > valid_loss:
        print(
            f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
        )
        min_valid_loss = valid_loss
        # Saving State Dict
        model.save(
            validation_fname,
            epoch=epoch + 1,
            training=train_param,
            loss=min_valid_loss,
            dataset=experiment.param,
        )

    # Checkpoint every 10 iters anyway
    # if epoch % 10 == 0:
    model.save_checkpoint(
        pathlib.Path(str(checkpoint_fname).replace("*", f"{epoch+1:04d}")),
        epoch + 1,
        total_loss,
        optimiser,
        train_param,
        dataset=experiment.param,
    )


model.save(
    final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
    dataset=experiment.param,
)
