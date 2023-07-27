#%% Learned Primal Dual


#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib
import AItomotools.CTtools.ct_geometry as ctgeo
import AItomotools.CTtools.ct_utils as ct
from AItomotools.data_loaders.LIDC_IDRI import LIDC_IDRI
from AItomotools.models.MS_D import MS_D
from AItomotools.utils.parameter import Parameter
import AItomotools.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk


#%% Chose device:
device = torch.device("cuda:2")
torch.cuda.set_device(device)
savefolder = pathlib.Path("/home/ab2860/rds/hpc-work/models/low_dose_full_angle")
datafolder = pathlib.Path("/home/ab2860/rds/hpc-work/AItomotools/processed/LIDC-IDRI/")
savefolder = pathlib.Path("/store/DAMTP/ab2860/wip_models/")
datafolder = pathlib.Path("/local/scratch/public/AItomotools/processed/LIDC-IDRI/")
# ##################################################
#%% Define experiment
experiment = ct_experiments.LowDoseCTRecon()
experiment.param.data_loader_params.folder = datafolder

#%% Dataset
lidc_dataset = experiment.get_training_dataset()

# Use the same amount of training
batch_size = 16
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)

#%% Model
# Default model is already from the paper.
model = MS_D().to(device)

#%% Operators for self-supervised training

# number of data splits
k = 4
op = []
geo = experiment.geo
angles = geo.angles.copy()
for i in range(k):
    geo.angles = angles[i:-1:k]
    op.append(ct.make_operator(geo))
#%% Optimizer
train_param = Parameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 100
train_param.learning_rate = 1e-3
train_param.loss = "MSELoss"
optimiser = torch.optim.Adam(model.parameters(), lr=train_param.learning_rate)

# learning parameter update
steps = len(lidc_dataloader)
model.train()
total_loss = np.zeros(train_param.epochs)
start_epoch = 0

# %% Check if there is a checkpoint saved, and if so, start from there.

my_file = savefolder.joinpath("Noise2Inverse_final_iter.pt")
if my_file.is_file():
    print("Final version found, no need to loop further, exiting")
    exit()


checkpoints = sorted(list(savefolder.glob("Noise2Inverse_check_*.pt")))
if checkpoints:
    model, options, data = MS_D.load_checkpoint(savefolder.joinpath(checkpoints[-1]))
    optimiser.load_state_dict(data["optimizer_state_dict"])
    start_epoch = data["epoch"]
    total_loss = data["loss"]

print(f"Starting iteration at epoch {start_epoch}")
#%% train
for epoch in range(start_epoch, train_param.epochs):
    train_loss = 0.0
    for index, (sinogram, target_reconstruction) in tqdm(enumerate(lidc_dataloader)):
        optimiser.zero_grad()

        # do the FBP recon per k split
        size_noise2inv = list(target_reconstruction.shape)
        size_noise2inv.insert(0, k)
        bad_recon = torch.zeros(size_noise2inv, device=device)
        for sino in range(sinogram.shape[0]):
            for split in range(k):
                bad_recon[split, sino] = fdk(op[split], sinogram[sino, 0, split:-1:k])
        # pick one random to be the single
        label_array = torch.zeros(target_reconstruction.shape, device=device)
        target = torch.zeros(target_reconstruction.shape, device=device)
        for sino in range(sinogram.shape[0]):
            indices = np.arange(k)
            label = np.random.randint(k)
            target[sino] = bad_recon[label, sino].detach().clone()
            label_array[sino] = torch.mean(
                bad_recon[np.delete(indices, label), sino].detach().clone(), axis=0
            )
        # feed to network
        reconstruction = model(label_array)

        loss = loss_fcn(reconstruction, target)

        loss.backward()

        train_loss += loss.item()

        optimiser.step()
    total_loss[epoch] = train_loss

    # Checkpoint every 10 iters anyway
    if epoch % 10 == 0:
        model.save_checkpoint(
            savefolder.joinpath(f"Noise2Inverse_check_{epoch+1:04d}.pt"),
            epoch + 1,
            total_loss,
            optimiser,
            train_param,
        )


model.save(
    savefolder.joinpath("Noise2Inverse_final_iter.pt"),
    epoch=train_param.epochs,
    training=train_param,
)
