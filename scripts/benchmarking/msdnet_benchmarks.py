#%% This example shows how to train MSDNet for full angle, noisy measurements.


#%% Imports
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pathlib
from LION.experiments import ct_benchmarking_experiments
from LION.models.CNNs.MSDNets.FBPMS_D import FBPMSD_Net, OGFBPMSD_Net
from LION.models.CNNs.MSDNets.MS_D import MS_D
from LION.models.CNNs.MSDNets.MS_D2 import MSD_Params
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments


#%%
# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/cs2186/trained_models/clinical_dose/")


#
#%% Define experiment
experiments = [ct_benchmarking_experiments.FullDataCTRecon(), ct_benchmarking_experiments.LimitedAngle150CTRecon(), ct_benchmarking_experiments.LimitedAngle120CTRecon(), ct_benchmarking_experiments.LimitedAngle90CTRecon(), ct_benchmarking_experiments.LimitedAngle60CTRecon(), ct_benchmarking_experiments.SparseAngle720CTRecon(), ct_benchmarking_experiments.SparseAngle360CTRecon(), ct_benchmarking_experiments.SparseAngle180CTRecon(), ct_benchmarking_experiments.SparseAngle120CTRecon(), ct_benchmarking_experiments.SparseAngle90CTRecon(), ct_benchmarking_experiments.SparseAngle60CTRecon(), ct_benchmarking_experiments.LowDoseCTRecon(), ct_benchmarking_experiments.BeamHardeningCTRecon()]

for experiment in experiments:
    experiment_str = str(type(experiment)).split("ct_benchmarking_experiments.")[1][:-2]
    #%% Dataset
    lidc_dataset = experiment.get_training_dataset()
    lidc_dataset_val = experiment.get_validation_dataset()

    #%% Define DataLoader
    # Use the same amount of training
    batch_size = 10
    lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
    lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)

    #%% Model
    width, depth = 1, 100
    dilations = []
    for i in range(depth):
        for j in range(width):
            dilations.append((((i * width) + j) % 10) + 1)
    model_params = MSD_Params(
        in_channels=1,
        width=width,
        depth=depth,
        dilations=dilations,
        look_back_depth=-1,
        final_look_back_depth=-1,
        activation="ReLU",
    )
    model = FBPMSD_Net(geometry_parameters=experiment.geo, model_parameters=model_params).to(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {count_parameters(model)} parameters")

    #%% Optimizer
    train_param = LIONParameter()

    # loss fn
    loss_fcn = torch.nn.MSELoss()
    train_param.optimiser = "adam"

    # optimizer
    train_param.epochs = 1
    train_param.learning_rate = 1e-3
    train_param.betas = (0.9, 0.99)
    train_param.loss = "MSELoss"
    train_param.accumulation_steps = 1
    optimiser = torch.optim.Adam(
        model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
    )

    # learning parameter update
    steps = len(lidc_dataloader)
    model.train()
    min_valid_loss = np.inf
    total_loss = np.zeros(train_param.epochs)
    start_epoch = 0

    total_loss = np.resize(total_loss, train_param.epochs)
    print(f"Starting iteration at epoch {start_epoch}")
    optimiser.zero_grad()

    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    start_time = time.time()
    total_train_time = 0
    total_validation_time = 0
    #%% train
    for epoch in range(start_epoch, train_param.epochs):
        train_loss = 0.0
        print("Training...")
        model.train()
        for idx, (sinogram, target_reconstruction) in enumerate(tqdm(lidc_dataloader)):   
            sinogram = sinogram.to(device)
            target_reconstruction = target_reconstruction.to(device)
            with torch.autocast("cuda"):
                reconstruction = model(sinogram)
                loss = loss_fcn(reconstruction, target_reconstruction)
                loss = loss / train_param.accumulation_steps

            scaler.scale(loss).backward()

            train_loss += loss.item()

            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()

        total_loss[epoch] = train_loss
        total_train_time += time.time() - start_time
        
        # Validation
        valid_loss = 0.0
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            print("Validating...")
            for sinogram, target_reconstruction in tqdm(lidc_validation):
                reconstruction = model(sinogram)
                loss = loss_fcn(target_reconstruction, reconstruction.to(device))
                valid_loss += loss.item()

        print(
            f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(lidc_dataloader)} \t\t Validation Loss: {valid_loss / len(lidc_validation)}"
        )
        
        total_validation_time += time.time() - start_time

    with open(f"{experiment_str}_benchmarking.txt", 'w') as f:
        f.write("Model took {total_train_time/train_param.epochs}s /epoch to train on average\n")
        f.write(f"Model took {total_train_time/train_param.epochs}s /epoch to train on average\n")
        f.write(f"Model achieved minimum training loss of {min(total_loss)}\n")
        f.write(f"Model took {total_validation_time/train_param.epochs}s /epoch to validate on average\n")
        f.write(f"Model achieved validation loss of {min_valid_loss}\n")


