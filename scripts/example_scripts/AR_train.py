#%% This example shows how to train MSDNet for full angle, noisy measurements.

#%% Imports
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Subset
import pathlib
from LION.models.LIONmodel import LIONmodel, ModelInputType, ModelParams
from LION.models.learned_regularizer.ACR import my_psnr, my_ssim
from LION.optimizers.ARsolver import ARParams, ARSolver
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments


#%%
# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)

# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/cs2186/trained_models/test_debugging/")

final_result_fname = "arlessdata_final_iter.pt"
checkpoint_fname = "arlessdata_check_*.pt"
validation_fname = "arlessdata_min_val.pt"
#
#%% Define experiment
# experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)
experiment = ct_experiments.clinicalCTRecon()
#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()
lidc_dataset_test = experiment.get_testing_dataset()

#%% Define DataLoader
# Use the same amount of training
batch_size = 4
# lidc_dataset = Subset(lidc_dataset, [i for i in range(250)])
lidc_dataset_val = Subset(lidc_dataset_val, [i for i in range(3)])
lidc_dataset_test = Subset(lidc_dataset_test, [i for i in range(20)])
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, 1, shuffle=True)
lidc_test = DataLoader(lidc_dataset_test, 1, shuffle=True)
#%% Model
class network(LIONmodel):
    def __init__(self, model_parameters=None, n_chan=1):
        super(network, self).__init__(model_parameters)

        self.leaky_relu = nn.LeakyReLU()
        self.convnet = nn.Sequential(
            nn.Conv2d(n_chan, 16, kernel_size=(5, 5), padding=2),
            self.leaky_relu,
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding=2),
            self.leaky_relu,
            nn.Conv2d(32, 32, kernel_size=(5, 5), padding=2, stride=2),
            self.leaky_relu,
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2, stride=2),
            self.leaky_relu,
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=2, stride=2),
            self.leaky_relu,
            nn.Conv2d(64, 128, kernel_size=(5, 5), padding=2, stride=2),
            self.leaky_relu,
        )

        self.fc = nn.Sequential(
            nn.Linear(131072, 256),
            self.leaky_relu,
            nn.Linear(256, 1),
        )

    @staticmethod
    def default_parameters(mode="ct") -> ModelParams:
        return ModelParams(model_input_type=ModelInputType.IMAGE, n_chan=1)

    def forward(self, image):
        output = self.convnet(image)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


model = network().to(device)
model.model_parameters.model_input_type = ModelInputType.IMAGE
#%% Optimizer
@dataclass
class TrainParam(LIONParameter):
    optimiser: str
    epochs: int
    learning_rate: float
    betas: Tuple[float, float]
    loss: str
    accumulation_steps: int


train_param = TrainParam("adam", 5, 1e-3, (0.9, 0.99), "MSELoss", 1)

optimizer = Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)


val_loss = nn.MSELoss()

#%% Solver
solver_params = ARParams(False, 500, 1e-6, 0.5, 0.95, 1e-3)
solver = ARSolver(model, optimizer, SGD, experiment.geo, True, device)
solver.solver_params.no_steps = 500
solver.solver_params.momentum = 0

solver.set_saving(savefolder, final_result_fname)
solver.set_checkpointing(checkpoint_fname, 5)
solver.set_training(lidc_dataloader)
solver.set_validation(lidc_validation, 5, val_loss, validation_fname)
solver.set_normalization(False)

# train regularizer
# solver.train(train_param.epochs)

# solver.save_final_results()
# solver.clean_checkpoints()

model, _, _ = network.load(savefolder.joinpath(final_result_fname))

model.model_parameters.model_input_type = ModelInputType.IMAGE
solver.model = model

solver.set_testing(lidc_test, my_ssim)
ssims = solver.test()
solver.set_testing(lidc_test, my_psnr)
psnrs = solver.test()

with open("ar_results.txt", "w") as f:
    f.write(
        f"Min SSIM {np.min(ssims)}, Max SSIM {np.max(ssims)}, Mean SSIM {np.mean(ssims)}, SSIM std {np.std(ssims)}\n"
    )
    f.write(
        f"Min PSNR {np.min(psnrs)}, Max PSNR {np.max(psnrs)}, Mean PSNR {np.mean(psnrs)}, PSNR std {np.std(psnrs)}\n"
    )

#%% Use regularizer
