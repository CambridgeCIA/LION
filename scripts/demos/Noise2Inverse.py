from LION.classical_algorithms.fdk import fdk
from LION.optimizers.noise2inverse import Noise2InverseParams, Noise2InverseSolver
from LION.models.CNNs.MSDNets.MSDNet import MSDNet
import LION.experiments.ct_experiments as ct_experiments
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import torch.nn as nn
import torch

device = torch.device("cuda:0")

experiment = ct_experiments.clinicalCTRecon()

batch_size = 1
dataloader = DataLoader(experiment.get_testing_dataset(), batch_size, True)

model = MSDNet()
optimizer = Adam(model.parameters())
loss_fn = nn.MSELoss()

solver_params = Noise2InverseParams(
    4, fdk, Noise2InverseSolver.X_one_strategy(4)
)  # this is default, but if you want to define your own...
solver = Noise2InverseSolver(
    model, optimizer, loss_fn, solver_params, True, experiment.geo, device
)

solver.set_saving("path/to/save/folder", "finalresult.pt")
solver.set_checkpointing("checkpointfname", 3)
solver.set_loading("path/to/load/folder", False)
solver.set_training(dataloader)

epochs = 5
solver.train(epochs)

solver.save_final_results()
solver.clean_checkpoints()
