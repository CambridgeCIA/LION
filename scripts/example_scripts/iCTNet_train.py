import LION.experiments.ct_experiments as ct_experiments
from LION.models.CNNs.iCTNet import iCTNet
from torch.utils.data import DataLoader, Subset
from torch.optim.adam import Adam
import torch.nn as nn
import torch

from LION.optimizers.supervised_learning import SupervisedSolver

device = torch.device("cuda:3")

experiment = ct_experiments.LimitedAngleCTRecon()

dataset = experiment.get_training_dataset()
dataset = Subset(dataset, [i for i in range(20)])
dataloader = DataLoader(dataset, 1, True)

model = iCTNet(experiment.geo).to(device)

optimizer = Adam(model.parameters())

solver = SupervisedSolver(model, optimizer, nn.MSELoss(), True, experiment.geo, None, device)
solver.set_training(dataloader)

solver.train(3)
