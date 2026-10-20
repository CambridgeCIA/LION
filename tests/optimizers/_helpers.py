"""Small helpers shared by the solver tests."""

import torch
from torch.utils.data import DataLoader, TensorDataset

import LION.optimizers.SupervisedSolver as supervised_solver
from LION.CTtools.ct_geometry import Geometry
from LION.models.CNNs.dncnn import DnCNN
from LION.models.LIONmodel import ModelInputType


def small_dncnn(input_type=ModelInputType.IMAGE):
    params = DnCNN.default_parameters()
    params.blocks = 3
    params.int_channels = 4
    params.model_input_type = input_type
    return DnCNN(params)


def loader(n=4, batch_size=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, 8, 8, generator=g)
    y = torch.rand(n, 1, 8, 8, generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def make_solver(model=None, lr=1e-2, **kwargs):
    model = small_dncnn() if model is None else model
    return supervised_solver.SupervisedSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=lr),
        torch.nn.MSELoss(),
        geometry=Geometry.default_parameters(),
        device=torch.device("cpu"),
        **kwargs,
    )
