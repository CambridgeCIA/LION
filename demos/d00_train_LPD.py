import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from AItomotools.data_loaders.luna16_dataset import Luna16Dataset
from AItomotools.CTtools import ct_transforms
from AItomotools.models.LPD import LPD

#%% Demo to show how to load, define, train, a model.
# Use this as a template to train your networks and design new ones

# Lets set the GPU.
device = torch.device("cuda:0")

#%% Creating our LPD model


mode = "training"


n_angles = 360

batch_size = 8
epochs = 10
n_iters_LPD = 5
learning_rate = 1e-4

luna16_training = Luna16Dataset(device, mode)
geom = luna16_training.geometry
sinogram_transform = ct_transforms.CompleteSinogramTransform(
    1000, torch.Size((1, n_angles, geom.detector_shape[1])), 5, 0.05, None, None
)
luna16_training.set_sinogram_transform(sinogram_transform)

luna16_dataloader = DataLoader(luna16_training, batch_size, shuffle=True)

model = LPD(LPD.default_parameters(), geom).to(device)

loss_fcn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    epoch_loss = 0
    for index, (sinogram, target_reconstruction) in tqdm(enumerate(luna16_dataloader)):
        optimiser.zero_grad()
        reconstruction = model(sinogram)
        loss = loss_fcn(reconstruction, target_reconstruction)
        loss.backward()
        epoch_loss += loss.item()
        optimiser.step()

        if index % 100 == 0:
            plt.matshow(reconstruction[0, 0].detach().cpu())
            plt.savefig("current_reconstruction.jpg")
            plt.clf()

    torch.save(model.state_dict(), "lpd.pt")
