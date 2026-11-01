import os
import json
import numpy as np
import h5py
import LION.CTtools.ct_utils as ct_utils
import LION.CTtools.ct_geometry as ctgeo
from LION.utils.parameter import LIONParameter
import torch
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from LION.classical_algorithms.fdk import fdk
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver
from LION.models.CNNs.UNets.Unet import UNet
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import torch.nn as nn
import random

##Activate lion_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


OUTPUT_DIR = "./Dataset_Hackathon/"
SAVE_DIR = "./trained_models"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


class FoamLike(Dataset):
    def __init__(self, mode, parameters: LIONParameter):
        self.path_to_dataset = parameters.path_to_dataset
        self.mode = mode

        all_sino = sorted(glob.glob(os.path.join(self.path_to_dataset, "sino_*.h5")))
        all_gt = sorted(glob.glob(os.path.join(self.path_to_dataset, "gt_*.h5")))

        max_samples = getattr(parameters, "max_samples", len(all_sino))
        all_sino = all_sino[:max_samples]
        all_gt = all_gt[:max_samples]

        num = len(all_sino)
        if mode == "train":
            self.sino_files = all_sino[: int(num * 0.8)]
            self.gt_files = all_gt[: int(num * 0.8)]
        elif mode == "validation":
            self.sino_files = all_sino[int(num * 0.8) : int(num * 0.9)]
            self.gt_files = all_gt[int(num * 0.8) : int(num * 0.9)]
        else:
            self.sino_files = all_sino
            self.gt_files = all_gt

    def __len__(self):
        return len(self.sino_files)

    def __getitem__(self, index):
        with h5py.File(self.sino_files[index], "r") as f:
            sino = f["sinogram"][:].astype(np.float32)
        with h5py.File(self.gt_files[index], "r") as f:
            gt = f["image"][:].astype(np.float32)

        input_tensor = torch.from_numpy(sino).unsqueeze(0)
        target_tensor = torch.from_numpy(gt).unsqueeze(0)

        return input_tensor, target_tensor


# Geometry of the experiment: the one the data were made with
# (generate_foam_dataset.py writes it)
with open(os.path.join(OUTPUT_DIR, "geometry.json")) as f:
    geometry_values = json.load(f)
geometry_values["angles"] = np.array(geometry_values["angles"])
geometry = ctgeo.Geometry(**geometry_values)

params = LIONParameter()
params.geometry = geometry
params.path_to_dataset = OUTPUT_DIR
params.max_samples = 30

# 30 samples
samples = FoamLike(mode="all", parameters=params)


# Add noise to the sample sinograms
sigma_paper = 5
sigma_blurred_paper = 0.8
ks = int(sigma_blurred_paper * 3) * 2 + 1

noisy_list_y = []
target_list = []

for i in range(30):
    _, target = samples[i]

    sino_clean, _ = samples[i]
    # relative noise model: the foam's projections are not densities in mm
    y = ct_utils.sinogram_add_noise(
        sino_clean,
        I0=10000,
        sigma=sigma_paper,
        sigma_blur=sigma_blurred_paper,
        ks_value=3,
        model="relative",
    )
    noisy_list_y.append(y.view(sino_clean.shape))
    target_list.append(target)
    if i == 0:
        print(f"Min sino_clean: {sino_clean.min().item():.4f}")
        print(f"Max sino_clean: {sino_clean.max().item():.4f}")

sino_tensor_y = torch.stack(noisy_list_y)
target_tensor = torch.stack(target_list)
train_data = data_utils.TensorDataset(sino_tensor_y, target_tensor)
dataloader = DataLoader(train_data, batch_size=5, shuffle=True)

# Noise2Inverse
model = UNet().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

n2i_params = Noise2InverseSolver.default_parameters()
n2i_params.sino_split_count = 4
n2i_params.recon_fn = fdk
n2i_params.cali_J = Noise2InverseSolver.X_one_strategy(n2i_params.sino_split_count)

solver = Noise2InverseSolver(
    model, optimizer, loss_fn, n2i_params, geometry, verbose=True, device=device
)

solver.set_training(dataloader)
solver.train(500)
solver.save_final_results("final_model_N2I.pt", SAVE_DIR)
