# Standard imports
import matplotlib.pyplot as plt
import pathlib
import imageio
from tqdm import tqdm

# Torch imports
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# Lion imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_benchmarking_experiments as ct_benchmarking
from ts_algorithms import fdk, sirt, tv_min, nag_ls
from ts_algorithms.tv_min import tv_min2d
from LION.CTtools.ct_utils import make_operator

device = torch.device("cuda:1")
idx = 182

reg_params_f = [1e-10,1e-8,1e-7,1e-8,1e-6,5e-6,5e-6,5e-6,1e-8]

def extract_recon(experiment, reg_param, idx, savename):

    testing_data = experiment.get_testing_dataset()
    testing_dataloader = DataLoader(testing_data, 1, shuffle=False)

    # prepare operator for classical methods
    op = make_operator(experiment.geo)

    #model.eval()

    with torch.no_grad():
        for index, (data, target) in enumerate(tqdm(testing_dataloader)):
            if index == idx:
                recon = tv_min2d(op,data[0].to(device),lam=reg_param).detach().cpu().numpy().squeeze()

                np.save(str(savename[:-4]+".npy"),recon)
                imageio.imwrite(str(savename[:-4]+"_slice"+str(idx)+".tif"), recon)
    return None

# CAREFUL Change the idx in the path below.
savefolder = pathlib.Path("/export/scratch3/mbk/LION/bm_models/")
slicefolder = pathlib.Path("/export/scratch3/mbk/LION/slice_182_recons/")
# use min validation, or final result, whichever you prefer

model_names = ["CHP_FullData_min_val.pt",
"CHP_Limited120_min_val.pt", "CHP_Limited90_min_val.pt", "CHP_Limited60_min_val.pt",
"CHP_Sparse360_min_val.pt", "CHP_Sparse120_min_val.pt", "CHP_Sparse60_min_val.pt",
"CHP_LowDose_min_val.pt", "CHP_BeamHardening_min_val.pt"]

# CAREFUL change model here and in for loop at the bottom
#from LION.models.iterative_unrolled.LG import LG
from LION.models.post_processing.FBPUNet import FBPUNet

# Define all experiments in experiments list
experiments = []

# Standard dataset
experiments.append(ct_benchmarking.FullDataCTRecon())

# Limited angle
experiments.append(ct_benchmarking.LimitedAngle120CTRecon())
experiments.append(ct_benchmarking.LimitedAngle90CTRecon())
experiments.append(ct_benchmarking.LimitedAngle60CTRecon())

# Sparse angle
experiments.append(ct_benchmarking.SparseAngle360CTRecon())
experiments.append(ct_benchmarking.SparseAngle120CTRecon())
experiments.append(ct_benchmarking.SparseAngle60CTRecon())

# Low dose
experiments.append(ct_benchmarking.LowDoseCTRecon())

# Beam Hardening
experiments.append(ct_benchmarking.BeamHardeningCTRecon())


for model_idx, model_name in enumerate(tqdm(model_names)):
    savename = str(str(slicefolder.joinpath(model_name[:-11]))+".pdf")
    # CAREFUL change model here as well
    #model, options, data = FBPUNet.load(savefolder.joinpath(model_name))
    #model.to(device)
    extract_recon(experiments[model_idx],reg_params_f[model_idx], idx, savename)
