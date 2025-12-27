TESTING = False
#%% Noise2Inverse train

#%% Imports

# Basic science imports
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
#from piq import gmsd
# from skimage.metrics import structural_similarity as ssim
# basic python imports

from tqdm import tqdm
import pathlib
import copy
import threading #locking our files just in case
import pdb
# LION imports
#import LION.models.iterative_unrolled.LPD import LPD
import LION.CTtools.ct_utils as ct

from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk, sirt
from skimage.transform import iradon

import pandas as pd


from LION.models.iterative_unrolled.LPD import LPD
print("LPD imported")
from LION.models.iterative_unrolled.LPD_SIRT import LPD_SIRT
print("LPD_SIRT imported")
from LION.models.iterative_unrolled.ItNet import ItNet
print("ItNet imported")
from LION.models.iterative_unrolled.ItNet_TV import ItNet as ItNetTV
print("ItNetTV imported")
from LION.models.iterative_unrolled.LG import LG
print("LG imported")
# from LION.models.iterative_unrolled.ADZnet import ItNetSobel
print("ItNetSobel imported")
from LION.models.iterative_unrolled.LEARN import LEARN
print("LEARN imported")
from LION.models.iterative_unrolled.ItNet_LBFGS import ItNet_LBFGS as ItNetLBFGS
print("ItNet LBFGS imported")
from LION.models.iterative_unrolled.ItNet_LBFGS_TV import ItNet_LBFGS as ItNet_LBFGSTV
print("ItNet LBFGSTV imported")
from LION.models.iterative_unrolled.ScharrNet import ItNetScharr
print("ItNetScharr imported")
from LION.models.iterative_unrolled.SobelNet import ItNetSobel

string_to_class = {}
string_to_class['LPD_FBP'] = LPD
string_to_class['LPD_SIRT'] = LPD_SIRT
string_to_class['ItNet_FBP'] = ItNet
string_to_class['ItNetTV_FBP'] = ItNetTV
string_to_class['LG_FBP'] = LG
string_to_class['ItNetSobel_FBP'] = ItNetSobel
string_to_class['LEARN_FBP'] = LEARN
string_to_class['ItNetLBFGS_FBP']=ItNetLBFGS
string_to_class['ItNetLBFGSTV_FBP']=ItNet_LBFGSTV
string_to_class['ItNetScharr_FBP']=ItNetScharr

#create dictionary to store data and convert to pandas datafram

data = {}


#converts NN and FBP output to one that scikit image can use torch tensor ->(numpy 2d array)
def tensor_to_scikit_input(x, y):
    try:
        x = x.cpu().numpy().squeeze()
        y = y.cpu().detach().numpy().squeeze()
    except(Exception):
        try:
            x = x.squeeze()
            y = y.cpu().numpy().squeeze()
        except(Exception):
            x = x.squeeze()
            y = y.cpu().detach().numpy().squeeze()
    return x, y, np.amax(x)-np.amin(x)

#Finds SSIM takes in pytorch tensors and outputs 1 numpy float32 I think  
def my_ssim(x, y):
    a, b, c = tensor_to_scikit_input(x, y)
    return ssim(a, b, data_range = c)

#peak signal to noise ratio
def my_PSNR(x, y):
    a, b, c = tensor_to_scikit_input(x, y)
    return psnr(a, b, data_range = c)

#mean square error
def my_MSE(x, y):
    a, b, c = tensor_to_scikit_input(x, y)
    return mse(a, b)

#FBP for low dose recon (default params) to write another check LIons.utils.ct_tools.geometry and check Lion.experiments.ct_experiments to see what kind of params are beign used
#scikit iradon needs an angle

def FBP_iradon(sinogram, angle = np.linspace(0,360,900, endpoint=False), filter_name='ramp'):
    return iradon(sinogram.cpu().numpy().squeeze().T, filter_name = filter_name, circle = False)

# def fdk(op, sinogram.cpu().squeeze(),angle=np.linspace(0, 2*np.pi, 360, endpoint=False), filter_name = 'ramp'):
#     return iradon(sinogram.cpu().numpy().squeeze().T, theta=angle, filter_name=filter_name, circle=False)

def FBP_fdk(op,sinogram, truth = 0):
    unclamped = fdk(op, sinogram.cpu().squeeze())
    #sinogram is 4 channels Batch size,views , detectors, color channels (should be one)
    #batch size, channels, width, height
    # if clamp and (type(truth) != type(0)):
    #     return np.clip(unclamped, 0 , np.amax(truth))
    return unclamped


mylock = threading.Lock()
model_name = "LPD"



#%%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path(f"/gscratch/uwb/CTImages25/Scripts/")
datafolder = pathlib.Path(
    "/gscratch/uwb/LION_data/processed/LIDC-IDRI/"
)

#0 is 
final_result_fname = {}

EXP = ["LowDose","SparseView", "LimitedAngle", "ExLowDose"]

# final_result_fname["LowDose_LPD"] = pathlib.Path("/gscratch/uwb/CTImages25/Scripts/ALL_MODELS/LPD.pt")


models_path = f"./ALL_MODELS/"

model_files = glob.glob(models_path + "*.pt")

model_paths = []
for i , path in enumerate(model_files):
    file_name = path[len(models_path)::].replace("/","")

    fart = file_name.replace(".pt","").split("_")
    fart.append(file_name)
    model_paths.append(fart)


    #[MODELNAMECLASS(STRING), EXPERIMENT(STRING), MODIFIER(STRING), pt FILE PATH]






# checkpoint_fname = savefolder.joinpath("LPD_check_*.pt")
#
#%% Define experiment
def experiment_load_buffer(exptype):
    if exptype == "LowDose":
        return ct_experiments.LowDoseCTRecon(datafolder = datafolder)
    elif exptype == "SparseView":
        return ct_experiments.SparseAngleCTRecon(datafolder = datafolder)
    elif exptype == "LimitedAngle":
        return ct_experiments.LimitedAngleCTRecon(datafolder = datafolder)
    elif exptype == "ExLowDose":
        return ct_experiments.ExtremeLowDoseCTRecon(datafolder = datafolder)
    else:
        print("UNRECOGNIZED EXPERIMENT TYPE LINE 137")
        quit()

#[MODELNAMECLASS(STRING), EXPERIMENT(STRING), MODIFIER(STRING), pt FILE PATH]
models = []

calc_trad_model = {}
for exp in EXP:
    calc_trad_model[exp] = True


for list in model_paths:

    # print(list)

    experiment = experiment_load_buffer(list[1])
    op = ct.make_operator(experiment.geometry)
    dataset = experiment.get_testing_dataset()
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    NN_model, NN_param, NN_data = string_to_class[f"{list[0]}_{list[2]}"].load("/gscratch/uwb/CTImages25/Scripts/ALL_MODELS/"+list[3])
    NN_model.eval()
    print(f"{list[0]}_{list[2]}_{list[1]} LOADED")

    models.append((NN_model, dataloader, op, f"{list[0]}_{list[2]}_{list[1]}", list[1]))


#If the person running the program would like to save images, then create a folder under the designated savepath labeled with the run number



# loop trhough testing data
for iiii, model in enumerate(models):
    NN_model, dataloader, op, name, exp_name = model

    pathname = f"./../Images/raw/"
    # output_path = savefolder / pathname
    # output_path.mkdir(parents = True, exist_ok = True)

    print(f"LOADING {name}")

    for index, (sinogram, target_reconstruction) in tqdm(enumerate(dataloader)):

        if(TESTING and (index == 20)):
            print("\nYOU ARE IN TESTING MODE\n")
            break


        #find SSIM between ground truth and LPD and append to list
        neural_out = NN_model(sinogram)
        

        if(calc_trad_model[exp_name]):


            fbp_image = fdk(op, sinogram[0].cpu()).squeeze()
            tr_image = target_reconstruction.cpu().numpy().squeeze()
            sirt_image = sirt(op, sinogram.cpu().squeeze(), 50, tr_image.min(), tr_image.max()).squeeze()

            np.save(pathname + "GROUNDTRUTH_" + str(index), tr_image)
            np.save(pathname + f"Conv_FBP_{exp_name}_" + str(index), fbp_image.numpy())
            np.save(pathname + f"Conv_SIRT50_{exp_name}_" + str(index), sirt_image.numpy())
            
        
        np.save(pathname + f"{name}_{str(index)}", neural_out.cpu().detach().numpy().squeeze())
        


        #save up to 10 of the first images
        # if(index ==9):
        #     break
    calc_trad_model[exp_name] = False


#Updating the version file 

# vers = int(vers) + 1
# mylock.acquire()
# with open("./../Version/version.txt", 'w') as f:
#         f.write(str(vers))
# mylock.release()
# print(f'\nYour result is the {vers-1}th version')
