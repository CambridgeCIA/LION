#When this is True, the model will only process 20 images from the dataset
TESTING = False

#This is a generalized file to save images and performance metrics from a given nn model in LION
#NECESSARY CHANGES:

#1. find and replace: "modelname" , "MODELNAME", and the import statement for the model 
# (The modification against the standards MUST be described here for example, LPD_SIRT)
# The images are saved thsi way and if unchanged the csvs will overwrite
#
#      ex. "modelname" <- "lpd_sirt"
#      ex. from LION.models.iterative_unrolled.MODELNAME import MODELNAME
#please comment under here what yo have changed model and MODELNAME to for stats troubleshooting purposes

#      "modelname" : ""
#      "MODELNAME" : ""


#2. change the loading file to the file path of your trained neural network
#      ctrl + f :  "CHANGE FILE PATH OF TRAINED NEURAL NETWORK"

#3 Change the experiment type so it saves in the right results folder and so LION loads the correct experiment params
#     ctrl + f : "SELECT ONE STRING AND DELETE THE REST"
#      ex.   experiment_type = "LowDose"
#     ctrl + f : "Define Experiment"


#5. Pay special attention to the import It should import from LION iterative unrolled but your choice of model name may not
# align with LION's if you do not like their naming convention for the CSVs and image displays figure it out yourself. I believe in you <3

#FIRST TIME SET UP:

#3. if there is not a version.txt please specify one (this is a txt file witha  number in it to prevent images from over writing)
#      ctrl + f : "PATH TO VERSION FILE"

#4. add a folder for the csvs to save into and images
#     ctrl + f : #DATASET FOR TESTING AND SAVE PATH OF CSV AND IMAGES

#NOT REQUIRED BELOW HERE:

#YOU MAY CHOOSE EXTRA METRICS TO IMPORT HERE:
#For example if you would like to include Mean Squared Error ctrl + f <- "DEFINE NAME AND FUNCTION HERE METRICS"


Clamping = True
#%% Imports
# Basic science imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse


from tqdm import tqdm
import pathlib
import copy
import threading
import pdb
import LION.CTtools.ct_utils as ct
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk
from skimage.transform import iradon

import pandas as pd

#create dictionary to store data and convert to pandas datafram

data = {}



#FBP for low dose recon (default params) to write another check LIons.utils.ct_tools.geometry and check Lion.experiments.ct_experiments to see what kind of params are beign used
#scikit iradon needs an angle

def FBP_iradon(sinogram, angle = np.linspace(0,360,900, endpoint=False), filter_name='ramp'):
    return iradon(sinogram.cpu().numpy().squeeze().T, filter_name = filter_name, circle = False)

# def fdk(op, sinogram.cpu().squeeze(),angle=np.linspace(0, 2*np.pi, 360, endpoint=False), filter_name = 'ramp'):
#     return iradon(sinogram.cpu().numpy().squeeze().T, theta=angle, filter_name=filter_name, circle=False)

def FBP_fdk(op, sinogram, target_range):
    unclamped = fdk(op, sinogram.cpu().squeeze())

    #sinogram is 4 channels Batch size,views , detectors, color channels (should be one)
    #batch size, channels, width, height
    if Clamping and (target_range != 0):
        return np.clip(unclamped, 0 , target_range)
    return unclamped

print(f"Clamping is : {Clamping}")
mylock = threading.Lock()

#MODELNAME_EXPERIMENT_MODIFIER
#exp. LPD_LowDose_FBP
model_name = "DEFAULT"
model_path = "DeFAULT_DEFAULT_DEFAULT"
#SELECT ONE STRING AND DELETE THE REST

experiment_type = ["LowDose","SparseView", "LimitedAngle", "ExLowDose"]

#%%
# % Chose device:


#prompt the user for how many images they would like to save images limited to 10 total default to 5
try:
    numimages = int(input("How many images would you like to save? "))
except(Exception):
    print("Invalid input setting to default of 5")
    numimages = 5
numimages = 10 if (numimages > 10) else (0 if (numimages < 0) else numimages)


#see if the user is trying to save the image comparison metrics by 
longsave = "fartpoopoodefault"
try:
    longsave = input("Would you like to save just the comparison metrics (y, yes)? \n(will save FBP, but will significantly slow the script ETA ~6mins for 390 test images)\n ")
except(Exception):
    print("Invalid input. Not calculating FBPs")

longsave = (longsave.lower() == "y") or (longsave.lower() == "yes")


vers = ""

#PATH TO VERSION FILE
if(numimages > 0 or longsave):
    try:
    	with open('./version.txt', 'r') as file:
            vers = file.read()
            vers.strip()
        
    except(FileNotFoundError):
    	print("File was not found")


device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths


#DATASET FOR TESTING AND SAVE PATH OF CSV AND IMAGES
savecsvfolder = pathlib.Path(f"/gscratch/uwb/CTImages25/CSV/{experiment_type}/")
savefolder = pathlib.Path(f"/gscratch/uwb/CTImages25/Images/{experiment_type}/")
datafolder = pathlib.Path(
    "/gscratch/uwb/LION_data/processed/LIDC-IDRI/"
)



# CHANGE FILE PATH OF TRAINED NEURAL NETWORK
final_result_fname =pathlib.Path(f"/gscratch/uwb/CTImages25/Scripts/ALL_MODELS/{model_path}.pt")



#%% Define experiment
experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)

op = ct.make_operator(experiment.geometry)




#%% Dataset
dataset = experiment.get_testing_dataset()
batch_size = 1
dataloader = DataLoader(dataset, batch_size, shuffle=False)

#%% Load model
MODELNAME_model, MODELNAME_param, MODELNAME_data = MODELNAME.load(final_result_fname)
MODELNAME_model.eval()


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

def my_image_range(x,y):
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
    return np.amax(y)-np.amin(y)
#DEFINE NAME AND FUNCTION HERE METRICS (examples below (return a number or imported function value that will be saved to the csv))
#After defining the function, include it's column name as the KEY in the data_types dictionary and function name as the value pair
            #gradient magnitude similarity deviation (asses similarity of image gradients (does it preserve sharpness and detail))
            # def my_GMSD(x, y):
            #     pass

            #weighted spectral distance
            #   pass

#declare types of data and function names
data_types = {'SSIM' : my_ssim, 'MSE' : my_MSE, 'PSNR' : my_PSNR, 'IMG_RANGE' : my_image_range}



#If the person running the program would like to save images, then create a folder under the designated savepath labeled with the run number
if(numimages > 0):
    pathname = f"{model_name}_test_images_"+vers
    output_path = savefolder / pathname
    output_path.mkdir(parents = True, exist_ok = True)

#list to store SSIM values for each image
diff = []
lowest = [[],[],[],[]]
best = [[],[],[],[]]


if (longsave):
    for name in data_types:
        data[f'{model_name}_{name}'] = []
        data[f'FBP_{name}'] = []



# loop trhough testing data
for index, (sinogram, target_reconstruction) in tqdm(enumerate(dataloader)):

    if(TESTING and (index == 20)):
        print("\nYOU ARE IN TESTING MODE\n")
        break


    #find SSIM between ground truth and MODELNAME and append to list
    if(Clamping):
        target_range = np.amax(target_reconstruction.cpu().numpy().squeeze()) - np.amin(target_reconstruction.cpu().numpy().squeeze())
        modelname_out = MODELNAME_model(sinogram)
        modelname_out = torch.clamp(modelname_out, min =0, max=target_range)
    else:
        modelname_out = MODELNAME_model(sinogram)
        target_range = 0


    if (longsave):
        FBP = FBP_fdk(op, sinogram, target_range)
        for name, comparison_function in data_types.items():
            data[f'{model_name}_{name}'].append(comparison_function(target_reconstruction, modelname_out))
            data[f'FBP_{name}'].append(comparison_function(target_reconstruction, FBP))
        continue
    else:
        currssim = my_ssim(target_reconstruction, modelname_out)
        diff.append(currssim)

    #save up to 10 of the first images
    if(index < numimages and numimages > 0):
        
        FBP = FBP_fdk(op, sinogram, target_range)

        #save target reconstruction
        TRA = target_reconstruction.cpu().numpy().squeeze()
        ground_truth_filename = "ground_truth_" + str(index) + ".png"
        plt.imsave(output_path / ground_truth_filename , TRA, cmap = 'gray')

        #save neural network's reconstruction
        MODELNAME_O = modelname_out.cpu().detach().numpy().squeeze()
        modelname_filename = f"{model_name}_reconstruction_" + str(index) + ".png"
        plt.imsave(output_path / modelname_filename, MODELNAME_O, cmap = 'gray')

        #save filtered back projection

        #save images side by side
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(TRA, cmap = 'gray')
        axs[0].set_title('Ground Truth')
        axs[0].set_xlabel('SSIM: 1')

        axs[1].imshow(FBP.squeeze(), cmap = 'gray')
        axs[1].set_title('FBP')
        axs[1].set_xlabel(f'SSIM: {my_ssim(target_reconstruction,FBP)}')

        axs[2].imshow(MODELNAME_O, cmap = 'gray')
        axs[2].set_title('MODELNAME')
        axs[2].set_xlabel(f'SSIM: {currssim}')

        
        fig.suptitle(f'{experiment_type} Reconstruction Comparison #{index+1}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        side_by_side_name = f'side_by_side_{index}.png'
        plt.savefig(output_path / side_by_side_name)



    if index < 3:
        FBP = FBP_fdk(op, sinogram, target_range)
        FBP = FBP.squeeze()
        lowest[0].append(currssim)
        lowest[1].append(modelname_out)
        lowest[2].append(target_reconstruction)
        lowest[3].append(FBP)
        best[0].append(currssim)
        best[1].append(modelname_out)
        best[2].append(target_reconstruction)
        best[3].append(FBP)

    
    i = lowest[0].index(max(lowest[0]))
    if(lowest[0][i] > currssim):
        FBP = FBP_fdk(op, sinogram, target_range)
        FBP = FBP.squeeze()
        lowest[0][i] = currssim
        lowest[1][i] = modelname_out.cpu().detach().numpy().squeeze()
        lowest[2][i] = target_reconstruction.cpu().numpy().squeeze()
        lowest[3][i] = FBP

    i = best[0].index(min(best[0]))
    if(best[0][i] < currssim):
        FBP = FBP_fdk(op, sinogram, target_range)
        FBP = FBP.squeeze()
        best[0][i] = currssim
        best[1][i] = modelname_out.cpu().detach().numpy().squeeze()
        best[2][i] = target_reconstruction.cpu().numpy().squeeze()
        best[3][i] = FBP


    #prioritize saving specific images

for i in range(len(lowest[0])):
    lowest_quality_recon_name = "lowest_quality_recon_" + str(i) + ".png"
    lowest_quality_ground_name = "lowest_quality_ground_" + str(i) + ".png"
    lowest_quality_FBP_name = "lowest_quality_FBP_" + str(i) + ".png"
    MODELNAME_O = lowest[1][i]
    TR = lowest[2][i]
    FBP = lowest[3][i]
    plt.imsave(output_path / lowest_quality_recon_name, MODELNAME_O, cmap = 'gray')
    plt.imsave(output_path / lowest_quality_ground_name, TR, cmap = 'gray')
    plt.imsave(output_path / lowest_quality_FBP_name, FBP, cmap = 'gray')

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(TR, cmap = 'gray')
    axs[0].set_title('Ground Truth')
    axs[0].set_xlabel('SSIM: 1')

    axs[1].imshow(FBP.squeeze(), cmap = 'gray')
    axs[1].set_title('Filtered Back Projection (ramp)')
    axs[1].set_xlabel(f'SSIM: {my_ssim(TR,FBP)}')

    axs[2].imshow(MODELNAME_O, cmap = 'gray')
    axs[2].set_title(f'{model_name}')
    axs[2].set_xlabel(f'SSIM: {lowest[0][i]}')

    
    fig.suptitle(f'{experiment_type} Worst SSIM Reconstruction Comparison #{i+1}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path / f'worst_side_by_side_{i}.png')

for i in range(len(best[0])):
    best_quality_recon_name = "best_quality_recon_" + str(i) + ".png"
    best_quality_ground_name = "best_quality_ground_" + str(i) + ".png"
    best_quality_FBP_name = "best_quality_FBP_" + str(i) + ".png"
    MODELNAME_O = best[1][i]
    TR = best[2][i]
    FBP = best[3][i]

    #save individually
    plt.imsave(output_path / best_quality_recon_name, MODELNAME_O, cmap = 'gray')
    plt.imsave(output_path / best_quality_ground_name, TR, cmap = 'gray')
    plt.imsave(output_path / best_quality_FBP_name, FBP, cmap = 'gray')

    #save side by side
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(TR, cmap = 'gray')
    axs[0].set_title('Ground Truth')
    axs[0].set_xlabel('SSIM: 1')

    axs[1].imshow(FBP.squeeze(), cmap = 'gray')
    axs[1].set_title('Filtered Back Projection (ramp)')
    axs[1].set_xlabel(f'SSIM: {my_ssim(TR,FBP)}')

    axs[2].imshow(MODELNAME_O, cmap = 'gray')
    axs[2].set_title(f'{model_name}')
    axs[2].set_xlabel(f'SSIM: {best[0][i]}')

    
    fig.suptitle(f'{experiment_type} Best SSIM Reconstruction Comparison #{i+1}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path / f'Best_side_by_side_{i}.png')


    
if(not longsave):
    x_axis = np.array([i for i in range(len(diff))])
    y_axis = np.array(diff)

    plt.figure()
    plt.scatter(x_axis,y_axis)
    plt.xlabel("Current Image")
    plt.ylabel("SSIM")
    plt.savefig(output_path / "SSIM_graph.png")
else:
    df = pd.DataFrame.from_dict(data)
    df.to_csv(savecsvfolder / f"{model_name}.csv", index = False)


#Updating the version file 
#PATH TO VERSION FILE
vers = int(vers) + 1
mylock.acquire()
with open("./version.txt", 'w') as f:
        f.write(str(vers))
mylock.release()
print(f'\nYour result is the {vers-1} Dr. TH version')
    # do whatever you want with this.


#add grayscale for images

#add fpb side by side for easy comparison

#turn off shuffle

#look at FDK see if it's for 3d or 2s (cone vs fan)

