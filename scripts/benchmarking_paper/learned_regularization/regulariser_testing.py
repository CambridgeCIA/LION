#%% This python file is for checking how optimisation of the variational problem is going for ACR

#%% 0 - Imports
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Standard imports
import matplotlib.pyplot as plt
import pathlib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import LION.utils.utils as ai_utils

# Torch imports
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import numpy as np

# Lion imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_benchmarking_experiments as ct_benchmarking
from LION.models.learned_regularizer.ar_loss import WGAN_gradient_penalty_loss as ar_loss

# Just a temporary SSIM that takes troch tensors (will be added to LION at some point)
def my_ssim(x: torch.tensor, y: torch.tensor):
    if x.shape[0]==1:
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        return ssim(x, y, data_range=x.max() - x.min())
    else: 
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        vals=[]
        for i in range(x.shape[0]):
            vals.append(ssim(x[i], y[i], data_range=x[i].max() - x[i].min()))
        return np.array(vals)

def my_psnr(x: torch.tensor, y: torch.tensor):
    if x.shape[0]==1:
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        return psnr(x, y, data_range=x.max() - x.min())
    else: 
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        vals=[]
        for i in range(x.shape[0]):
            vals.append(psnr(x[i], y[i], data_range=x[i].max() - x[i].min()))
        return np.array(vals)


#%% 1 - Settingts
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Device
device = torch.device("cuda:3")
torch.cuda.set_device(device)

# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/zs334/LION/")
# savefolder = pathlib.Path("/store/DAMTP/ab2860/trained_models/test_debbuging/")

# Filenames and patters
final_result_fname = savefolder.joinpath("AR.pt")
checkpoint_fname = "AR_check_*.pt"  # if you use LION checkpoiting, remember to have wildcard (*) in the filename
validation_fname = savefolder.joinpath("AR_min_val.pt")

#%% 2 - Define experiment
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These are all the experiments we need to run for the benchmarking

# Standard dataset
# experiment = ct_benchmarking.FullDataCTRecon()
# Limited angle
# experiment = ct_benchmarking.LimitedAngle150CTRecon()
# experiment = ct_benchmarking.LimitedAngle120CTRecon()
# experiment = ct_benchmarking.LimitedAngle90CTRecon()
# experiment = ct_benchmarking.LimitedAngle60CTRecon()
# Sparse angle
# experiment = ct_benchmarking.SparseAngle720CTRecon()
experiment = ct_benchmarking.SparseAngle360CTRecon()
# experiment = ct_benchmarking.SparseAngle180CTRecon()
# experiment = ct_benchmarking.SparseAngle120CTRecon()
# experiment = ct_benchmarking.SparseAngle90CTRecon()
# experiment = ct_benchmarking.SparseAngle60CTRecon()
# Low dose
# experiment = ct_benchmarking.LowDoseCTRecon()
# Beam Hardening
# experiment = ct_benchmarking.BeamHardeningCTRecon()

#%% 3 - Obtaining Datasets from experiments
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
training_data = experiment.get_training_dataset()
validation_data = experiment.get_validation_dataset()
testing_data = experiment.get_testing_dataset()

# smaller dataset for testing if this template worksfor you.
##############################################################
# REMOVE THIS CHUNK IN THE FINAL VERSION
indices = torch.arange(1000)
training_data = data_utils.Subset(training_data, indices)
indices_val = torch.arange(20)
validation_data = data_utils.Subset(validation_data, indices_val)
# testing_data = data_utils.Subset(testing_data, indices_val)

# REMOVE THIS CHUNK IN THE FINAL VERSION
##############################################################

#%% 4 - Define Data Loader
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is standard pytorch, no LION here.

batch_size = 1
training_dataloader = DataLoader(training_data, batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size, shuffle=False)
testing_dataloader = DataLoader(testing_data, batch_size, shuffle=False)

#%% 5 - Load Model
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We show here how to do it for LPD, but you can do it for any model in LION

from math import inf
x_max = -inf
x_min = inf
for x in data_utils.Subset(training_data, torch.arange(100)):
    x_max = max(x[1].max(), x_max)
    x_min = min(x[1].min(), x_min)
# print(x_max,x_min)

from LION.models.learned_regularizer.ACR import ACR
from LION.models.learned_regularizer.AR import AR


# If you are happy with the default parameters, you can just do
# model = ACR(experiment.geo).to(device)
# Remember to use `experiment.geo` as an input, so the model knows the operator


# If you want to modify the default parameters, you can do it like this
# Default model is already from the paper. We can get the config of the detault by
default_parameters = AR.default_parameters()

# You can modify the parameters as wished here.
default_parameters.xmin = x_min.item()
default_parameters.xmax = x_max.item()
# default_parameters.n_iters = 5
# Now create the actual model. Remember to use `experiment.geo` as an input, so the model knows the operator
model = AR(experiment.geo, default_parameters).to(device)
# Now to estimate the model parameter
# model.load("/store/DAMTP/zs334/LION/ACR.pt")
model = AR.load("/store/DAMTP/zs334/LION/ARSparseAngle360CTRecon_check_0009.pt")[0].to(device)
print('Model loaded in')

model.estimate_lambda(dataset = validation_dataloader)

#%% 6 - Define Loss and Optimizer
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is standard pytorch, no LION here.

# loss fn
loss_fcn = ar_loss()
optimiser = "adam"

# optimizer
epochs = 10#15
learning_rate = 1e-3
betas = (0.9, 0.99)
loss = "MSELoss"
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)

#%% 7 - Define Solver
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# if your model is trained with a supported solver (for now only supervisedSolver and Noise2Inverse_solver), you can use the following code.
# If you have a custom training loop, you can just use that, pure pytorch is supported.
# Note: LIONmodel comes with a few quite useful functions, so you might want to use it even if you have a custom training loop. e.g. model.save_checkpoint() etc.
# Read demo d04_LION_models.py for more info.

# You know how to write pytorch loops, so let me show you how to use LION for training.


from LION.optimizers.weaklysupervised_learning import weaklysupervisedSolver



# create solver
solver = weaklysupervisedSolver(model, optimiser, loss_fcn, verbose=True)

# YOU CAN IGNORE THIS. You can 100% just write your own pytorch training loop.
# LIONSover is just a convinience class that does some stuff for you, no need to use it.

# set data
solver.set_training(training_dataloader)
# Set validation. If non defined by user, it uses the loss function to validate.
# If this is set, it will save the model with the lowest validation loss automatically, given the validation_fname
solver.set_validation(
    validation_dataloader, validation_freq=1, validation_fn=torch.nn.MSELoss(), validation_fname=validation_fname
)

# Set testing. Second input has to be a function that accepts torch tensors and returns a scalar
solver.set_testing(testing_dataloader, my_ssim)
# set checkpointing procedure. It will automatically checkpoint your models.
# If load_checkpoint=True, it will load the last checkpoint available in disk (useful for partial training loops in HPC)


#### Setup wandb logging
import wandb
options = LIONParameter()
# we need to at least know exactly when the model was saved, to at least be able to reproduce.
options.commit_hash = ai_utils.get_git_revision_hash()
options.model_name = model.__class__.__name__
options.model_parameters = model.model_parameters

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="LION_"+model.__class__.__name__,
    # track hyperparameters and run metadata
    config=options,
)



#%% 8 - TRAIN!
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from LION.utils.math import power_method
print((power_method(model.op)))
image_array = []
model.train()
for index, (data, target) in enumerate(validation_dataloader):
    # turn bchw into whc
    print(0.2/(model.op_norm)**2)
    image_array.append(target.permute(0, 2, 3, 1).cpu().detach().numpy())  
    for lr in [5e-8,0.2/(model.op_norm)**2]:#[1e-7,5e-7,1e-6,3e-6,6e-6]:
        # model.model_parameters.step_size = lr
        output = model.output(data.to(device),truth = target.to(device))
        print(f"SSIM: {my_ssim(target.to(device),output)}, PSNR: {my_psnr(target.to(device),output)}")
        image_array.append(output.permute(0, 2, 3, 1).cpu().detach().numpy())   
    break
images = (np.stack(image_array)).reshape(-1,output.shape[2],output.shape[3],1)
images = [wandb.Image((images[i]-images[i].min())/(images[i].max()-images[i].min()), caption="Reconstruction examples") for i in range(images.shape[0])]
wandb.log({"examples": images})


# Save the training.
# plt.figure()
# plt.semilogy(solver.train_loss)
# plt.savefig("loss.png")



# [optional] finish the wandb run, necessary in notebooks
wandb.finish()


# Now your savefolder should have the min validation and the final result.

#%% 9 - TEST
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# if verbose it will print mean+std
# result_vals_nparray = solver.test()

#%% 10 - Non-ML recon
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# these are pre-made for you
# from ts_algorithms import fdk, sirt, tv_min, nag_ls

# # use this function to make an operator.
# # If you want to do this inside the LIONmodel, please use its make_operator function, not this one, as this one does not make it autograd.
# from LION.CTtools.ct_utils import make_operator

# op = make_operator(experiment.geo)

# # just call with op.
# # See more in the ts_algorithms repository, this is not LION
# recon = fdk(op, testing_data[0])
# recon = sirt(
#     op, testing_data[0], 200
# )  # SIRT: almost Gradient descend, actually kazmarz method
# recon = tv_min(op, testing_data[0], 200, 0.1)  # Chambolle-Pock TV
# recon = nag_ls(
#     op, testing_data[0], 100
# )  # Nesterov Accelerated Gradient with Least Squares
