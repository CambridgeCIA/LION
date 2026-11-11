#%% This example shows how to train FBPConvNet for full angle, noisy measurements.

###START PARAMS
##
##
#
#
#
##



#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.models.CNNs.REDCNN import REDCNN
from LION.models.CNNs.UNets.GradDiscriminator import UNetGrad, SobelOperator
from LION.models.CNNs.UNets.ImageDiscriminator import UNetImage
from LION.utils.parameter import LIONParameter
from ts_algorithms import fdk
from cutmix import random_rectangle, batch_of_masks, cut_mix
import torch.nn.functional as F


import LION.experiments.ct_experiments as ct_experiments



#Choose loss function for the discriminators. By default it is using the MSE
def ls_gan(inputs, targets):
    try:
        return torch.mean((inputs - targets) **2 )
    except:
        return torch.mean((inputs.to(device) - targets.to(device)) ** 2)

#%%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/gscratch/uwb/ahuang54/saves/dugan")
datafolder = pathlib.Path(
    "/gscratch/uwb/LION_data/processed/LIDC-IDRI/"
)

redcnn_final_result_fname = savefolder.joinpath("Generator_final_iter.pt")
redcnn_checkpoint_fname = savefolder.joinpath("Generator_check_*.pt")
redcnn_discrim_validation_fname = savefolder.joinpath("Generator_min_val.pt")

grad_discrim_final_result_fname = savefolder.joinpath("grad_discrim_final_iter.pt")
grad_discrim_checkpoint_fname = savefolder.joinpath("grad_discrim_check_*.pt")


im_discrim_final_result_fname = savefolder.joinpath("im_discrim_final_iter.pt")
im_discrim_checkpoint_fname = savefolder.joinpath("im_discrim_check_*.pt")


#
#%% Define experiment
experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)

#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

#%% Define DataLoader``
# Use the same amount of training``
batch_size = 2
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)

#%% Model
# Default model is already from the paper.
generator = REDCNN()
im_discrim = UNetImage()
grad_discrim = UNetGrad()


    

# First, pretrain the models

generator.to(device)
im_discrim.to(device)
grad_discrim.to(device)
train_param = LIONParameter()

#train 
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

train_param.epochs = 75
train_param.learning_rate = 1e-3
train_param.loss = "MSELoss"

#choose optimizers
gen_optim = torch.optim.Adam(generator.parameters(), lr=train_param.learning_rate)
im_discrim_optim = torch.optim.Adam(im_discrim.parameters(), lr=train_param.learning_rate)
grad_discrim_optim = torch.optim.Adam(grad_discrim.parameters(), lr=train_param.learning_rate)

#Choose convolution for specialized discriminator. For the default Dugan an edge-highlighting filter is used
sobel = SobelOperator().to(device)


#specify data augmentation technique. Cutmix is used in DUGAN
def generate_mask(image_size, batchsize, min_cut, dim):
    return batch_of_masks(random_rectangle(min_cut=min_cut,image_size=image_size),dim = dim)


#Train the discriminators
def discrim_train(discrim,discrim_optim,sinogram,target_reconstruction,apply_cutmix=False):

    discrim_optim.zero_grad()
    bad_recon = torch.zeros(target_reconstruction.shape, device=device)

    for sino in range(sinogram.shape[0]):
        bad_recon[sino] = fdk(lidc_dataset.operator, sinogram[sino])
    
    gen_recon = generator(bad_recon)



    real_enc, real_dec = discrim(target_reconstruction)
    fake_enc, fake_dec = discrim(gen_recon)
    source_enc, source_dec = discrim(bad_recon)


    disc_loss = ls_gan(real_enc, 1.) + ls_gan(real_dec, 1.) + \
                ls_gan(fake_enc, 0.) + ls_gan(fake_dec, 0.) + \
                ls_gan(source_enc, 0.) + ls_gan(source_dec, 0.)

    total_loss = disc_loss

    #Add data augmentation for the generator
    if apply_cutmix:
        mask = generate_mask(image_size=512,batchsize =batch_size,min_cut = 50, dim = target_reconstruction.size()[0])
        mask.to(device)

        cutmix_enc, cutmix_dec = discrim(cut_mix(gen_recon,target_reconstruction,mask,device))

        cutmix_disc_loss = ls_gan(cutmix_enc, torch.zeros_like(cutmix_enc)) + ls_gan(cutmix_dec, mask)

        cr_loss = F.mse_loss(cutmix_dec, cut_mix(real_dec,fake_dec,mask,device))


        total_loss += cutmix_disc_loss + cr_loss * 5.08720932695335
   
    total_loss.backward()
    discrim_optim.step()

    return disc_loss.item()


#START PARAMS
grad_gen_loss_weight = .1
img_gen_loss_weight = .1
pix_loss_weight = 1
grad_loss_weight = 20

def train_gen(sinogram,target_reconstruction):
    generator.zero_grad()

    bad_recon = torch.zeros(target_reconstruction.shape, device=device)

    for sino in range(sinogram.shape[0]):
        bad_recon[sino] = fdk(lidc_dataset.operator, sinogram[sino])
    
    gen_recon = generator(bad_recon)

    img_gen_enc, img_gen_dec = im_discrim(gen_recon)
    img_gen_loss = ls_gan(img_gen_enc, 1.) + ls_gan(img_gen_dec, 1.)
    
    grad_gen_enc, grad_gen_dec = grad_discrim(gen_recon)
    grad_gen_loss = ls_gan(grad_gen_enc, 1.) + ls_gan(grad_gen_dec, 1.)

    total_loss = grad_gen_loss*grad_gen_loss_weight

    pix_loss = F.mse_loss(gen_recon,target_reconstruction)

    l1_loss = F.l1_loss(gen_recon, target_reconstruction)
    
    grad_gen_recon = sobel.forward(gen_recon)
    grad_target = sobel.forward(target_reconstruction)

    grad_loss = F.l1_loss(grad_gen_recon, grad_target)

    total_loss += img_gen_loss * img_gen_loss_weight + \
                pix_loss * pix_loss_weight + \
                grad_loss * grad_loss_weight
    
    total_loss.backward()
    gen_optim.step()    
        #add TV? or other regularizers

    return l1_loss.item(), total_loss.item()

#the conditional with this also has "or epoch == 0"
min_valid_loss = 1e10

def warmup(warmuptime,prob,curriter):
    return min(curriter * prob / warmuptime, prob)

start_epoch = 0
total_loss = []
warmupcounter = 0
for epoch in range(start_epoch, train_param.epochs):

    im_discrim.train()
    grad_discrim.train()
    generator.train()
    train_loss = [0,0,0,0]

    for index, (sinogram, target_reconstruction) in tqdm(
        enumerate(lidc_dataloader)
    ):
        warmupcounter +=1
        seed = np.random.rand()
        apply_cutmix = seed < warmup(1000,.7615524094697519,warmupcounter)
            

        #Training Image Discriminator
        im_loss = discrim_train(im_discrim, im_discrim_optim, sinogram, target_reconstruction,apply_cutmix=apply_cutmix)

        #Training Gradient Discriminator
        grad_loss = discrim_train(im_discrim, im_discrim_optim, sinogram, target_reconstruction,apply_cutmix=apply_cutmix)

        #Training Generator
        gen_l1_loss, gen_total_loss = train_gen(sinogram, target_reconstruction)

        #organize printouts for DUGAN_OUT
        train_loss[0] += im_loss
        train_loss[1] += grad_loss
        train_loss[2] += gen_l1_loss
        train_loss[3] += gen_total_loss

    total_loss.append(train_loss)
    print(f"Training for epoch {epoch+1} is complete! Starting Validation")

    # Validation
    valid_loss = 0.0
    generator.eval()
    for index, (sinogram, target_reconstruction) in tqdm(enumerate(lidc_validation)):
        bad_recon = torch.zeros(target_reconstruction.shape, device=device)

        for sino in range(sinogram.shape[0]):
            bad_recon[sino] = fdk(lidc_dataset.operator, sinogram[sino])
    
        reconstruction = generator(bad_recon)
        loss = loss_fcn(target_reconstruction, reconstruction)
        valid_loss += loss.item()

    print(
        f"Epoch {epoch+1} \t\t Training Loss: {train_loss[0] / len(lidc_dataloader)} \t\t Validation Loss: {valid_loss / len(lidc_validation)}"
    )


    #Save if validation loss decreased
    if min_valid_loss > valid_loss or epoch == 0:
        print(
            f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
        )
        min_valid_loss = valid_loss
        # Saving State Dict
        generator.save(
            redcnn_discrim_validation_fname,
            epoch=epoch + 1,
            training=train_param,
            loss=min_valid_loss,
        )

    print(f"Finished epoch {epoch+1}.")
    #print out epoch stats to DUGAN OUT
    try:
        print(f"Finished with Image Discriminator Loss of : {sum(total_loss[epoch][0])}")
        print(f"Finished with Gradient Discriminator Loss of : {sum(total_loss[epoch][1])}")
        print(f"Finished with Generator l1 (MSE with target) Loss of : {sum(total_loss[epoch][2])}")
        print(f"Finished with Generator Loss of : {sum(total_loss[epoch][3])}")
    except:
        try:
            print(f"Finished with Image Discriminator Loss of : {total_loss[-1][0]}")
            print(f"Finished with Gradient Discriminator Loss of : {total_loss[-1][1]}")
            print(f"Finished with Generator l1 (MSE with target) Loss of : {total_loss[-1][2]}")
            print(f"Finished with Generator Loss of : {total_loss[-1][3]}")
        except:
            print("Could not print loss")

    #Checkpoint saves
    try:
        if (epoch+1) % 25 == 0:
            generator.save_checkpoint(
                pathlib.Path(str(redcnn_checkpoint_fname).replace("*", f"{epoch+1:04d}")),
                epoch + 1,
                total_loss[-1][3],
                gen_optim,
                train_param,
                dataset=experiment.param,
            )
            im_discrim.save_checkpoint(
                pathlib.Path(str(im_discrim_checkpoint_fname).replace("*", f"{epoch+1:04d}")),
                epoch + 1,
                total_loss[-1][0],
                im_discrim_optim,
                train_param,
                dataset=experiment.param,
            ) 
            grad_discrim.save_checkpoint(
                pathlib.Path(str(grad_discrim_checkpoint_fname).replace("*", f"{epoch+1:04d}")),
                epoch + 1,
                total_loss[-1][1],
                grad_discrim_optim,
                train_param,
                dataset=experiment.param,
            )
    except:
        print("CHECK MODEL SAVE FAILED USE LATTER TOTAL LOSS INDEXING")
        if (epoch+1) % 25 == 0:
            generator.save_checkpoint(
                pathlib.Path(str(redcnn_checkpoint_fname).replace("*", f"{epoch+1:04d}")),
                epoch + 1,
                total_loss[epoch][3],
                gen_optim,
                train_param,
                dataset=experiment.param,
            )
            im_discrim.save_checkpoint(
                pathlib.Path(str(im_discrim_checkpoint_fname).replace("*", f"{epoch+1:04d}")),
                epoch + 1,
                total_loss[epoch][0],
                im_discrim_optim,
                train_param,
                dataset=experiment.param,
            ) 
            grad_discrim.save_checkpoint(
                pathlib.Path(str(grad_discrim_checkpoint_fname).replace("*", f"{epoch+1:04d}")),
                epoch + 1,
                total_loss[epoch][1],
                grad_discrim_optim,
                train_param,
                dataset=experiment.param,
            )


#Save the final generator and discriminators
generator.save(
    redcnn_final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
    dataset=experiment.param,
)
im_discrim.save(
    im_discrim_final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
    dataset=experiment.param,
)
grad_discrim.save(
    grad_discrim_final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
    dataset=experiment.param,
)