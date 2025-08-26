import pathlib
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Torch imports
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss

# Lion imports
import LION.experiments.ct_experiments as ct_experiments
from LION.classical_algorithms.fdk import fdk

# Model imports
# Replace to your model
from DeepFBP import DeepFBPNetwork
from FusionFBP import FusionFBPNetwork
from DeepFusionBP import DeepFusionBPNetwork

# Choose deivce
device = torch.device("cuda:2")
torch.cuda.set_device(device)

# Define experiment
experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")

# Dataset
lidc_dataset_test = experiment.get_testing_dataset()
lidc_test = DataLoader(lidc_dataset_test, 1, shuffle=False)

# Load model
# modify model name and model path
model_name= ['DeepFBP_I','DeepFBP_II', 'FusionFBP_I','FusionFBP_II','DeepFusionBP_I','DeepFusionBP_II']

def load_model(model_rec,path):
    model,_,_ =  model_rec.load(path)
    return model.to(device).eval()

models = {
    model_name[0]: load_model(DeepFBPNetwork,'/store/LION/gq217/trained_models/test_debugging/DeepFBP_I.pt'),
    model_name[1]: load_model(DeepFBPNetwork,'/store/LION/gq217/trained_models/test_debugging/DeepFBP_II.pt'),
    model_name[2]: load_model(FusionFBPNetwork,'/store/LION/gq217/trained_models/test_debugging/FuFBP_I.pt'),
    model_name[3]: load_model(FusionFBPNetwork,'/store/LION/gq217/trained_models/test_debugging/FuFBP_II.pt'),
    model_name[4]: load_model(DeepFusionBPNetwork,'/store/LION/gq217/trained_models/test_debugging/DeepFuBP_I.pt'),
    model_name[5]: load_model(DeepFusionBPNetwork,'/store/LION/gq217/trained_models/test_debugging/DeepFuBP_II.pt')
}


# Metrics
metric= {f"{name}_ssim": np.zeros(len(lidc_test)) for name in model_name}
metric2= {f"{name}_psnr": np.zeros(len(lidc_test)) for name in model_name}

def model_ssim(x, y):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())

def psnr(x,y, max_val=1.0):
    mse_fn = MSELoss()
    mse = mse_fn(x, y).item()
    return 10 * math.log10(max_val ** 2 / mse) if mse != 0 else float('inf')


# Dictionary for storing models and their performace
denoised_dict = {name: [] for name in model_name}
targets = []
ssim_stats = {}
all_in_one = False  # Parameter for generate comparison seperately or all in one.

# For boxplot
all_ssim_values =[]
all_psnr_values =[]
label_num= list(range(1,len(model_name)+1))

def boxplot_model(all_ssim_values,all_psnr_values):
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.title("Metrics")
    plt.boxplot(all_ssim_values)
    plt.xticks(label_num, model_name)
    #plt.xlabel("Model")
    plt.ylabel("SSIM")

    plt.subplot(2, 1, 2)
    plt.boxplot(all_psnr_values)
    plt.xticks(label_num, model_name)
    plt.xlabel("Model")
    plt.ylabel("PSNR")

    plt.tight_layout()
    plt.savefig(f"boxplot_model.png")
    plt.close()    
    
# Plot the grounth truth and reconstruction images with multiple models
def plot_images(all_in_one):
    if all_in_one == True:
        fig, axes = plt.subplots(len(model_name), 3, figsize=(18, 5*len(model_name)))
        fig.suptitle('Model Reconstruction Comparison (SSIM Max/Median/Min)', fontsize=20, y=0.98)        
        for row, modelname in enumerate(model_name):
            for col,stat_type in enumerate(['Max', 'Median', 'Min']):
                idx = ssim_stats[modelname][stat_type][1]
                target_img = targets[idx][0].squeeze()  # Groundth Truth 
                recon_img = denoised_dict[modelname][idx][0].squeeze()  # Reconstruction image
                ssim_val = ssim_stats[modelname][stat_type][0]
                psnr_val = ssim_stats[modelname][stat_type][2]
                axes[row, col].imshow(np.hstack((target_img, recon_img)), cmap='gray')
                axes[row, col].set_title(f'{modelname} - Index:{idx} -  {stat_type} SSIM: {ssim_val:.4f} \nPSNR: {psnr_val:.4f} \nGT vs Recon')
                if col == 1:
                    axes[row, col].set_title(f'{modelname} Average:{ssim_stats[modelname][stat_type][3]:.4f}\nIndex:{idx} -  {stat_type} SSIM: {ssim_val:.4f} \nPSNR: {psnr_val:.4f} \nGT vs Recon')
                axes[row, col].axvline(x=target_img.shape[1], color='r', linestyle='--')
                axes[row, col].axis('off')
        plt.tight_layout(pad=3.0)
        #plt.subplots_adjust(top=0.95)
        plt.savefig('all_models_comparison.png')
        plt.close()
        
    else:
        for row, modelname in enumerate(model_name):
            for col,stat_type in enumerate(['Max', 'Median', 'Min']):
                idx = ssim_stats[modelname][stat_type][1]
                target_img = targets[idx][0].squeeze()  # Groundth Truth 
                recon_img = denoised_dict[modelname][idx][0].squeeze()  # Reconstruction image
                ssim_val = ssim_stats[modelname][stat_type][0]
                psnr_val = ssim_stats[modelname][stat_type][2]
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(target_img, cmap='gray')
                axes[0].set_title(f'Ground Truth (Index: {idx})')
                axes[0].axis('off')
                axes[1].imshow(recon_img, cmap='gray')
                axes[1].set_title(f'{modelname} - Index:{idx} - \n{stat_type} SSIM: {ssim_val:.4f}\nPSNR: {psnr_val:.4f}')
                if col == 1:
                    axes[1].set_title(f'{modelname} Average:{ssim_stats[modelname][stat_type][3]:.4f}\nIndex:{idx}\n{stat_type} SSIM: {ssim_val:.4f}\nPSNR: {psnr_val:.4f}')
                axes[1].axis('off')
                plt.tight_layout()
                plt.savefig(f"{modelname}_{stat_type}_SSIM.png")
                plt.close()

#-------------------------- Main -----------------------------
with torch.no_grad():
    for i, (sino, target) in enumerate(lidc_test): # Load the test datasets
        sino = sino.to(device)
        target = target.to(device)
        recon = fdk(sino,op=experiment.geometry) 
        targets.append(target.cpu().numpy())

        for name, model in models.items():
            denoised = model(sino)
            denoised_dict[name].append(denoised.cpu().numpy())  # may need to be modify as it can lead to high memory usage
            metric[f"{name}_ssim"][i]= model_ssim(target[0], denoised[0])
            metric2[f"{name}_psnr"][i]= psnr(target[0], denoised[0])
            
    for idx, modelname in enumerate(model_name):
        ssim_values = list(metric[f"{modelname}_ssim"])
        max_ssim = np.max(ssim_values)
        min_ssim = np.min(ssim_values)
        median_ssim = np.median(ssim_values)
        mean = np.mean(ssim_values)
        max_idx = np.argmax(ssim_values)
        min_idx = np.argmin(ssim_values)
        median_idx = np.argsort(ssim_values)[len(ssim_values)//2]

        psnr_values = list(metric2[f"{modelname}_psnr"])
        # Corresponding PSNR value
        max_psnr = psnr_values[max_idx]      
        min_psnr = psnr_values[min_idx]
        median_psnr = psnr_values[median_idx]
        
        ssim_stats[modelname] = {
            'Max': (max_ssim, max_idx, max_psnr),
            'Median': (median_ssim, median_idx, median_psnr,mean),
            'Min': (min_ssim, min_idx, min_psnr)
        }       

        # Model boxplot
        all_ssim_values.append(ssim_values)
        all_psnr_values.append(psnr_values)        
    boxplot_model(all_ssim_values,all_psnr_values)

    # plot the grounth truth and reconstruction images with multiple models
    plot_images(all_in_one)  
    exit()
    


    




