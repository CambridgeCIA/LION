# Standard imports
import matplotlib.pyplot as plt
import pathlib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Torch imports
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from tqdm import tqdm
from LION.utils.math import power_method

# Lion imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_benchmarking_experiments as ct_benchmarking
from LION.models.learned_regularizer.ACR import ACR
from LION.models.learned_regularizer.AR import AR
from LION.models.learned_regularizer.TDV import TDV
from LION.models.learned_regularizer.TDV_files.model import L2DenoiseDataterm

from ts_algorithms import fdk, sirt, tv_min, nag_ls


from LION.CTtools.ct_utils import make_operator

# Just a temporary SSIM that takes troch tensors (will be added to LION at some point)
def my_ssim(x: torch.tensor, y: torch.tensor, data_range=None):
    x = x.detach().cpu().numpy().squeeze()
    y = y.detach().cpu().numpy().squeeze()
    if data_range is None:
        data_range = x.max() - x.min()
    elif type(data_range) == torch.Tensor:
        data_range = data_range.cpu().numpy().squeeze()
    return ssim(x, y, data_range=data_range)


def my_psnr(x: torch.tensor, y: torch.tensor, data_range=None):
    x = x.detach().cpu().numpy().squeeze()
    y = y.detach().cpu().numpy().squeeze()
    if data_range is None:
        data_range = x.max() - x.min()
    elif type(data_range) == torch.Tensor:
        data_range = data_range.cpu().numpy().squeeze()
    return psnr(x, y, data_range=data_range)



def plot_supplementary_neuroIPS(recon:list, gt:list, indices:list, model_name:str, experiment_name:str,max_val:float=0.008,pctg_error:float=0.05):
    '''
    Recon: len = 3 list of the best, worst and mean reconstructions. They should be a 2D np array.
    gt: len = 3 list of the best, worst and mean ground truth. They should be a 2D np array.
    indices: len = 3 list of slice numbers for best, worst and mean reconstructions. They should be ints.
    model_name
    experiment_name (use the same as in the paper!)
    max_val: max value for the images (do not change)
    pctg_error: percentage of the error plot w.r.t. the image DISPLAY. Not data, DISPLAY! (do not change)
    '''
    example_folder = pathlib.Path("/store/DAMTP/zs334/lion_examples/")
    titles = ["Best", "Worst", "Mean"]

    fig = plt.figure(figsize=(8, 8))

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=None)
    plt.suptitle(f"{model_name} - {experiment_name}", fontsize=16)
    for i in range(3): #best/worst/mean
        title = titles[i] + f", slice #{indices[i]}"
        plt.subplot(3,3,i+1)
        plt.imshow(gt[i], cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.clim(0, max_val)

        plt.subplot(3,3,(i+1)+3)
        plt.imshow(recon[i], cmap="gray")
        plt.title(f"SSIM: {ssim(gt[i],recon[i],data_range=gt[i].max()-gt[i].min()):.2f}")
        plt.axis("off")
        plt.clim(0, max_val)

        plt.subplot(3,3,(i+1)+6)
        err = plt.imshow((gt[i] -recon[i]),cmap="seismic")
        if i==1:
            plt.title(f"Error")
        plt.axis("off")
        plt.clim(-max_val * pctg_error, max_val * pctg_error)
        cb_ax = fig.add_axes([.2,.08,.6,.012])
        cbar = fig.colorbar(err,orientation='horizontal',cax=cb_ax)
        cbar.set_ticks([-max_val * pctg_error, 0, max_val * pctg_error],labels=[f"-{pctg_error*100:.2f}%", "0", f"{pctg_error*100:.2f}%"])
        cbar.ax.tick_params(labelsize=10)
    plt.savefig(example_folder.joinpath(f"eval_{model_name}_{experiment_name}.png"), dpi=300,bbox_inches='tight', pad_inches=0)



# e.g. dict(method_name='AR_last', method = AR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ARFullDataCTRecon_check_0025.pt "),
def eval_experiment(evaluation,indices_eval = 128):
    savefolder = pathlib.Path("/store/DAMTP/zs334/LION/")
    fig_folder = pathlib.Path("/store/DAMTP/zs334/lion_figs/")
    example_folder = pathlib.Path("/store/DAMTP/zs334/lion_examples/")
    # from LION.models.post_processing.FBPMSDNet import FBPMS_D
    
    # model, options, data = ACR.load(savefolder.joinpath(model_name))
    # model = ACR.load(savefolder.joinpath(model_name))[0]
    model = evaluation['method'].load(savefolder.joinpath(evaluation['checkpoint']))[0]
    model.to(device)##???
    
    # model.load("/store/DAMTP/zs334/LION/ACR.pt")
    print('Model loaded')
    # indices = torch.arange(5)
    testing_data = evaluation['experiment'].get_testing_dataset()
    testing_data = data_utils.Subset(testing_data, indices_eval)
    testing_dataloader = DataLoader(testing_data, 1, shuffle=False)

    validation_data = evaluation['experiment'].get_validation_dataset()
    indices_val = torch.arange(50)
    validation_data = data_utils.Subset(validation_data, indices_val)
    validation_dataloader = DataLoader(validation_data, 1, shuffle=False)

    print(f'Data loaded testing: {len(testing_dataloader)}, validation: {len(validation_dataloader)}')

    ### if had method estimate then estimate otherwise nothing
    if hasattr(model, 'estimate_lambda'):
        model.estimate_lambda(dataset = validation_dataloader)
    # do more steps for acr
    # model.model_parameters.no_steps = 300


    # prepare models/algos
  

    op = make_operator(evaluation['experiment'].geo)

    # TEST 1: metrics
    test_ssim = np.zeros(len(testing_dataloader))
    test_psnr = np.zeros(len(testing_dataloader))

    fdk_ssim = np.zeros(len(testing_dataloader))
    fdk_psnr = np.zeros(len(testing_dataloader))



    op_norm = power_method(op)
    
    # model.model_parameters.step_size = 0.2/(op_norm)**2
    if ('AR' in evaluation['method_name'] or 'ACR' in evaluation['method_name']):
        print(model.model_parameters.step_size, 0.2/(op_norm)**2)
        if('BeamHardening' == evaluation['experiment_name']): 
            # model.model_parameters.step_size = 0.2*1e-2/(op_norm)**2
            # model.model_parameters.no_steps = 300
            model.lamb = model.lamb*1e-1

    outputs=[]
    gts=[]
    indices=[]
    # (recon:list, gt:list, indices:list, model_name:str, experiment_name:str,max_val:float=0.008,pctg_error:float=0.05):
    pbar = tqdm(enumerate(testing_dataloader), total=len(testing_dataloader))
    for index, (data, target) in pbar:
        pbar.set_description(f"Testing {index}")
        if 'TDV' in evaluation['method_name']:
            with torch.no_grad(): output = model.output(data.to(device))
        else: output = model.output(data.to(device),truth=target.to(device))
        outputs.append(output.detach().cpu().numpy().squeeze())
        gts.append(target.detach().cpu().numpy().squeeze())
        indices.append(indices_eval[index])
    plot_supplementary_neuroIPS(outputs, gts, indices, evaluation['algo_name'], evaluation['experiment_name'])
        


if __name__ == "__main__":
    #%% 1 - Settingts
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Device
    device = torch.device("cuda:3")
    torch.cuda.set_device(device)
    
    # Define your data paths
    savefolder = pathlib.Path("/store/DAMTP/zs334/LION/")

    evaluations=[]
    algo_names = ["AR", "TDV", "ACR",]
    experiment_names = ["FullData", "Limited120", "Limited90", "Limited60",
    "Sparse360", "Sparse120", "Sparse60", "LowDose", "BeamHardening"]

    indices_eval = []
    evaluations += [
        dict(algo_name='TDV', experiment_name='FullData'      ,method_name='TDV_best', method = TDV, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "TDVFullDataCTRecon_check_0008.pt"),
        dict(algo_name='TDV', experiment_name='Limited120'    ,method_name='TDV_best',  method = TDV, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "TDVLimitedAngle120CTRecon_check_0004.pt"),
        dict(algo_name='TDV', experiment_name='Limited90'     ,method_name='TDV_best', method = TDV, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "TDVLimitedAngle90CTRecon_check_0011.pt"),
        dict(algo_name='TDV', experiment_name='Limited60'     ,method_name='TDV_best', method = TDV, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "TDVLimitedAngle60CTRecon_check_0011.pt"),
        dict(algo_name='TDV', experiment_name='LowDose'       ,method_name='TDV_best', method = TDV, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "TDVLowDoseCTRecon_check_0006.pt"),
        dict(algo_name='TDV', experiment_name='Sparse360'     ,method_name='TDV_best', method = TDV, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "TDVSparseAngle360CTRecon_check_0011.pt"),
        dict(algo_name='TDV', experiment_name='Sparse120'     ,method_name='TDV_best', method = TDV, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "TDVSparseAngle120CTRecon_check_0011.pt"),
        dict(algo_name='TDV', experiment_name='Sparse60'      ,method_name='TDV_best', method = TDV, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "TDVSparseAngle60CTRecon_check_0011.pt"),
        dict(algo_name='TDV', experiment_name='BeamHardening' ,method_name='TDV_best', method = TDV, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "TDVBeamHardeningCTRecon_check_0004.pt"),
    ]
    
    indices_eval += [(205,368,122),
                     (205,238,121),
                     (205,371,121),
                     (205,26,121),
                     (79,186,122),
                     (205,368,122),
                     (205,194,122),
                     (205,101,121),
                     (79,226,460),
                     ]
    
    
    evaluations += [
        dict(algo_name='AR', experiment_name='FullData'      ,method_name='AR_best', method = AR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ARFullDataCTRecon_check_0017.pt"),
        dict(algo_name='AR', experiment_name='Limited120'    ,method_name='AR_best', method = AR, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "ARLimitedAngle120CTRecon_check_0008.pt"),
        dict(algo_name='AR', experiment_name='Limited90'     ,method_name='AR_best', method = AR, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "ARLimitedAngle90CTRecon_check_0003.pt"),
        dict(algo_name='AR', experiment_name='Limited60'     ,method_name='AR_best', method = AR, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "ARLimitedAngle60CTRecon_check_0025.pt"),
        dict(algo_name='AR', experiment_name='LowDose'       ,method_name='AR_best', method = AR, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "ARLowDoseCTRecon_check_0003.pt"),
        dict(algo_name='AR', experiment_name='Sparse360'     ,method_name='AR_best', method = AR, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "ARSparseAngle360CTRecon_check_0009.pt"),
        dict(algo_name='AR', experiment_name='Sparse120'     ,method_name='AR_best', method = AR, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "ARSparseAngle120CTRecon_check_0003.pt"),
        dict(algo_name='AR', experiment_name='Sparse60'      ,method_name='AR_best', method = AR, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "ARSparseAngle60CTRecon_check_0025.pt"),
        dict(algo_name='AR', experiment_name='BeamHardening' ,method_name='AR_best', method = AR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ARBeamHardeningCTRecon_check_0023.pt"),
    ]
    indices_eval += [(205,203,122),
                        (0, 189, 125),
                        (205,397,457),
                        (205,371,125),
                        (205,203,121),
                        (205,429,122),
                        (205,271,122),
                        (205,262,122),
                        (205,338,460),]
                        
    
    
    
    evaluations += [
        dict(algo_name='ACR', experiment_name='FullData'      ,method_name='ACR_best', method = ACR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ACRFullDataCTRecon_check_0025.pt"),
        dict(algo_name='ACR', experiment_name='Limited120'    ,method_name='ACR_best', method = ACR, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "ACRLimitedAngle120CTRecon_check_0009.pt"),
        dict(algo_name='ACR', experiment_name='Limited90'     ,method_name='ACR_best', method = ACR, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "ACRLimitedAngle90CTRecon_check_0010.pt"),
        dict(algo_name='ACR', experiment_name='Limited60'     ,method_name='ACR_best', method = ACR, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "ACRLimitedAngle60CTRecon_check_0004.pt"),
        dict(algo_name='ACR', experiment_name='LowDose'       ,method_name='ACR_best', method = ACR, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "ACRLowDoseCTRecon_check_0024.pt"),
        dict(algo_name='ACR', experiment_name='Sparse360'     ,method_name='ACR_best', method = ACR, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "ACRSparseAngle360CTRecon_check_0024.pt"),
        dict(algo_name='ACR', experiment_name='Sparse120'     ,method_name='ACR_best', method = ACR, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "ACRSparseAngle120CTRecon_check_0023.pt"),
        dict(algo_name='ACR', experiment_name='Sparse60'      ,method_name='ACR_best', method = ACR, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "ACRSparseAngle60CTRecon_check_0023.pt"),
        dict(algo_name='ACR', experiment_name='BeamHardening' ,method_name='ACR_best', method = ACR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ACRBeamHardeningCTRecon_check_0025.pt"),
    ]
    indices_eval += [
            (205,35,122),
            (205,431,460),
            (205,326,217),
            (205,147,121),
            (205,45,121),
            (205,32,122),
            (205,56,121),
            (0, 327, 122),
            (205,200,122)]
       

    for evaluation,indices in zip(evaluations,indices_eval):
        print('Evaluating:', evaluation['checkpoint'],indices)
        print(evaluation['method_name'])
        print(evaluation['checkpoint'])
        # eval_experiment(experiment,savefolder)
        eval_experiment(evaluation,indices_eval=torch.tensor((indices[0],indices[2],indices[1])))# was ordered as best mean worst, but for plotting its best worst mean


def plot_supplementary_neuroIPS(recon:list, gt:list, indices:list, model_name:str, experiment_name:str,max_val:float=0.008,pctg_error:float=0.05):
    '''
    Recon: len = 3 list of the best, worst and mean reconstructions. They should be a 2D np array.
    gt: len = 3 list of the best, worst and mean ground truth. They should be a 2D np array.
    indices: len = 3 list of slice numbers for best, worst and mean reconstructions. They should be ints.
    model_name
    experiment_name (use the same as in the paper!)
    max_val: max value for the images (do not change)
    pctg_error: percentage of the error plot w.r.t. the image DISPLAY. Not data, DISPLAY! (do not change)
    '''

    titles = ["Best", "Worst", "Mean"]

    fig = plt.figure(figsize=(8, 8))

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=None)
    plt.suptitle(f"{model_name} - {experiment_name}", fontsize=16)
    for i in range(3): #best/worst/mean
        title = titles[i] + f", slice #{indices[i]}"
        plt.subplot(3,3,i+1)
        plt.imshow(gt[i], cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.clim(0, max_val)

        plt.subplot(3,3,(i+1)+3)
        plt.imshow(recon[i], cmap="gray")
        plt.title(f"SSIM: {ssim(gt[i],recon[i],data_range=gt[i].max()-gt[i].min()):.2f}")
        plt.axis("off")
        plt.clim(0, max_val)

        plt.subplot(3,3,(i+1)+6)
        err = plt.imshow((gt[i] -recon[i]),cmap="seismic")
        if i==1:
            plt.title(f"Error")
        plt.axis("off")
        plt.clim(-max_val * pctg_error, max_val * pctg_error)
        cb_ax = fig.add_axes([.2,.08,.6,.012])
        cbar = fig.colorbar(err,orientation='horizontal',cax=cb_ax)
        cbar.set_ticks([-max_val * pctg_error, 0, max_val * pctg_error],labels=[f"-{pctg_error*100:.2f}%", "0", f"{pctg_error*100:.2f}%"])
        cbar.ax.tick_params(labelsize=10)
    plt.savefig(f"eval_{model_name}_{experiment_name}.png", dpi=300,bbox_inches='tight', pad_inches=0)