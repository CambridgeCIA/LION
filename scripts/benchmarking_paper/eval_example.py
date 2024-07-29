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


# e.g. dict(method_name='AR_last', method = AR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ARFullDataCTRecon_check_0025.pt "),
def eval_experiment(evaluation,index_eval = 128):
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
    testing_data = data_utils.Subset(testing_data, torch.tensor([index_eval]))
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


    pbar = tqdm(enumerate(testing_dataloader), total=len(testing_dataloader))
    for index, (data, target) in pbar:
        pbar.set_description(f"Testing {index}")
        if 'TDV' in evaluation['method_name']:
            with torch.no_grad(): output = model.output(data.to(device))
        else: output = model.output(data.to(device),truth=target.to(device))
        # print(index, output, output.shape)
        test_ssim[index] = my_ssim(target, output)
        test_psnr[index] = my_psnr(target, output)
        # standard algo:
        recon = fdk(op, data[0].to(device))
        fdk_ssim[index] = my_ssim(target, recon)
        fdk_psnr[index] = my_psnr(target, recon)

    print(f"Testing SSIM: {test_ssim.mean()} +- {test_ssim.std()}")
    print(f"FDK SSIM: {fdk_ssim.mean()} +- {fdk_ssim.std()}")
    print(f"Testing PSNR: {test_psnr.mean()} +- {test_psnr.std()}")
    print(f"FDK PSNR: {fdk_psnr.mean()} +- {fdk_psnr.std()}")

    algo = evaluation['algo_name']
    exp = evaluation['experiment_name']
    model_name = str(algo+"_"+exp+".npy")
    
    np.save(example_folder.joinpath(model_name),output.detach().cpu().numpy().squeeze())
    # TEST 2: Display examples

    import matplotlib.pyplot as plt


    # Display options
    # This is the scale of the error plot w.r.t. the image DISPLAY. Not data, DISPLAY!
    pctg_error = 0.05  # 5%
    max_val = 0.010
    ########################################################
    # Just plot code from now one
    ########################################################

    # BEST PSNR
    target = testing_data[0][1]
    data = testing_data[0][0]
    recon = fdk(op, data)
    output = output

    plt.figure()
    plt.subplot(231)
    plt.imshow(target.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title("Ground truth")
    plt.axis("off")
    plt.clim(0, max_val)
    plt.subplot(232)
    plt.imshow(recon.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title("FDK, PSNR: {:.2f}".format(fdk_psnr[0]))
    plt.axis("off")

    plt.clim(0, max_val)
    plt.subplot(233)
    plt.imshow(output.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title(f"{type(model).__name__}, PSNR: {test_psnr[0]:.2f}")
    plt.clim(0, max_val)
    plt.axis("off")

    plt.subplot(235)
    plt.imshow(
        (recon.detach().cpu().numpy().squeeze() - target.detach().cpu().numpy().squeeze()),
        cmap="seismic",
    )
    plt.title(f"FDK - GT")
    plt.axis("off")
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(-max_val * pctg_error, max_val * pctg_error)
    cbar.set_ticks(
        [-max_val * pctg_error, 0, max_val * pctg_error],
        labels=[f"-{pctg_error*100:.2f}%", "0", f"{pctg_error*100:.2f}%"],
    )
    cbar.ax.tick_params(labelsize=5)

    plt.subplot(236)
    plt.imshow(
        (output.detach().cpu().numpy().squeeze() - target.detach().cpu().numpy().squeeze()),
        cmap="seismic",
    )
    plt.title(f"{type(model).__name__} - GT")
    plt.axis("off")
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(-max_val * pctg_error, max_val * pctg_error)
    cbar.set_ticks(
        [-max_val * pctg_error, 0, max_val * pctg_error],
        labels=[f"-{pctg_error*100:.2f}%", "0", f"{pctg_error*100:.2f}%"],
    )
    cbar.ax.tick_params(labelsize=5)

    plt.suptitle(f"Example index {index_eval}")
    plt.savefig(fig_folder.joinpath(f"eval_{index_eval}_{evaluation['method_name']}_{evaluation['checkpoint']}.png"), dpi=300)
    plt.close()
    
    #np.save(model_name,img_array)
        


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


    # ############# FULL angle
    # evaluations += [
    #     dict(algo_name='TDV', experiment_name='FullData',method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "TDVFullDataCTRecon_check_0008.pt"),
    #     dict(algo_name='AR', experiment_name='FullData',method_name='AR_minval', method = AR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ARFullDataCTRecon_check_0017.pt"),
    #     dict(algo_name='ACR', experiment_name='FullData',method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ACRFullDataCTRecon_check_0025.pt"),
    # ]
    # ############# Limited 120
    # evaluations += [
    #     dict(algo_name='TDV', experiment_name='Limited120',method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "TDVLimitedAngle120CTRecon_check_0004.pt"),
    #     dict(algo_name='AR', experiment_name='Limited120',method_name='AR_minval', method = AR, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "ARLimitedAngle120CTRecon_check_0008.pt"),
    #     dict(algo_name='ACR', experiment_name='Limited120',method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "ACRLimitedAngle120CTRecon_check_0009.pt"),
    # ]
    # ############# Limited 90
    
    # evaluations += [
    #     dict(algo_name='TDV', experiment_name='Limited90',method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "TDVLimitedAngle90CTRecon_check_0011.pt"),
    #     dict(algo_name='AR', experiment_name='Limited90',method_name='AR_minval', method = AR, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "ARLimitedAngle90CTRecon_check_0003.pt"),
    #     dict(algo_name='ACR', experiment_name='Limited90',method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "ACRLimitedAngle90CTRecon_check_0010.pt"),
    # ]

    # ############# Limited 60
    
    # evaluations += [
    #     dict(algo_name='TDV', experiment_name='Limited60',method_name='TDV_minval', method = TDV, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "TDVLimitedAngle60CTRecon_check_0011.pt"),
    #     dict(algo_name='AR', experiment_name='Limited60',method_name='AR_last', method = AR, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "ARLimitedAngle60CTRecon_check_0025.pt"),
    #     dict(algo_name='ACR', experiment_name='Limited60',method_name='ACR_minval', method = ACR, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "ACRLimitedAngle60CTRecon_check_0004.pt"),
    # ]
    
    # ############# Sparse 360
    
    # evaluations += [
    #     dict(algo_name='TDV', experiment_name='Sparse360',method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "TDVSparseAngle360CTRecon_check_0011.pt"),
    #     dict(algo_name='AR', experiment_name='Sparse360',method_name='AR_minval', method = AR, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "ARSparseAngle360CTRecon_check_0009.pt"),
    #     dict(algo_name='ACR', experiment_name='Sparse360',method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "ACRSparseAngle360CTRecon_check_0024.pt"),
    # ]
    
    # ############# Sparse 120

    # evaluations += [
    #     dict(algo_name='TDV', experiment_name='Sparse120',method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "TDVSparseAngle120CTRecon_check_0011.pt"),
    #     dict(algo_name='AR', experiment_name='Sparse120',method_name='AR_minval', method = AR, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "ARSparseAngle120CTRecon_check_0003.pt"),
    #     dict(algo_name='ACR', experiment_name='Sparse120',method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "ACRSparseAngle120CTRecon_check_0023.pt"),
    # ]
    
    # ############# Sparse 60
    
    # evaluations += [
        # dict(algo_name='TDV', experiment_name='Sparse60',method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "TDVSparseAngle60CTRecon_check_0011.pt"),
    #     dict(algo_name='AR', experiment_name='Sparse60',method_name='AR_last', method = AR, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "ARSparseAngle60CTRecon_check_0025.pt"),
    #     dict(algo_name='ACR', experiment_name='Sparse60',method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "ACRSparseAngle60CTRecon_check_0023.pt"),
    # ]
    
    # ############# Beam Hardening
   
    # evaluations += [
        # dict(algo_name='TDV', experiment_name='BeamHardening',method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "TDVBeamHardeningCTRecon_check_0009.pt"),
        # dict(algo_name='AR', experiment_name='BeamHardening',method_name='AR_mintr',  method = AR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ARBeamHardeningCTRecon_check_0023.pt"),
        # dict(algo_name='ACR', experiment_name='BeamHardening',method_name='ACR_last', method = ACR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ACRBeamHardeningCTRecon_check_0025.pt"),
        # dict(algo_name='ACR', experiment_name='BeamHardening',method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ACRBeamHardeningCTRecon_check_0001.pt"),
    # ]

    # ############# Low dose
    
    # evaluations += [
    #     dict(algo_name='TDV', experiment_name='LowDose',method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "TDVLowDoseCTRecon_check_0006.pt"),
    #     dict(algo_name='AR', experiment_name='LowDose',method_name='AR_mintr',  method = AR, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "ARLowDoseCTRecon_check_0003.pt"),
    #     dict(algo_name='ACR', experiment_name='LowDose',method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "ACRLowDoseCTRecon_check_0024.pt"),
    # ]
    
   

    for evaluation in evaluations:
        print(evaluation['method_name'])
        print(evaluation['checkpoint'])
        # eval_experiment(experiment,savefolder)
        eval_experiment(evaluation,index_eval=182)
