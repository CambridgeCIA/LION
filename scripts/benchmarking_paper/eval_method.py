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
def eval_experiment(evaluation):
    savefolder = pathlib.Path("/store/DAMTP/zs334/LION/")
    fig_folder = pathlib.Path("/store/DAMTP/zs334/lion_figs/")
    
    # from LION.models.post_processing.FBPMSDNet import FBPMS_D
    
    # model, options, data = ACR.load(savefolder.joinpath(model_name))
    # model = ACR.load(savefolder.joinpath(model_name))[0]
    model = evaluation['method'].load(savefolder.joinpath(evaluation['checkpoint']))[0]
    model.to(device)##???
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    return 0
    # model.geo = evaluation['experiment'].geo
    # model._make_operator()
    # model.op_norm = power_method(model.op)
    # model.vn.op_norm = model.op_norm
    # model.vn.D = L2DenoiseDataterm(None, model.A, model.AT)

    
    # model.load("/store/DAMTP/zs334/LION/ACR.pt")
    print('Model loaded')
    indices = torch.arange(5)
    testing_data = evaluation['experiment'].get_testing_dataset()
    # testing_data = data_utils.Subset(testing_data, indices)
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

    print(len(testing_dataloader))
    op_norm = power_method(op)
    if ('AR' in evaluation['method_name'] or 'ACR' in evaluation['method_name']):
        print(model.model_parameters.step_size, 0.2/(op_norm)**2)
        if('BeamH' in evaluation['checkpoint']): 
            # model.model_parameters.step_size = 0.2*1e-2/(op_norm)**2
            model.lamb = model.lamb*1e-1



    pbar = tqdm(enumerate(testing_dataloader), total=len(testing_dataloader))
    for index, (data, target) in pbar:
        # model
        # print(index)
        pbar.set_description(f"Testing {index}")
        ### if we have TDV the to dwithout grad
        # print(model.vn.lmbda)
        if 'TDV' in evaluation['method_name']:
            with torch.no_grad(): output = model.output(data.to(device))
        else: output = model.output(data.to(device))
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


    # TEST 2: Display examples

    min_idx = np.argmin(test_psnr)
    max_idx = np.argmax(test_psnr)


    def arg_find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx


    # find the closest to the mean
    mean_idx = arg_find_nearest(test_psnr, test_psnr.mean())
    
    print(f"Best PSNR {max_idx}: {test_psnr[max_idx]}")
    print(f"Mean PSNR {mean_idx}: {test_psnr[mean_idx]}")
    print(f"Worst PSNR {min_idx}: {test_psnr[min_idx]}")

    min_idx_ssim = np.argmin(test_ssim)
    max_idx_ssim = np.argmax(test_ssim)
    mean_idx_ssim = arg_find_nearest(test_ssim, test_ssim.mean())
    
    print(f"Best SSIM {max_idx_ssim}: {test_ssim[max_idx_ssim]}")
    print(f"Mean SSIM {mean_idx_ssim}: {test_ssim[mean_idx_ssim]}")
    print(f"Worst SSIM {min_idx_ssim}: {test_ssim[min_idx_ssim]}")
    
    # Save the results
    np.savez(
        savefolder.joinpath(
            f"results_{evaluation['method_name']}_{evaluation['checkpoint']}.npz"
        ),
        test_ssim=test_ssim,
        test_psnr=test_psnr,
        fdk_ssim=fdk_ssim,
        fdk_psnr=fdk_psnr,
    )


    import matplotlib.pyplot as plt


    # Display options
    # This is the scale of the error plot w.r.t. the image DISPLAY. Not data, DISPLAY!
    pctg_error = 0.05  # 5%
    max_val = 0.010
    ########################################################
    # Just plot code from now one
    ########################################################

    # BEST PSNR
    target = testing_data[max_idx][1]
    data = testing_data[max_idx][0]
    recon = fdk(op, data)
    output = model.output(data.unsqueeze(0).to(device))

    plt.figure()
    plt.subplot(231)
    plt.imshow(target.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title("Ground truth")
    plt.axis("off")
    plt.clim(0, max_val)
    plt.subplot(232)
    plt.imshow(recon.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title("FDK, PSNR: {:.2f}".format(fdk_psnr[max_idx]))
    plt.axis("off")

    plt.clim(0, max_val)
    plt.subplot(233)
    plt.imshow(output.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title(f"{type(model).__name__}, PSNR: {test_psnr[max_idx]:.2f}")
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

    plt.suptitle("Best PSNR")
    plt.savefig(fig_folder.joinpath(f"eval_best_psnr_{evaluation['method_name']}_{evaluation['checkpoint']}.png"), dpi=300)
    
    del target, data, recon, output

    # WORST PSNR

    target = testing_data[min_idx][1]
    data = testing_data[min_idx][0]
    recon = fdk(op, data)
    output = model.output(data.unsqueeze(0).to(device))

    plt.figure()
    plt.subplot(231)
    plt.imshow(target.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title("Ground truth")
    plt.axis("off")
    plt.clim(0, max_val)
    plt.subplot(232)
    plt.imshow(recon.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title("FDK, PSNR: {:.2f}".format(fdk_psnr[min_idx]))
    plt.axis("off")

    plt.clim(0, max_val)
    plt.subplot(233)
    plt.imshow(output.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title(f"{type(model).__name__}, PSNR: {test_psnr[min_idx]:.2f}")
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

    plt.suptitle("Worst PSNR")
    plt.savefig(fig_folder.joinpath(f"eval_worst_psnr_{evaluation['method_name']}_{evaluation['checkpoint']}.png"), dpi=300)
    del target, data, recon, output

    # MEAN PSNR
    target = testing_data[mean_idx][1]
    data = testing_data[mean_idx][0]
    recon = fdk(op, data)
    output = model.output(data.unsqueeze(0).to(device))

    plt.figure()
    plt.subplot(231)
    plt.imshow(target.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title("Ground truth")
    plt.axis("off")
    plt.clim(0, max_val)
    plt.subplot(232)
    plt.imshow(recon.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title("FDK, PSNR: {:.2f}".format(fdk_psnr[mean_idx]))
    plt.axis("off")

    plt.clim(0, max_val)
    plt.subplot(233)
    plt.imshow(output.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title(f"{type(model).__name__}, PSNR: {test_psnr[mean_idx]:.2f}")
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

    plt.suptitle("Mean PSNR")
    plt.savefig(fig_folder.joinpath(f"eval_mean_psnr_{evaluation['method_name']}_{evaluation['checkpoint']}.png"), dpi=300)
        


if __name__ == "__main__":
    #%% 1 - Settingts
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    # Define your data paths
    savefolder = pathlib.Path("/store/DAMTP/zs334/LION/")
    # savefolder = pathlib.Path("/store/DAMTP/ab2860/trained_models/test_debbuging/")

    # list of experiments to run
    # experiments = {
    # # "FullDataCTRecon": ct_benchmarking.FullDataCTRecon(),
    # # "LimitedAngle120CTRecon": ct_benchmarking.LimitedAngle120CTRecon(),
    # # "LimitedAngle90CTRecon": ct_benchmarking.LimitedAngle90CTRecon(),
    # # "LimitedAngle60CTRecon": ct_benchmarking.LimitedAngle60CTRecon(),
    # # "LowDoseCTRecon": ct_benchmarking.LowDoseCTRecon(),
    # "SparseAngle360CTRecon": ct_benchmarking.SparseAngle360CTRecon(),
    # "SparseAngle120CTRecon": ct_benchmarking.SparseAngle120CTRecon(),
    # "SparseAngle60CTRecon": ct_benchmarking.SparseAngle60CTRecon(),
    # "BeamHardeningCTRecon": ct_benchmarking.BeamHardeningCTRecon()
    # }
    evaluations=[]
    
    
    
      ############## TDV evaluations
    
    
    # evaluations = [
        # dict(method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "TDVFullDataCTRecon_check_0008.pt"),
        # dict(method_name='TDV_mintr',  method = TDV, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "TDVFullDataCTRecon_check_0010.pt"),
    # #     dict(method_name='TDV_last', method = TDV, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "TDVFullDataCTRecon_check_0010.pt"),
    # ]
    # evaluations += [
        # dict(method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "TDVLimitedAngle120CTRecon_check_0011.pt"),
        # dict(method_name='TDV_mintr',  method = TDV, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "TDVLimitedAngle120CTRecon_check_0004.pt"),
        # dict(method_name='TDV_last', method = TDV, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "TDVLimitedAngle120CTRecon_check_0011.pt"),
    # ]
    # evaluations += [
        # dict(method_name='TDV_cross', method = TDV, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "TDVFullDataCTRecon_check_0010.pt"),
        #cross-test 
        # dict(method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "TDVLimitedAngle90CTRecon_check_0011.pt"),
    #     dict(method_name='TDV_mintr',  method = TDV, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "TDVLimitedAngle90CTRecon_check_0011.pt"),
    #     dict(method_name='TDV_last', method = TDV, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "TDVLimitedAngle90CTRecon_check_0011.pt"),
    # ]
    # evaluations += [
    #     dict(method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "TDVSparseAngle360CTRecon_check_0011.pt"),
    # #     dict(method_name='TDV_mintr',  method = TDV, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "TDVSparseAngle360CTRecon_check_0011.pt"),
    # #     dict(method_name='TDV_last', method = TDV, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "TDVSparseAngle360CTRecon_check_0011.pt"),
    # ]
    # evaluations += [
    #     dict(method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "TDVSparseAngle120CTRecon_check_0011.pt"),
    # #     dict(method_name='TDV_mintr',  method = TDV, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "TDVSparseAngle120CTRecon_check_0012.pt"),
    # #     dict(method_name='TDV_last', method = TDV, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "TDVSparseAngle120CTRecon_check_0012.pt"),
    # ]
    # evaluations += [
        # dict(method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "TDVSparseAngle60CTRecon_check_0011.pt"),
    # #     dict(method_name='TDV_mintr',  method = TDV, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "TDVSparseAngle60CTRecon_check_000?.pt"),
    # #     dict(method_name='TDV_last', method = TDV, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "TDVSparseAngle60CTRecon_check_000?.pt"),
    # ]
    # evaluations += [
    #     dict(method_name='TDV_minval', method = TDV, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "TDVLimitedAngle60CTRecon_check_0011.pt"),
    # #     dict(method_name='TDV_mintr',  method = TDV, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "TDVLimitedAngle60CTRecon_check_0011.pt"),
    # #     dict(method_name='TDV_last', method = TDV, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "TDVLimitedAngle60CTRecon_check_0011.pt"),
    # ]
    # evaluations += [
        # dict(method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "TDVBeamHardeningCTRecon_check_0009.pt"),
    # #     dict(method_name='TDV_mintr',  method = TDV, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "TDVBeamHardeningCTRecon_check_000?.pt"),
    # #     dict(method_name='TDV_last', method = TDV, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "TDVBeamHardeningCTRecon_check_000?.pt"),
    # ]
    # # ### Add this one
    # evaluations += [
        # dict(method_name='TDV_minval', method = TDV, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "TDVLowDoseCTRecon_check_0006.pt"),
    # #     dict(method_name='TDV_mintr',  method = TDV, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "TDVLowDoseCTRecon_check_0008.pt"),
    # #     dict(method_name='TDV_last', method = TDV, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "TDVLowDoseCTRecon_check_0008.pt"),
    # ]
    
    
    
    
    
    ############## AR evaluations
    
    
    # evaluations += [
    #     # dict(method_name='AR_minval', method = AR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ARFullDataCTRecon_check_0017.pt"),
    #     # dict(method_name='AR_mintr',  method = AR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ARFullDataCTRecon_check_0025.pt"),
    #     dict(method_name='AR_last', method = AR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ARFullDataCTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     # dict(method_name='AR_minval', method = AR, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "ARLimitedAngle120CTRecon_check_0008.pt"),
    #     # dict(method_name='AR_mintr',  method = AR, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "ARLimitedAngle120CTRecon_check_0016.pt"),
    #     dict(method_name='AR_last', method = AR, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "ARLimitedAngle120CTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     dict(method_name='AR_minval', method = AR, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "ARLimitedAngle90CTRecon_check_0003.pt"),
    #     dict(method_name='AR_mintr',  method = AR, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "ARLimitedAngle90CTRecon_check_0025.pt"),
    # # #     dict(method_name='AR_last', method = AR, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "ARLimitedAngle90CTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     # dict(method_name='AR_minval', method = AR, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "ARSparseAngle360CTRecon_check_0009.pt"),
    #     dict(method_name='AR_mintr',  method = AR, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "ARSparseAngle360CTRecon_check_0025.pt"),
    #     # dict(method_name='AR_last', method = AR, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "ARSparseAngle360CTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     dict(method_name='AR_minval', method = AR, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "ARSparseAngle120CTRecon_check_0003.pt"),
    #     dict(method_name='AR_mintr',  method = AR, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "ARSparseAngle120CTRecon_check_0024.pt"),
    # # #     dict(method_name='AR_last', method = AR, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "ARSparseAngle120CTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     # dict(method_name='AR_minval', method = AR, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "ARSparseAngle60CTRecon_check_0001.pt"),
    #     dict(method_name='AR_mintr',  method = AR, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "ARSparseAngle60CTRecon_check_0015.pt"),
    #     dict(method_name='AR_last', method = AR, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "ARSparseAngle60CTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     # dict(method_name='AR_minval', method = AR, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "ARLimitedAngle60CTRecon_check_0004.pt"),
    #     # dict(method_name='AR_mintr',  method = AR, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "ARLimitedAngle60CTRecon_check_0023.pt"),
    #     dict(method_name='AR_last', method = AR, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "ARLimitedAngle60CTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     dict(method_name='AR_minval', method = AR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ARBeamHardeningCTRecon_check_0011.pt"),
    #     dict(method_name='AR_mintr',  method = AR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ARBeamHardeningCTRecon_check_0023.pt"),
    # #     dict(method_name='AR_last', method = AR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ARBeamHardeningCTRecon_check_0025.pt"),
    # ]
    # # ### Add this one
    # evaluations += [
    #     dict(method_name='AR_minval', method = AR, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "ARLowDoseCTRecon_check_0010.pt"),
    #     dict(method_name='AR_mintr',  method = AR, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "ARLowDoseCTRecon_check_0003.pt"),
    #     # dict(method_name='AR_last', method = AR, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "ARLowDoseCTRecon_check_0025.pt"),
    # ]
    
    
   
    
    
    
    
    ############## ACR evaluations
    # evaluations = [
    #     dict(method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ACRFullDataCTRecon_check_0025.pt"),
    #     dict(method_name='ACR_mintr',  method = ACR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ACRFullDataCTRecon_check_0002.pt"),
    #     dict(method_name='ACR_last', method = ACR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ACRFullDataCTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     dict(method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "ACRLimitedAngle120CTRecon_check_0009.pt"),
    #     dict(method_name='ACR_mintr',  method = ACR, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "ACRLimitedAngle120CTRecon_check_0003.pt"),
    #     dict(method_name='ACR_last', method = ACR, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "ACRLimitedAngle120CTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     dict(method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "ACRLimitedAngle90CTRecon_check_0010.pt"),
    #     dict(method_name='ACR_mintr',  method = ACR, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "ACRLimitedAngle90CTRecon_check_0012.pt"),
    #     dict(method_name='ACR_last', method = ACR, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "ACRLimitedAngle90CTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     dict(method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "ACRSparseAngle360CTRecon_check_0024.pt"),
    #     dict(method_name='ACR_mintr',  method = ACR, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "ACRSparseAngle360CTRecon_check_0011.pt"),
    #     dict(method_name='ACR_last', method = ACR, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "ACRSparseAngle360CTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     dict(method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "ACRSparseAngle120CTRecon_check_0023.pt"),
    #     dict(method_name='ACR_mintr',  method = ACR, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "ACRSparseAngle120CTRecon_check_0022.pt"),
    #     dict(method_name='ACR_last', method = ACR, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "ACRSparseAngle120CTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     dict(method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "ACRSparseAngle60CTRecon_check_0023.pt"),
    #     dict(method_name='ACR_mintr',  method = ACR, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "ACRSparseAngle60CTRecon_check_0007.pt"),
    #     dict(method_name='ACR_last', method = ACR, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "ACRSparseAngle60CTRecon_check_0025.pt"),
    # ]
    # evaluations += [
    #     dict(method_name='ACR_minval', method = ACR, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "ACRLimitedAngle60CTRecon_check_0004.pt"),
    #     dict(method_name='ACR_mintr',  method = ACR, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "ACRLimitedAngle60CTRecon_check_0007.pt"),
    #     dict(method_name='ACR_last', method = ACR, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "ACRLimitedAngle60CTRecon_check_0025.pt"),
    # ]
    # evaluations += [
        # dict(method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ACRBeamHardeningCTRecon_check_0001.pt"),
        # dict(method_name='ACR_mintr',  method = ACR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ACRBeamHardeningCTRecon_check_0002.pt"),
        # dict(method_name='ACR_last', method = ACR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ACRBeamHardeningCTRecon_check_0025.pt"),
    # ]
    ### Add this one
    # evaluations += [
    #     dict(method_name='ACR_minval', method = ACR, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "ACRLowDoseCTRecon_check_0024.pt"),
    #     dict(method_name='ACR_mintr',  method = ACR, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "ACRLowDoseCTRecon_check_0014.pt"),
    #     dict(method_name='ACR_last', method = ACR, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "ACRLowDoseCTRecon_check_0025.pt"),
    # ]
    
    
    ###BEST evaluations (the ones in the table)
    
    
    evaluations += [
    #     dict(method_name='TDV_best', method = TDV, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "TDVFullDataCTRecon_check_0008.pt"),
    #     dict(method_name='TDV_best',  method = TDV, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "TDVLimitedAngle120CTRecon_check_0004.pt"),
    #     dict(method_name='TDV_best', method = TDV, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "TDVLimitedAngle90CTRecon_check_0011.pt"),
    #     dict(method_name='TDV_best', method = TDV, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "TDVLimitedAngle60CTRecon_check_0011.pt"),
    #     dict(method_name='TDV_best', method = TDV, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "TDVLowDoseCTRecon_check_0006.pt"),
    #     dict(method_name='TDV_best', method = TDV, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "TDVSparseAngle360CTRecon_check_0011.pt"),
    #     dict(method_name='TDV_best', method = TDV, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "TDVSparseAngle120CTRecon_check_0011.pt"),
        dict(method_name='TDV_best', method = TDV, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "TDVSparseAngle60CTRecon_check_0011.pt"),
    #     dict(method_name='TDV_best', method = TDV, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "TDVBeamHardeningCTRecon_check_0004.pt"),
    ]
    
    
    evaluations += [
        # dict(method_name='AR_best', method = AR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ARFullDataCTRecon_check_0017.pt"),
        # dict(method_name='AR_best', method = AR, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "ARLimitedAngle120CTRecon_check_0008.pt"),
        # dict(method_name='AR_best', method = AR, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "ARLimitedAngle90CTRecon_check_0003.pt"),
        # dict(method_name='AR_best', method = AR, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "ARLimitedAngle60CTRecon_check_0025.pt"),
        # dict(method_name='AR_best', method = AR, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "ARLowDoseCTRecon_check_0003.pt"),
        # dict(method_name='AR_best', method = AR, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "ARSparseAngle360CTRecon_check_0009.pt"),
        # dict(method_name='AR_best', method = AR, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "ARSparseAngle120CTRecon_check_0003.pt"),
        dict(method_name='AR_best', method = AR, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "ARSparseAngle60CTRecon_check_0025.pt"),
        # dict(method_name='AR_best', method = AR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ARBeamHardeningCTRecon_check_0023.pt"),
    ]
    
    evaluations += [
        # dict(method_name='ACR_best', method = ACR, experiment = ct_benchmarking.FullDataCTRecon(), checkpoint = "ACRFullDataCTRecon_check_0025.pt"),
        # dict(method_name='ACR_best', method = ACR, experiment = ct_benchmarking.LimitedAngle120CTRecon(), checkpoint = "ACRLimitedAngle120CTRecon_check_0009.pt"),
        # dict(method_name='ACR_best', method = ACR, experiment = ct_benchmarking.LimitedAngle90CTRecon(), checkpoint = "ACRLimitedAngle90CTRecon_check_0010.pt"),
        # dict(method_name='ACR_best', method = ACR, experiment =  ct_benchmarking.LimitedAngle60CTRecon(), checkpoint = "ACRLimitedAngle60CTRecon_check_0004.pt"),
        # dict(method_name='ACR_best', method = ACR, experiment = ct_benchmarking.LowDoseCTRecon(), checkpoint = "ACRLowDoseCTRecon_check_0024.pt"),
        # dict(method_name='ACR_best', method = ACR, experiment = ct_benchmarking.SparseAngle360CTRecon(), checkpoint = "ACRSparseAngle360CTRecon_check_0024.pt"),
        # dict(method_name='ACR_best', method = ACR, experiment = ct_benchmarking.SparseAngle120CTRecon(), checkpoint = "ACRSparseAngle120CTRecon_check_0023.pt"),
        dict(method_name='ACR_best', method = ACR, experiment = ct_benchmarking.SparseAngle60CTRecon(), checkpoint = "ACRSparseAngle60CTRecon_check_0023.pt"),
        # dict(method_name='ACR_best', method = ACR, experiment = ct_benchmarking.BeamHardeningCTRecon(), checkpoint = "ACRBeamHardeningCTRecon_check_0025.pt"),
    ]
    
    # Device
    device = torch.device("cuda:2")
    torch.cuda.set_device(device)

    for evaluation in evaluations:
        print(evaluation['method_name'])
        print(evaluation['checkpoint'])
        # eval_experiment(experiment,savefolder)
        eval_experiment(evaluation)
