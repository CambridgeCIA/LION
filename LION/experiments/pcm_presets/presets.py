"""Fixed configurations for PCM experiments. New experiments can be added here."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from LION.classical_algorithms.conjugate_gradient import CGProxParameter
from LION.classical_algorithms.spgl1_torch import SPGL1Parameter
from LION.data_loaders.pcm.data_primitives import (
    PCMDataParameter,
    PCMSamplingParameter,
    PCMScalingParameter,
    PCMSingleSampleParameter,
    SamplesParameter,
)
from LION.experiments.pcm_experiments import (
    DEFAULT_RESULTS_DIR,
    PCMExperimentParameter,
    PCMLogParameter,
    PCMSingleSampleRuntimeParameter,
)
from LION.pcm.noise import PCMNoiseSimulation
from LION.pcm.pcm_config import (
    SelfSupervisedPCMMethodParameter,
    SupervisedPCMMethodParameter,
    ZeroShotPCMMethodParameter,
)
from LION.pcm.reconstruct import ScalarTVParameter
from LION.pcm_tools.pcm_prox import PCMProxParameter
from LION.reconstructors.BasePnP import DenoiserParameter
from LION.reconstructors.GSPnP import GSPnPParameter, NormParameter
from LION.reconstructors.PnPADMM import PnPADMMParameter
from LION.utils.paths import (
    PCM_EXAMPLE_DATA_PATH,
    PCM_REAL_DATA_EXAMPLE_PATH,
    PCM_SYNTHETIC_PNG_PATH,
    ROOT,
)
from LION.utils.plot_helper import ColorParameter, PlotHelper, ZoomParameter

# # Preset for the Silicon-2 512x512 image example
# Si_2_256_512x512_IMAGE_PRESET = PCMExperimentParameter(
#     id="si_2_256_512x512_image",
#     data=PCMDataParameter(root=PCM_EXAMPLE_DATA_PATH, data_type="image", j_order=9),
#     this_sample=PCMSingleSampleParameter(
#         data_filename="Si_2_256_512x512.npy", ground_truth_filename="Si_2_256_512x512.npy"
#     ),
#     scaling=PCMScalingParameter(
#         tests_scale_ground_truth=False, is_out_of_distribution=True, r_high=2e-5, r_low=-2e-6, scale_eps=1e-12
#     ),
#     sampling=PCMSamplingParameter(
#         randomising_scheme="uniform", sampling_ratios=[0.2], coarse_j_offsets_from_j_order=[2]
#     ),
#     methods = {
#         'pnp_admm':PnPADMMParameter(
#             denoiser=DenoiserParameter(
#                 # name="gs_drunet",
#                 name="drunet",
#             ),
#             iters=50,
#             eta=0.01,
#             cg_iters=20,
#             cg_eps=1e-20,
#             cg_rel_tol=0.0,
#             drunet_sigma=0.05,
#         ),
#         'spgl1':SPGL1Parameter(
#             factor=1e5, max_iter=100, debias_max_iter=10, debias_support_tol=1e-5, debias_tol=1e-7, wavelet_name="db4"
#         ),
#     }
#     runtime=PCMSingleSampleRuntimeParameter(root_output_dir=DEFAULT_RESULTS_DIR, num_trials=1),
#     plot=PlotHelper(
#         zoom=ZoomParameter(
#             factor=2.5, loc="lower right", loc1=2, loc2=1, roi=(322, 85, 100, 100), adds_insets=True, show_rect=True
#         ),
#         color=ColorParameter(clim=(0.0, 1.5e-5), cmap_max=0.8, inverse_sign=False),
#     ),
# )

DEFAULT_CG = CGProxParameter(
    iters=20,
    eta=0.01,
    eps=1e-20,
    rel_tol=0.0,
    normalize=True,
    # normalize=False,
)

QUICK_METHODS = {
    # # "pnp_admm_drunet_cg": PnPADMMParameter(
    # #     denoiser=DenoiserParameter(name="drunet"), iters=50, prox=DEFAULT_CG, sigma=0.05
    # # ),
    # "pnp_admm_drunet_pcm_prox": PnPADMMParameter(
    "pnp_admm_drunet": PnPADMMParameter(
        denoiser=DenoiserParameter(name="drunet"),
        iters=50,
        # iters=30,
        # prox=PCMProxParameter(rho=0.001),
        prox=PCMProxParameter(rho=0.01),
        # prox=PCMProxParameter(rho=0.1),
        # prox=PCMProxParameter(rho=0.2),
        # prox=DEFAULT_CG,
        # sigma=0.01,
        sigma=0.05,
        # sigma=0.03,
        # sigma=0.08,
        # sigma=0.1,
        # sigma=0.2,
    ),
    # # "pnp_admm_drunet_admm": PnPADMMParameter(
    # #     denoiser=DenoiserParameter(name="drunet_admm", b=0.1),
    # #     iters=50,
    # #     prox=PCMProxParameter(rho=0.01),
    # #     sigma=0.05,
    # # ),
    "pnp_admm_dwtdnet_tiny": PnPADMMParameter(
        denoiser=DenoiserParameter(
            name="dwtdnet",
            size="tiny",
            weights_path=Path(
                ROOT
                + "store/LION/ttv22/pnp/denoiser/20260909_210018_dwt_denoise_df2k_sigma005_dwtdnet_tiny/best.pt"
            ),
        ),
        iters=50,
        # iters=30,
        prox=PCMProxParameter(rho=0.01),
        sigma=0.05,
        # sigma=0.03,
        # # sigma=0.1,
        # sigma=0.2,
    ),
    # # "pnp_admm_dwtdnet_tiny_admm": PnPADMMParameter(
    # #     denoiser=DenoiserParameter(
    # #         name="dwtdnet_admm",
    # #         size="tiny",
    # #         weights_path=Path(
    # #             "/mnt/store_ext4/store/LION/ttv22/pnp/denoiser/20260909_210018_dwt_denoise_df2k_sigma005_dwtdnet_tiny/best.pt"
    # #         ),
    # #         b=0.1,
    # #     ),
    # #     iters=50,
    # #     prox=PCMProxParameter(rho=0.01),
    # #     sigma=0.05,
    # # ),
    # # "pnp_admm_dwtdnet_regular": PnPADMMParameter(
    # #     denoiser=DenoiserParameter(
    # #         name="dwtdnet",
    # #         size="regular",
    # #         weights_path=Path(
    # #             "/mnt/store_ext4/store/LION/ttv22/pnp/denoiser/dwt_denoise_df2k_sigma005_regular_2/best.pt"
    # #         ),
    # #     ),
    # #     iters=50,
    # #     prox=PCMProxParameter(rho=0.01),
    # #     sigma=0.05,
    # # ),
    # # "pnp_admm_dwtdnet_regular_admm": PnPADMMParameter(
    # #     denoiser=DenoiserParameter(
    # #         name="dwtdnet_admm",
    # #         size="regular",
    # #         weights_path=Path(
    # #             "/mnt/store_ext4/store/LION/ttv22/pnp/denoiser/dwt_denoise_df2k_sigma005_regular_2/best.pt"
    # #         ),
    # #         b=0.1,
    # #     ),
    # #     iters=50,
    # #     prox=PCMProxParameter(rho=0.01),
    # #     sigma=0.05,
    # # ),
    # # "pnp_gs_drunet": GSPnPParameter(
    # #     denoiser=DenoiserParameter(name="gs_drunet", weights_path=None),
    # #     iters=50,
    # #     lambd=0.1,  # See GS-PnP paper Figure 11?
    # #     # lambd=0.01,  # See GS-PnP paper Figure 11?
    # #     stepsize=None,  # Default: initial tau_0=1/lambda. Note tau < 1/(lambd*L) (paper said L~=1?)
    # #     eps=1e-6,  # Early stopping, smaller --> may stop earlier. Paper used 1e-6 for super-resolution
    # #     use_backtracking=True,
    # #     gamma_backtracking=0.1,  # gamma in (0, 0.5). Paper used gamma=0.1
    # #     eta_backtracking=0.9,  # tau:=eta*tau each backtracking step. eta in (0, 1). Paper used eta=0.9
    # #     sigma=0.05,
    # #     data_fidelity=NormParameter(
    # #         ord=2,
    # #         # normalize=True,
    # #         normalize=False,
    # #     ),
    # #     # data_fidelity_prox=CGParameter(iters=20, eta=0.01, eps=1e-20, rel_tol=0.0),
    # #     data_fidelity_prox=PCMProxParameter(),
    # #     uses_deepinv=False,
    # # ),
    "pnp_gs_deepinv_drunet": GSPnPParameter(
        denoiser=DenoiserParameter(name="gs_drunet", weights_path=None),
        iters=50,
        # lambd=0.01,  # See GS-PnP paper Figure 11?
        lambd=0.1,  # See GS-PnP paper Figure 11?
        # lambd=1.0,  # See GS-PnP paper Figure 11?
        # lambd=0.5,  # See GS-PnP paper Figure 11?
        stepsize=None,  # Default: initial tau_0=1/lambda. Note tau < 1/(lambd*L) (paper said L~=1?)
        eps=1e-6,  # Early stopping, smaller --> may stop earlier. Paper used 1e-6 for super-resolution
        use_backtracking=True,
        gamma_backtracking=0.1,  # gamma in (0, 0.5). Paper used gamma=0.1
        eta_backtracking=0.9,  # tau:=eta*tau each backtracking step. eta in (0, 1). Paper used eta=0.9
        # sigma=0.01,
        sigma=0.05,
        # sigma=0.03,
        # sigma=0.1,
        # sigma=0.2,
        # sigma=0.08,
        # sigma=0.03,
        data_fidelity=NormParameter(
            ord=2,
            # normalize=True,
            normalize=False,
        ),
        # data_fidelity_prox=CGParameter(iters=20, eta=0.01, eps=1e-20, rel_tol=0.0),
        data_fidelity_prox=PCMProxParameter(),
        uses_deepinv=True,
        # max_backtracking_attempts=100,
        max_backtracking_attempts=20,
    ),
    # # "pnp_admm_gs_dwtdnet_tiny": PnPADMMParameter(
    # #     denoiser=DenoiserParameter(
    # #         name="gs_dwtdnet",
    # #         size="tiny",
    # #         weights_path=Path(
    # #             "/mnt/store_ext4/store/LION/ttv22/pnp/denoiser/20260909_211519_df2k_bsd68_sigma_0_05_dwtdnet_tiny_gs/best.pt"
    # #         ),
    # #     ),
    # #     iters=50,
    # #     # prox=PCMProxParameter(rho=0.01),
    # #     prox=DEFAULT_CG,
    # #     sigma=0.05,
    # # ),
    # # "pnp_admm_gs_drunet_deepinv": PnPADMMParameter(
    # #     denoiser=DenoiserParameter(name="gs_drunet_deepinv", weights_path=None),
    # #     iters=50,
    # #     prox=PCMProxParameter(rho=0.01),
    # #     # prox=DEFAULT_CG,
    # #     sigma=0.05,
    # # ),
    # # "pnp_admm_gs_drunet": PnPADMMParameter(
    # #     denoiser=DenoiserParameter(name="gs_drunet", weights_path=None),
    # #     iters=50,
    # #     prox=PCMProxParameter(rho=0.01),
    # #     sigma=0.05,
    # # ),
    # # "pnp_gs_dwtdnet_tiny": GSPnPParameter(
    # #     denoiser=DenoiserParameter(
    # #         name="gs_dwtdnet",
    # #         size="tiny",
    # #         weights_path=Path(
    # #             "/mnt/store_ext4/store/LION/ttv22/pnp/denoiser/20260909_211519_df2k_bsd68_sigma_0_05_dwtdnet_tiny_gs/best.pt"
    # #         ),
    # #     ),
    # #     iters=50,
    # #     # iters=100,
    # #     lambd=0.01,
    # #     # lambd=0.1,  # See GS-PnP paper Figure 11?
    # #     # lambd=1,
    # #     # lambd=2,
    # #     stepsize=None,  # Default: initial tau_0=1/lambda. Note tau < 1/(lambd*L) (paper said L~=1?)
    # #     eps=1e-6,  # Early stopping, smaller --> may stop earlier. Paper used 1e-6 for super-resolution
    # #     use_backtracking=True,
    # #     gamma_backtracking=0.1,  # gamma in (0, 0.5). Paper used gamma=0.1
    # #     eta_backtracking=0.9,  # tau:=eta*tau each backtracking step. eta in (0, 1). Paper used eta=0.9
    # #     sigma=0.05,
    # #     data_fidelity=NormParameter(
    # #         ord=2,
    # #         # normalize=True,
    # #         normalize=False,
    # #     ),
    # #     # data_fidelity_prox=CGParameter(iters=20, eta=0.01, eps=1e-20, rel_tol=0.0),
    # #     # data_fidelity_prox=PCMProxParameter(),
    # #     data_fidelity_prox=DEFAULT_CG,
    # #     uses_deepinv=False,
    # # ),
    "pnp_gs_deepinv_dwtdnet_tiny": GSPnPParameter(
        denoiser=DenoiserParameter(
            name="gs_dwtdnet",
            size="tiny",
            weights_path=Path(
                ROOT
                + "store/LION/ttv22/pnp/denoiser/20260909_211519_df2k_bsd68_sigma_0_05_dwtdnet_tiny_gs/best.pt"
            ),
        ),
        iters=50,
        # iters=100,
        # lambd=0.01,
        lambd=0.1,  # See GS-PnP paper Figure 11?
        # lambd=1,
        # lambd=2,
        stepsize=None,  # Default: initial tau_0=1/lambda. Note tau < 1/(lambd*L) (paper said L~=1?)
        eps=1e-6,  # Early stopping, smaller --> may stop earlier. Paper used 1e-6 for super-resolution
        use_backtracking=True,
        gamma_backtracking=0.1,  # gamma in (0, 0.5). Paper used gamma=0.1
        eta_backtracking=0.9,  # tau:=eta*tau each backtracking step. eta in (0, 1). Paper used eta=0.9
        sigma=0.05,
        # sigma=0.03,
        # sigma=0.1,
        # sigma=0.2,
        data_fidelity=NormParameter(
            ord=2,
            # normalize=True,
            normalize=False,
        ),
        # data_fidelity_prox=CGParameter(iters=20, eta=0.01, eps=1e-20, rel_tol=0.0),
        # data_fidelity_prox=PCMProxParameter(),
        data_fidelity_prox=DEFAULT_CG,
        uses_deepinv=True,
        # max_backtracking_attempts=100,
        max_backtracking_attempts=20,
    ),
    "supervised_utv": SupervisedPCMMethodParameter(
        run_dir=DEFAULT_RESULTS_DIR / "pcm" / "pcm_supervised" / "20260920_021902_TV",
        checkpoint="best_val_loss_s0.20_cJ6.pt",
        scale_factor_case=(0.2, 6),
    ),
    "self_supervised_utv": SelfSupervisedPCMMethodParameter(
        run_dir=DEFAULT_RESULTS_DIR
        / "pcm"
        / "pcm_self_supervised"
        / "20260920_165908_TV_self_supervised",
        checkpoint="best_val_loss_s0.20_cJ6.pt",
        scale_factor_case=(0.2, 6),
    ),
    "spgl1": SPGL1Parameter(
        factor=1e5,
        iters=100,
        debias_iters=10,
        debias_support_tol=1e-5,
        debias_tol=1e-7,
        wavelet_name="db4",
    ),
    "scalar_tv": ScalarTVParameter(
        # lambda_reg=0.001,
        lambda_reg=0.02,
        iters=1024,
        metric_every=1024,
    ),
}

GRID_METHODS = {
    **QUICK_METHODS,
    "zero_shot_utv": ZeroShotPCMMethodParameter(
        config_path=Path("config/pcm/pcm_tv_zero_shot.yaml"),
        criterion="best_val_ssdu_loss",
    ),
}

BOX_PLOT_METHODS = {
    **QUICK_METHODS,
    "zero_shot_utv": ZeroShotPCMMethodParameter(
        config_path=Path("config/pcm/pcm_tv_zero_shot.yaml"),
        criterion="best_val_ssdu_loss",
    ),
}

# Preset for the CIGS example 256x256 image example
CIGS_example_256x256_IMAGE_PRESET = PCMExperimentParameter(
    id="cigs_example_256x256",
    uses_normalized_op=True,
    # base_seed=0,
    base_seed=42,
    noise=PCMNoiseSimulation.npl_default(
        current_to_model_scale=1.0 / 3.5e-8, j_order=8
    ),  # Canonical noise model from NPL values
    # noise=PCMNoiseSimulation.simulation_default(scale=1.0 / 3.5e-2),  # Simulate noise!
    data=PCMDataParameter(root=PCM_EXAMPLE_DATA_PATH, data_type="image", j_order=8),
    this_sample=PCMSingleSampleParameter(
        data_filename="example_CIGS_256x256.npy",
        ground_truth_filename="example_CIGS_256x256.npy",
    ),
    scaling=PCMScalingParameter(
        tests_scale_ground_truth=False, is_out_of_distribution=False
    ),
    sampling=PCMSamplingParameter(
        # sampling_ratios=[0.1, 0.2, 0.5, 0.8, 1.0],
        sampling_ratios=[0.1, 0.2, 0.5],
        # sampling_ratios=[0.01],
        # randomising_scheme="uniform",
        randomising_scheme="multilevel",
        coarse_j_values=[6],
        # coarse_j_offsets_from_j_order=[2],
    ),
    # methods={
    #     method_id: QUICK_METHODS[method_id]
    #     for method_id in [
    #         # "pnp_gs_deepinv_dwtdnet_tiny",
    #         "supervised_utv",
    #         "self_supervised_utv",
    #     ]
    # },
    methods=QUICK_METHODS,
    runtime=PCMSingleSampleRuntimeParameter(
        root_output_dir=DEFAULT_RESULTS_DIR, num_trials=1
    ),
    log=PCMLogParameter(
        save_plots=True, save_recon=True, save_diff=True, save_zero_filled=True
    ),
    plot=PlotHelper(
        zoom=ZoomParameter(
            factor=2.5,
            loc="center left",
            loc1=3,
            loc2=4,
            roi=(110, 210, 40, 40),
            adds_insets=True,
            show_rect=True,
        ),
        color=ColorParameter(clim=(0.0, 1.0), cmap_max=0.8, inverse_sign=False),
    ),
)


# Preset for the E27 position-1 real measurement with the average of 4 maps as ground truth
E27_POS1_REAL_MEASUREMENT_PRESET = PCMExperimentParameter(
    id="e27_pos1_real_measurement",
    uses_normalized_op=True,
    base_seed=42,
    noise=None,
    data=PCMDataParameter(
        root=PCM_REAL_DATA_EXAMPLE_PATH / "E27" / "pos1",
        data_type="original_measurement_data_txt",
        j_order=10,
    ),
    this_sample=PCMSingleSampleParameter(
        data_filename="E27-full-map-1-default-settings.txt",
        ground_truth_filename="E27-full-map-default-settings-average-of-4.txt.map",
    ),
    scaling=PCMScalingParameter(
        tests_scale_ground_truth=False,
        is_out_of_distribution=True,
        r_high=3.5e-8,
        r_low=0.0,
        divide_by=64.0,
        data_sign=-1.0,
        fill_value=0.0,
    ),
    sampling=PCMSamplingParameter(
        # sampling_ratios=[0.05],
        sampling_ratios=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # Use all measurements in order. These are real measurements so
        # there is temporal dependency, cannot randomise here yet.
        # coarse_j_values=[10],
        coarse_j_values=[0],
        # randomising_scheme="multilevel",
        randomising_scheme=None,
    ),
    # methods={
    #     method_id: QUICK_METHODS[method_id]
    #     for method_id in [
    #         # "pnp_gs_deepinv_dwtdnet_tiny",
    #         "supervised_utv",
    #         "self_supervised_utv",
    #     ]
    # },
    methods=QUICK_METHODS,
    runtime=PCMSingleSampleRuntimeParameter(
        root_output_dir=DEFAULT_RESULTS_DIR, num_trials=1
    ),
    log=PCMLogParameter(
        save_plots=True, save_recon=True, save_diff=True, save_zero_filled=True
    ),
    plot=PlotHelper(
        # color=ColorParameter(clim=(0.0, 4e-8), cmap_max=1.0, inverse_sign=True),
        color=ColorParameter(clim=(0.0, 3.5e-8), cmap_max=1.0, inverse_sign=False),
        dpi=100,
    ),
)


# Preset for the E27 position-1 average map with simulated NPL noise
E27_POS1_AVERAGE_MAP_SIMULATED_NOISE_PRESET = PCMExperimentParameter(
    id="e27_pos1_average_map_simulated_noise",
    uses_normalized_op=True,
    base_seed=42,
    noise=PCMNoiseSimulation.npl_default(
        current_to_model_scale=1.0 / (3.5e-8),
        j_order=10,
        n_repeat_acquisitions=1,
    ),
    data=PCMDataParameter(
        root=PCM_REAL_DATA_EXAMPLE_PATH / "E27" / "pos1",
        data_type="image_txt_map",
        j_order=10,
    ),
    this_sample=PCMSingleSampleParameter(
        data_filename="E27-full-map-default-settings-average-of-4.txt.map",
        ground_truth_filename="E27-full-map-default-settings-average-of-4.txt.map",
    ),
    scaling=PCMScalingParameter(
        tests_scale_ground_truth=False,
        is_out_of_distribution=True,
        r_high=3.5e-8,
        r_low=0.0,
        divide_by=64.0,
        data_sign=-1.0,
        fill_value=0.0,
    ),
    sampling=PCMSamplingParameter(
        # sampling_ratios=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # sampling_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        sampling_ratios=[0.2],
        # sampling_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        # sampling_ratios=[0.7, 0.8, 0.9],
        # sampling_ratios=[1.0],
        # coarse_j_values=None,
        # coarse_j_values=[0, 2, 4, 6, 8, 10],
        # coarse_j_values=[4, 6, 8],
        coarse_j_values=[8],
        # coarse_j_values=[10],
        randomising_scheme="multilevel",
    ),
    methods=QUICK_METHODS,
    # methods={
    #     # method_id: QUICK_METHODS[method_id]
    #     method_id: GRID_METHODS[method_id]
    #     for method_id in [
    #         # "pnp_gs_deepinv_dwtdnet_tiny",
    #         "supervised_utv",
    #         "self_supervised_utv",
    #         # "zero_shot_utv",
    #     ]
    # },
    runtime=PCMSingleSampleRuntimeParameter(
        root_output_dir=DEFAULT_RESULTS_DIR, num_trials=1
    ),
    log=PCMLogParameter(
        save_plots=True, save_recon=True, save_diff=True, save_zero_filled=True
    ),
    plot=PlotHelper(
        # color=ColorParameter(clim=(0.0, 4e-8), cmap_max=1.0, inverse_sign=True),
        color=ColorParameter(clim=(0.0, 3.5e-8), cmap_max=1.0, inverse_sign=False),
        dpi=100,
    ),
)


# Preset for the synthetic PNG dataset grid search over many sampling ratios and coarse J values
# Use all categories, few samples per cat (only 10), only 1 trial per sample
SYNTHETIC_PNG_DATASET_PRESET_GRID = PCMExperimentParameter(
    id="synthetic_png_dataset_grid",
    uses_normalized_op=True,  # Should use normalized operator
    # base_seed=0,
    base_seed=42,
    noise=PCMNoiseSimulation.npl_default(
        current_to_model_scale=1.0 / 3.5e-8, j_order=8, n_repeat_acquisitions=1
    ),
    data=PCMDataParameter(
        root=PCM_SYNTHETIC_PNG_PATH,
        categories=[
            "CIGS",
            "Organic",
            "Perovskite",
            "Reference",
            "Silicone",
        ],  # 1. Five categories of PCM images
        samples=SamplesParameter(
            # id_range=[1, 2],
            # id_range=[1, 1000],
            total_samples=1000,
            # perm_range=[901, 1000],
            perm_range=[991, 1000],
            perm=5678,
        ),
        data_type="image",
        j_order=8,  # 256x256 images, j_order=8 (2^8=256)
    ),
    scaling=PCMScalingParameter(
        tests_scale_ground_truth=False, is_out_of_distribution=False
    ),
    sampling=PCMSamplingParameter(
        # randomising_scheme="uniform",
        randomising_scheme="multilevel",
        sampling_ratios=[  # 2. Sampling ratios
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        coarse_j_values=None,  # 3. Try all coarse J values
        # coarse_j_offsets_from_j_order=[2],  # 4. Coarse J offset from j_order=8, so coarse_j=6
    ),
    # methods=TESTED_METHODS,
    # methods=GRID_METHODS,
    methods={method_id: GRID_METHODS[method_id] for method_id in ["zero_shot_utv"]},
    runtime=PCMSingleSampleRuntimeParameter(
        root_output_dir=DEFAULT_RESULTS_DIR,
        num_trials=1,
        # num_trials=30,  # 4. Bootstrap
    ),
    log=PCMLogParameter(
        save_plots=True, save_recon=True, save_diff=True, save_zero_filled=True
    ),
    plot=PlotHelper(
        color=ColorParameter(clim=(0.0, 1.0), cmap_max=0.8, inverse_sign=False), dpi=100
    ),
)


# Preset for the synthetic PNG dataset many samples (100 per category --> 500 in total)
# Use all 5 categories, 1 sampling ratio (20%), 1 coarse J (coarse_j = 6), only 1 trial per sample
SYNTHETIC_PNG_DATASET_PRESET_BOX_PLOTS = PCMExperimentParameter(
    id="synthetic_png_dataset_box_plots",
    uses_normalized_op=True,  # Should use normalized operator
    # base_seed=0,
    base_seed=42,
    noise=PCMNoiseSimulation.npl_default(
        current_to_model_scale=1.0 / 3.5e-8, j_order=8
    ),
    data=PCMDataParameter(
        root=PCM_SYNTHETIC_PNG_PATH,
        categories=[
            "CIGS",
            "Organic",
            "Perovskite",
            "Reference",
            "Silicone",
        ],  # 1. Five categories of PCM images
        samples=SamplesParameter(
            # id_range=[1, 2],
            # id_range=[1, 1000],
            total_samples=1000,
            # perm_range=[901, 1000],
            perm_range=[991, 1000],
            perm=5678,
        ),
        data_type="image",
        j_order=8,  # 256x256 images, j_order=8 (2^8=256)
    ),
    scaling=PCMScalingParameter(
        tests_scale_ground_truth=False, is_out_of_distribution=False
    ),
    sampling=PCMSamplingParameter(
        # randomising_scheme="uniform",
        randomising_scheme="multilevel",
        sampling_ratios=[  # 2. Sampling ratios
            0.2,
            # 0.5,
            # 0.8,
            # 0.999,
        ],
        coarse_j_values=[6],
        # coarse_j_values=[7],
        # coarse_j_values=[8],
        # coarse_j_offsets_from_j_order=[2],  # 4. Coarse J offset from j_order=8, so coarse_j=6
    ),
    methods=QUICK_METHODS,
    # methods=BOX_PLOT_METHODS,
    # methods={
    # method_id: QUICK_METHODS[method_id]
    # method_id: BOX_PLOT_METHODS[method_id]
    # for method_id in [
    #     # "pnp_gs_deepinv_dwtdnet_tiny",
    #     "supervised_utv",
    #     "self_supervised_utv",
    #     # "zero_shot_utv",
    # ]
    # },
    runtime=PCMSingleSampleRuntimeParameter(
        root_output_dir=DEFAULT_RESULTS_DIR,
        num_trials=1,
        # num_trials=30,  # 4. Bootstrap
    ),
    log=PCMLogParameter(
        save_plots=True, save_recon=True, save_diff=True, save_zero_filled=True
    ),
    plot=PlotHelper(
        color=ColorParameter(clim=(0.0, 1.0), cmap_max=0.8, inverse_sign=False), dpi=100
    ),
)


# # Add scalar TV without changing the existing selection of box-plot methods.
# SYNTHETIC_PNG_DATASET_PRESET_BOX_PLOTS.methods["scalar_tv"] = deepcopy(BOX_PLOT_METHODS["scalar_tv"])

# # Run scalar TV alone against the same sample IDs, sampling masks and PCM noise.
# # This avoids re-running the other methods when extending earlier box plots.
# SYNTHETIC_PNG_DATASET_PRESET_BOX_PLOTS_SCALAR_TV = deepcopy(SYNTHETIC_PNG_DATASET_PRESET_BOX_PLOTS)
# SYNTHETIC_PNG_DATASET_PRESET_BOX_PLOTS_SCALAR_TV.id = "synthetic_png_dataset_box_plots_scalar_tv"
# SYNTHETIC_PNG_DATASET_PRESET_BOX_PLOTS_SCALAR_TV.methods = {
#     "scalar_tv": deepcopy(BOX_PLOT_METHODS["scalar_tv"]),
# }
# SYNTHETIC_PNG_DATASET_PRESET_BOX_PLOTS_SCALAR_TV.log = PCMLogParameter(
#     save_metrics=True,
#     save_plots=False,
#     save_recon=False,
#     save_diff=False,
#     save_zero_filled=False,
# )


# Preset for the synthetic PNG dataset many trials (30)
# Use all categories, 10 samples per cat, 3 sampling ratios, 1 coarse J (coarse_j = 6)
SYNTHETIC_PNG_DATASET_PRESET_30_trials = PCMExperimentParameter(
    id="synthetic_png_dataset_30_trials",
    uses_normalized_op=True,  # Should use normalized operator
    # base_seed=0,
    base_seed=42,
    noise=PCMNoiseSimulation.npl_default(
        current_to_model_scale=1.0 / 3.5e-8, j_order=8
    ),  # Canonical noise model from NPL values
    data=PCMDataParameter(
        root=PCM_SYNTHETIC_PNG_PATH,
        categories=[
            "CIGS",
            "Organic",
            "Perovskite",
            "Reference",
            "Silicone",
        ],  # 1. Five categories of PCM images
        # categories=["CIGS"],
        samples=SamplesParameter(
            # id_range=[1, 2],
            # id_range=[1, 1000],
            total_samples=1000,
            perm_range=[991, 1000],
            perm=5678,
        ),
        data_type="image",
        j_order=8,  # 256x256 images, j_order=8 (2^8=256)
    ),
    scaling=PCMScalingParameter(
        tests_scale_ground_truth=False, is_out_of_distribution=False
    ),
    sampling=PCMSamplingParameter(
        # randomising_scheme="uniform",
        randomising_scheme="multilevel",
        sampling_ratios=[  # 2. Sampling ratios
            0.2,
            0.5,
            0.8,
        ],
        coarse_j_values=[6],
        # coarse_j_offsets_from_j_order=[2],  # 4. Coarse J offset from j_order=8, so coarse_j=6
    ),
    methods=QUICK_METHODS,
    runtime=PCMSingleSampleRuntimeParameter(
        root_output_dir=DEFAULT_RESULTS_DIR,
        # num_trials=1,
        num_trials=30,  # 4. Bootstrap
    ),
    log=PCMLogParameter(
        save_plots=True, save_recon=True, save_diff=True, save_zero_filled=True
    ),
    plot=PlotHelper(
        color=ColorParameter(clim=(0.0, 1.0), cmap_max=0.8, inverse_sign=False), dpi=100
    ),
)


PRESET_LIST = [
    # Si_2_256_512x512_IMAGE_PRESET,
    CIGS_example_256x256_IMAGE_PRESET,
    E27_POS1_REAL_MEASUREMENT_PRESET,
    E27_POS1_AVERAGE_MAP_SIMULATED_NOISE_PRESET,
    SYNTHETIC_PNG_DATASET_PRESET_GRID,
    SYNTHETIC_PNG_DATASET_PRESET_BOX_PLOTS,
    SYNTHETIC_PNG_DATASET_PRESET_30_trials,
]


PRESETS = {preset.id: preset for preset in PRESET_LIST}


# _CURRENT_PRESET = Si_2_256_512x512_IMAGE_PRESET

# cigs_example_256x256_scale_data_config = replace(
#     _CURRENT_PRESET.sample,
#     data_filename="example_CIGS_256x256.npy",
#     ground_truth_filename="example_CIGS_256x256.npy",
#     data_type="image",
#     j_order=8,
#     is_out_of_distribution=True,
#     r_high=4e-8,
#     r_low=0,
#     requires_scaling=True,
#     fill_value=0.0,  # Sign is reversed so fill with max 0.0 instead
# )

# cigs_example_256x256_plot_config = replace(
#     _CURRENT_PRESET.plot,
#     zoom=2.5,
#     loc="center left",
#     loc1=3,
#     loc2=4,
#     roi=(110, 210, 40, 40),
#     # clim=(0.0, 1e-8),
#     # inverse_sign=True,
#     # clim=(-4e-8, 0.0),
#     clim=(0.0, 4e-8),
#     inverse_sign=False,
# )


# PRESETS: dict[str, PCMExperimentParameter] = {
#     "si_2_256_512x512_image": _CURRENT_PRESET,
#     "si_256_512x512_image": replace(
#         _CURRENT_PRESET,
#         name="si_256_512x512_image",
#         data=replace(
#             _CURRENT_PRESET.sample,
#             data_filename="Si_256_512x512.npy",
#             ground_truth_filename="Si_256_512x512.npy",
#             j_order=9,
#             r_high=1e-4,
#             r_low=-5e-6,
#         ),
#         plot=replace(
#             _CURRENT_PRESET.plot,
#             roi=(160, 60, 120, 120),
#             loc="lower left",
#             clim=(0.0, 3e-5),
#         ),
#     ),
#     "si_256_measurement_data": replace(
#         _CURRENT_PRESET,
#         name="si_256_measurement_data",
#         data=replace(
#             _CURRENT_PRESET.sample,
#             data_filename="Si_256_measurement_data.npy",
#             data_type="original_measurement_data",
#             ground_truth_filename=None,
#             j_order=8,
#             r_high=1e-6,
#             r_low=-1e-6,
#         ),
#         plot=replace(
#             _CURRENT_PRESET.plot,
#             zoom=2.0,
#             loc="lower left",
#             roi=(80, 30, 60, 60),
#             clim=(0.0, 1e-6),
#         ),
#         spgl1=replace(_CURRENT_PRESET.spgl1, factor=1e7),
#     ),
#     "si_2_256_measurement_data": replace(
#         _CURRENT_PRESET,
#         name="si_2_256_measurement_data",
#         data=replace(
#             _CURRENT_PRESET.sample,
#             data_filename="Si_2_256_measurement_data.npy",
#             data_type="original_measurement_data",
#             ground_truth_filename=None,
#             j_order=8,
#             r_high=1e-6,
#             r_low=-1e-6,
#         ),
#         plot=replace(
#             _CURRENT_PRESET.plot,
#             zoom=2.0,
#             loc="lower left",
#             roi=(32, 42, 50, 50),
#             clim=(0.0, 4e-7),
#         ),
#         spgl1=replace(_CURRENT_PRESET.spgl1, factor=1e7),
#     ),
#     "cigs_example_256x256": replace(
#         _CURRENT_PRESET,
#         name="cigs_example_256x256",
#         data=replace(
#             _CURRENT_PRESET.sample,
#             data_filename="example_CIGS_256x256.npy",
#             ground_truth_filename="example_CIGS_256x256.npy",
#             data_type="image",
#             j_order=8,
#             is_out_of_distribution=False,
#             r_high=None,
#             r_low=None,
#         ),
#         plot=replace(
#             _CURRENT_PRESET.plot,
#             zoom=2.5,
#             loc="center left",
#             loc1=3,
#             loc2=4,
#             roi=(110, 210, 40, 40),
#             clim=(0.0, 1.0),
#         ),
#         # spgl1=replace(_CURRENT_PRESET.spgl1, factor=1.0),
#         spgl1=replace(
#             _CURRENT_PRESET.spgl1,
#             # max_iter=100,
#             # debias_max_iter=100,
#             # debias_support_tol=1e-5 / 256,
#             # debias_tol=1e-7 / 256,
#             # wavelet_name="db4",
#             # factor=1.0 * 256 * 256,
#             factor=1.0,
#         ),
#     ),
#     "cigs_example_256x256_scale_no_noise": replace(
#         _CURRENT_PRESET,
#         name="cigs_example_256x256_scale_no_noise",
#         data=cigs_example_256x256_scale_data_config,
#         sampling=replace(_CURRENT_PRESET.sampling, sampling_ratios=(0.2,), coarse_j_values=(6,)),
#         pnp_admm=replace(
#             _CURRENT_PRESET.pnp_admm,
#             denoiser_name="gs_drunet",
#             cg_iters=50,
#             drunet_sigma=0.05,
#             eta=0.01,
#             iters=50,
#             uses_normalized_op=False,
#         ),
#         tv=TvPdhgTorchParameter(
#             device=None,
#             num_iters=128,
#             sigma=None,
#             tau=None,
#             theta=None,
#             lambda_v=0.002,
#             uses_normalized_op=True,
#         ),
#         plot=cigs_example_256x256_plot_config,
#     ),
#     "cigs_example_256x256_scale_noise_default": replace(
#         _CURRENT_PRESET,
#         name="cigs_example_256x256_scale_noise_default",
#         data=cigs_example_256x256_scale_data_config,
#         pcm_noise=PCMNoiseSimulation.simulation_default(),
#         sampling=replace(_CURRENT_PRESET.sampling, sampling_ratios=(0.2,), coarse_j_values=(6,)),
#         pnp_admm=replace(
#             _CURRENT_PRESET.pnp_admm,
#             denoiser_name="gs_drunet",
#             cg_iters=30,
#             drunet_sigma=0.1,
#             eta=0.5,
#             iters=100,
#             uses_normalized_op=False,
#         ),
#         tv=TvPdhgTorchParameter(num_iters=128, sigma=None, tau=None, theta=None, lambda_v=0.01),
#         spgl1=SPGL1Parameter(
#             factor=2e7,
#             # max_iter=100,
#             # debias_max_iter=100,
#             # debias_support_tol=1e-8,
#             # debias_tol=1e-10,
#             # debias_support_tol=1e-3,
#             # debias_tol=1e-5,
#         ),
#         plot=cigs_example_256x256_plot_config,
#     ),
#     "cigs_example_256x256_scale_noise_default_no_dark": replace(
#         _CURRENT_PRESET,
#         name="cigs_example_256x256_scale_noise_default_no_dark",
#         data=cigs_example_256x256_scale_data_config,
#         pcm_noise=PCMNoiseSimulation.simulation_default_no_dark(),
#         sampling=replace(_CURRENT_PRESET.sampling, sampling_ratios=(0.2,), coarse_j_values=(6,)),
#         pnp_admm=replace(
#             _CURRENT_PRESET.pnp_admm,
#             denoiser_name="gs_drunet",
#             cg_iters=30,
#             drunet_sigma=0.05,
#             eta=0.01,
#             iters=100,
#             uses_normalized_op=False,
#         ),
#         tv=TvPdhgTorchParameter(num_iters=128, sigma=None, tau=None, theta=None, lambda_v=0.002),
#         spgl1=SPGL1Parameter(
#             factor=1e5,
#             # max_iter=100,
#             # debias_max_iter=100,
#             # debias_support_tol=1e-8,
#             # debias_tol=1e-10,
#             # debias_support_tol=1e-3,
#             # debias_tol=1e-5,
#         ),
#         plot=cigs_example_256x256_plot_config,
#     ),
#     "silicon_1_image_768x1024_no_noise": replace(
#         _CURRENT_PRESET,
#         name="silicon_1_image_768x1024_no_noise",
#         data=replace(
#             _CURRENT_PRESET.sample,
#             data_filename="E27-full-map-default-settings-average-of-4.txt.map",
#             data_type="image_txt_map",
#             ground_truth_filename="E27-full-map-default-settings-average-of-4.txt.map",
#             data_dir=Path("example_data/E27/pos1"),
#             j_order=10,
#             r_high=0,
#             r_low=-4e-08,
#             divide_by=64.0,
#             fill_value=0.0,  # Sign is reversed so fill with max 0.0 instead
#         ),
#         sampling=replace(
#             _CURRENT_PRESET.sampling, sampling_ratios=(0.05,), coarse_j_values=(8,), randomising_scheme="multilevel"
#         ),
#         plot=replace(
#             _CURRENT_PRESET.plot,
#             zoom=6.0,
#             loc="lower right",
#             roi=(430, 260, 80, 80),
#             clim=(0.0, 4e-08),
#             cmap_max=1.0,
#             inverse_sign=True,
#             axis_off=False,
#         ),
#         pnp_admm=PnpAdmmParameter(
#             denoiser_name="gs_drunet",
#             iters=30,
#             eta=0.01,
#             cg_iters=20,
#             drunet_sigma=0.05,
#             uses_normalized_op=False,
#         ),
#         tv=TvPdhgTorchParameter(
#             num_iters=128,
#             sigma=None,
#             tau=None,
#             theta=None,
#             lambda_v=0.001,
#         ),
#         spgl1=SPGL1Parameter(factor=1e5),
#     ),
#     "silicon_1_image_768x1024_noise_default_10ms_100mA": replace(
#         _CURRENT_PRESET,
#         name="silicon_1_image_768x1024_noise_default_10ms_100mA",
#         data=replace(
#             _CURRENT_PRESET.sample,
#             data_filename="E27-full-map-default-settings-average-of-4.txt.map",
#             data_type="image_txt_map",
#             ground_truth_filename="E27-full-map-default-settings-average-of-4.txt.map",
#             data_dir=Path("example_data/E27/pos1"),
#             j_order=10,
#             r_high=0,
#             r_low=-4e-08,
#             divide_by=64.0,
#             fill_value=0.0,  # Sign is reversed so fill with max 0.0 instead
#         ),
#         pcm_noise=PCMNoiseSimulation.simulation_default(),
#         sampling=replace(
#             _CURRENT_PRESET.sampling,
#             sampling_ratios=(0.05,),
#             coarse_j_values=(8,),
#             randomising_scheme="multilevel",
#         ),
#         plot=replace(
#             _CURRENT_PRESET.plot,
#             zoom=6.0,
#             loc="lower right",
#             roi=(430, 260, 80, 80),
#             clim=(0.0, 4e-08),
#             cmap_max=1.0,
#             inverse_sign=True,
#             axis_off=False,
#         ),
#         pnp_admm=PnpAdmmParameter(
#             denoiser_name="gs_drunet",
#             iters=30,
#             eta=0.01,
#             cg_iters=20,
#             drunet_sigma=0.05,
#             uses_normalized_op=False,
#         ),
#         # precision_builder=PrecisionBuilder(
#         #     sigma_eta_rel=2.9e-5,
#         #     sigma_g_rel=1.6e-7,
#         #     sigma_beam_rel=0.012 / 100.0,
#         #     sigma_dark_abs=PCMNoiseSimulation.simulation_default().noise_params.dark_sigma_amps,
#         #     time_grid=None,
#         #     differential_factor=2.0,
#         #     variance_floor_rel=1e-6,
#         #     jitter=1e-20,
#         # ),
#         tv=TvPdhgTorchParameter(
#             num_iters=128,
#             sigma=None,
#             tau=None,
#             theta=None,
#             lambda_v=0.001,
#         ),
#         spgl1=SPGL1Parameter(
#             factor=1e5,
#             # max_iter=100,
#             # debias_max_iter=100,
#             # debias_support_tol=1e-8,
#             # debias_tol=1e-10,
#             # debias_support_tol=1e-3,
#             # debias_tol=1e-5,
#         ),
#     ),
#     "silicon_1_measurement_768x1024_real_noise": replace(
#         _CURRENT_PRESET,
#         name="silicon_1_measurement_768x1024_real_noise",
#         data=replace(
#             _CURRENT_PRESET.sample,
#             data_filename="E27-full-map-1-default-settings.txt",
#             data_type="original_measurement_data_txt",
#             ground_truth_filename="E27-full-map-default-settings-average-of-4.txt.map",
#             data_dir=Path("example_data/E27/pos1"),
#             j_order=10,
#             r_high=0,
#             r_low=-4e-08,
#             divide_by=64.0,
#             fill_value=0.0,  # Sign is reversed so fill with max 0.0 instead
#         ),
#         sampling=replace(
#             _CURRENT_PRESET.sampling,
#             sampling_ratios=(0.05,),
#             # Use all measurements in order. These are real measurements so
#             # there is temporal dependency, cannot randomise here yet.
#             coarse_j_values=(8,),
#             randomising_scheme="multilevel",
#         ),
#         plot=replace(
#             _CURRENT_PRESET.plot,
#             zoom=6.0,
#             loc="lower right",
#             roi=(430, 260, 80, 80),
#             clim=(0.0, 4e-08),
#             cmap_max=1.0,
#             inverse_sign=True,
#             axis_off=False,
#         ),
#         pnp_admm=replace(
#             _CURRENT_PRESET.pnp_admm,
#             denoiser_name="gs_drunet",
#             iters=50,
#             drunet_sigma=0.05,
#             eta=0.01,
#             cg_iters=20,
#             uses_normalized_op=False,
#         ),
#         tv=TvPdhgTorchParameter(
#             num_iters=128,
#             sigma=None,
#             tau=None,
#             theta=None,
#             lambda_v=0.01,
#         ),
#         spgl1=SPGL1Parameter(
#             factor=1e5,
#             # max_iter=100,
#             # debias_max_iter=100,
#             # debias_support_tol=1e-8,
#             # debias_tol=1e-10,
#             # debias_support_tol=1e-3,
#             # debias_tol=1e-5,
#         ),
#     ),
#     "silicon_1b_Ap100us_measurement_768x1024": replace(
#         _CURRENT_PRESET,
#         name="silicon_1b_Ap100us_measurement_768x1024",
#         data=replace(
#             _CURRENT_PRESET.sample,
#             data_filename="E27-full-map-1b-Ap100us.txt",
#             data_type="original_measurement_data_txt",
#             ground_truth_filename="E27-full-map-1b-default-settings.txt.map",
#             data_dir=Path("example_data/E27/pos1b"),
#             j_order=10,
#             r_high=0,
#             r_low=-4e-08,
#             divide_by=64.0,
#             fill_value=0.0,  # Sign is reversed so fill with max 0.0 instead
#         ),
#         sampling=replace(
#             _CURRENT_PRESET.sampling,
#             sampling_ratios=(0.1,),
#             # Use all measurements in order. These are real measurements so
#             # there is temporal dependency, cannot randomise here yet.
#             coarse_j_values=(8,),
#             randomising_scheme="multilevel",
#         ),
#         plot=replace(
#             _CURRENT_PRESET.plot,
#             zoom=6.0,
#             loc="lower right",
#             roi=(360, 260, 80, 80),
#             clim=(0.0, 4e-08),
#             cmap_max=1.0,
#             inverse_sign=True,
#         ),
#         pnp_admm=replace(
#             _CURRENT_PRESET.pnp_admm,
#             denoiser_name="gs_drunet",
#             cg_iters=30,
#             drunet_sigma=0.05,
#             lagged_iters=5,
#             lagged_damping=0.5,
#             # data_weight=0.5,
#             data_weight=1.0,
#         ),
#         precision_builder=PrecisionBuilder(
#             sigma_eta_rel=2.9e-5,
#             sigma_g_rel=1.6e-7,
#             # sigma_beam_rel=0.012 / 100.0,
#             sigma_beam_rel=10 * (0.012 / 100.0),
#             sigma_dark_abs=PCMNoiseSimulation.simulation_extrapolate().noise_params.dark_sigma_amps,
#             time_grid=None,
#             differential_factor=2.0,
#             variance_floor_rel=1e-6,
#             jitter=1e-20,
#         ),
#     ),
# }
