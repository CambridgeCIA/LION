# %% [markdown]
# # Plug-and-play ADMM for Photocurrent Mapping image reconstruction
#
# This notebook demonstrates Plug-and-Play (PnP) ADMM image reconstruction for
# Photocurrent Mapping (PCM) data.
#
# Starting from a high-resolution current map of a CIGS solar cell, subsampled
# PCM measurements are simulated using the `PhotocurrentMapOp` operator. Several
# reconstruction methods are then compared:
#
# - Zero-filled pseudo-inverse reconstruction.
# - Two compressed sensing baselines with a wavelet sparsity prior:
#   FISTA with an $\ell_1$ penalty and SPGL1.
# - PnP-ADMM with a pre-trained DRUNet denoiser as prior.
#
# The goal is not to optimise performance, but to illustrate how
# classical sparse reconstruction methods and PnP can be combined with the LION
# operators for PCM.


# %% [markdown]
# ## Setup

# %% [markdown]
# ### Device configuration
#
# Set the default device to a GPU if available. If multiple GPUs are present,
# the desired GPU index can be specified here.

# %%
from __future__ import annotations
from calendar import c
from operator import inv

from cv2 import add, log
import test
import torch
from wandb import init

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# %% [markdown]
# ### Imports
#
# Import the required libraries, including LION operators and reconstruction algorithms for PCM.

# %%
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import deepinv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from jaxtyping import Float
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from functools import partial
from tqdm import tqdm as std_tqdm

from LION.classical_algorithms.fista import fista_l1
from LION.classical_algorithms.spgl1_torch import spgl1_torch
from LION.operators.Wavelet2D import Wavelet2D
from LION.operators.CompositeOp import CompositeOp
from LION.operators.DebiasOp import debias_ls

# LION imports
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.operators.uniform_random_sample import uniform_random_sample
from LION.operators.multilevel_sample import multilevel_sample
from LION.reconstructors.PnP import PnP

# from LION.utils.scale import choose_measurement_scale_factor

from plot_helper import PlotHelper


# Use tqdm with dynamic column width that adapts to the terminal width
tqdm = partial(std_tqdm, dynamic_ncols=True)
tqdm_no_leave = partial(std_tqdm, dynamic_ncols=True, leave=False)

GrayscaleImage2D = Float[torch.Tensor, "height width"]
Measurement1D = Float[torch.Tensor, "num_measurements"]

# %% [markdown]
# ### Define the data file paths
#
# The example uses a single $256 \times 256$ current map of a CIGS solar cell
# stored as a NumPy array. This image will serve as the ground truth in the
# experiments.

# %%
data_dir = Path("data/photocurrent_data")
# data_dir = Path("your/path/to/photocurrent_data")

assert data_dir.exists(), f"Data directory {data_dir} does not exist."

# # These images are provided with pixels in range [0, 1]
# data_name, zoom, loc, loc1, loc2, roi = "CIGS_256x256", 2.5, "center left", 3, 4, (110, 210, 40, 40)  # (x, y, w, h)  with y increasing downwards
# # data_name, zoom, loc, loc1, loc2, roi = "silicon_256x256", 2.5, "lower right", 2, 1, (194, 1, 60, 60)  # (x, y, w, h)  with y increasing downwards
# # data_name, zoom, loc, loc1, loc2, roi = "silicon_512x512", 3, "lower right", 2, 1, (400, 5, 100, 100)  # (x, y, w, h)  with y increasing downwards
# # data_name, zoom, loc, loc1, loc2, roi = "organic_256x256", 2.5, "lower left", 2, 1, (70, 5, 50, 50)  # (x, y, w, h)  with y increasing downwards
# # data_name, zoom, loc, loc1, loc2, roi = "perovskite_256x256", 2.5, "upper left", 3, 4, (90, 190, 50, 50)  # (x, y, w, h)  with y increasing downwards
# data_name = "example_" + data_name  # prefix with "example_"
# is_out_of_distribution = False
# clim = (0.0, 1.0)
# inverses_sign = False
# # R_high, R_low = 1.0, 0.0  # default for normalized images
# is_out_of_distribution = False
# factor = 1  # no scaling for normalized images

# This sample was provided in image form at 512x512 resolution but the pixels are real measured current values
data_name, zoom, loc, loc1, loc2, roi = "Si_256_512x512", 2.5, "lower left", 2, 1, (160, 60, 120, 120)
clim = (0.0, 3e-5)
inverses_sign = True
R_high = 1e-4
R_low = -5e-6
factor = 1e5  # to scale up the photocurrent values for better numerical stability in SPGL1

# # This sample was provided in image form at 512x512 resolution but the pixels are real measured current values
# data_name, zoom, loc, loc1, loc2, roi = "Si_2_256_512x512", 2.5, "lower right", 2, 1, (322, 85, 100, 100)
# clim = (0.0, 1.5e-5)
# inverses_sign = True
# R_high = 2e-5
# R_low = -2e-6
# factor = 1e5  # to scale up the photocurrent values for better numerical stability in SPGL1

if "256x256" in data_name:
    J_order = 8  # J=8 => 2^8=256
elif "512x512" in data_name:
    J_order = 9  # J=9 => 2^9=512
else:
    raise ValueError(f"Unknown data_name {data_name}, cannot determine order_size.")

# # # This is the same data as Si_256_512x512 but provided as measurement data and only up to 256x256 resolution
# data_name, zoom, loc, loc1, loc2, roi = "Si_256_measurement_data", 2, "lower left", 2, 1, (80, 30, 60, 60)
# # data_name, zoom, loc, loc1, loc2, roi = "Si_256_hadamard_measurement_vector", 2, "lower left", 2, 1, (80, 30, 60, 60)
# # data_name, zoom, loc, loc1, loc2, roi = "Si_256_reconstructed_image", 2, "lower left", 2, 1, (80, 30, 60, 60)
# clim = (0.0, 1e-6)
# R_high = 1e-6
# R_low = -1e-6
# factor = 1e7  # to scale up the photocurrent values for better numerical stability in SPGL1
# J_order = 8  # 2^8 x 2^8 = 256 x 256

# # # This is the same data as Si_2_256_512x512 but provided as measurement data and only up to 256x256 resolution
# data_name, zoom, loc, loc1, loc2, roi = "Si_2_256_measurement_data", 2, "lower left", 2, 1, (32, 42, 50, 50)
# # data_name, zoom, loc, loc1, loc2, roi = "Si_2_256_hadamard_measurement_vector", 2, "lower left", 2, 1, (32, 42, 50, 50)
# # data_name, zoom, loc, loc1, loc2, roi = "Si_2_256_reconstructed_image", 2, "lower left", 2, 1, (32, 42, 50, 50)
# clim = (0.0, 4e-7)
# R_high = 1e-6
# R_low = -1e-6
# factor = 1e7  # to scale up the photocurrent values for better numerical stability in SPGL1
# J_order = 8  # 2^8 x 2^8 = 256 x 256

scale_eps = 1e-12
is_out_of_distribution = True
inverses_sign = True

tests_scale_ground_truth = False

data_filename = f"{data_name}.npy"
print("Loading data file:", data_filename)
assert (data_dir / data_filename).exists(), f"Data {data_filename} not found in {data_dir}."

# data_type = "original_measurement_data"
# data_type = "hadamard_measurement_vector"
data_type = "image"
print(f"The type of raw data is: {data_type}")

noise_seed = 42
noise_std = 0  # No noise
# noise_std = 0.05  # standard deviation of additive homoscedastic Gaussian white noise added to measurements

num_trials = 20
num_trials_skip = 6

runs_pnp_admm = True
# pnp_admm_iters = 1
# pnp_admm_iters = 20
pnp_admm_iters = 50
# pnp_admm_iters = 100
# pnp_admm_iters = 150
# pnp_admm_eta = 0.00001  # Undersampling artifacts may remain if eta is too small
# pnp_admm_eta = 0.00005  # Could still work
# pnp_admm_eta = 0.0001  # Could still work
# pnp_admm_eta = 0.001
# pnp_admm_eta = 0.005
pnp_admm_eta = 0.01  # Generally good
# pnp_admm_eta = 0.02
# pnp_admm_eta = 0.03
# pnp_admm_eta = 0.04
# pnp_admm_eta = 0.05
# pnp_admm_eta = 0.1
# pnp_admm_eta = 1
# pnp_admm_eta = 10
# pnp_admm_eta = 20
# pnp_admm_eta = 50
# pnp_admm_eta = 100  # Got nan for 100% sampling?
cg_iters = 20
# cg_iters = 50
cg_eps = 1e-20  # No real change compared to default, CG usually terminates quickly especially when measurements are small
# drunet_sigma = 0.01  # noise level for DRUNet denoiser
# drunet_sigma = 0.02  # noise level for DRUNet denoiser
drunet_sigma = 0.05  # noise level for DRUNet denoiser
# drunet_sigma = 0.1  # noise level for DRUNet denoiser

runs_fista_l1 = False

runs_spgl1 = True

randomizing_scheme = "multilevel"
# randomizing_scheme = "uniform"

cmap_max = 0.8  # take only the lower 0-80% of afmhot, reduce brightness
# cmap_max = 0.9  # take only the lower 0-90% of afmhot to avoid the white top
# cmap_max = 1.0  # take all of afmhot
adds_insets = True
# adds_insets = False
plot_helper = PlotHelper(
    roi=roi,
    zoom=zoom,
    loc=loc,
    show_rect=True,
    cmap=ListedColormap(matplotlib.colormaps['afmhot'](np.linspace(
        0.0,
        cmap_max,
        256,
    ))),
    clim=clim,
    loc1=loc1,
    loc2=loc2
)

# %% [markdown]
# Define a general function to run the photocurrent mapping reconstruction using a reconstruction method.
#
# The helper function `run_pcm_demo`:
#
# - Builds the PCM operator and simulates subsampled measurements.
# - Computes the zero-filled pseudo-inverse reconstruction.
# - Runs a chosen reconstruction method given by `recon_fn`.
# - Reports PSNR and SSIM for both reconstructions, displays and saves the images.


# %% mystnb={"code_prompt_show": "Show utility details"} tags=["hide-cell"]
def show_images_with_inset(
    images: list[torch.Tensor],
    fig_filepath: Path,
    plot_helper: PlotHelper,
    titles: list[str] | None = None,
    suptitle: str | None = None,
    adds_insets: bool = True,
) -> None:
    """Plot images."""
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, squeeze=False, figsize=(n_images * 4, 4))

    for i in range(n_images):
        img_np = images[i].squeeze().cpu().numpy()
        ax: plt.Axes = axes[0][i]
        if adds_insets:
            plot_helper.add_zoom_inset(ax, img_np)
        else:
            ax.imshow(img_np, cmap=plot_helper.cmap, clim=plot_helper.clim)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i], fontsize=10)
    if suptitle:
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.5, 0.02, suptitle, ha="center", va="bottom", fontsize=16)
    fig.savefig(fig_filepath, dpi=150)
    plt.close(fig)


def make_csv(method_name: str, log_dir: Path | str) -> None:
    log_dir = Path(log_dir) / method_name
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "metrics.csv"
    with csv_path.open("w") as f:
        f.write(
            "sampling_percentage,  coarse_J,  "
            "mse_zero_filled,  psnr_zero_filled,  ssim_zero_filled,  pearson_corr_zero_filled,  "
            "mse_recon,  psnr_recon,  ssim_recon,  pearson_corr_recon\n"
        )


# %%
def run_pcm_demo(
    recon_description: str,
    recon_fn: Callable[[PhotocurrentMapOp, Measurement1D, GrayscaleImage2D], GrayscaleImage2D],
    ground_truth_image: GrayscaleImage2D,
    method_name: str,
    image_name: str,
    J: int,  # image size will be 2^J x 2^J
    sampling_ratio: float,
    coarse_J: int,
    measurement_vector: Measurement1D | None = None,
    log_dir: Path | str = ".",
    device: torch.device | str = "cuda:0",
    seed: int = 0,
):
    zero_filled_dir = Path(log_dir) / "zero_filled"
    zero_filled_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = Path(log_dir) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(log_dir) / method_name
    log_dir.mkdir(parents=True, exist_ok=True)
    N = 1 << J
    im_tensor = ground_truth_image.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    sampling_percentage = sampling_ratio * 100
    in_order_measurements_percentage = (1 << (2 * coarse_J)) / (N * N) * 100
    print()
    print(f"Sampling rate: {sampling_percentage}%")
    print(f"Coarse levels to keep: {coarse_J} ({in_order_measurements_percentage}%)")

    rng = np.random.default_rng(seed)
    num_samples = int(sampling_ratio * N * N)
    if randomizing_scheme == "multilevel":
        sampled_indices = multilevel_sample(
            J=J, num_samples=num_samples, coarse_J=coarse_J, alpha=1.0, rng=rng
        )
    elif randomizing_scheme == "uniform":
        sampled_indices = uniform_random_sample(
            J=J, num_samples=num_samples, coarse_J=coarse_J, rng=rng
        )
    else:
        raise ValueError(f"Unknown sampling_scheme {randomizing_scheme}.")
    pcm_op = PhotocurrentMapOp(J=J, sampled_indices=sampled_indices, device=device)

    if measurement_vector is not None:
        print(f"Using provided measurement vector with shape {measurement_vector.shape}.")
        y_subsampled_tensor_noiseless = measurement_vector[sampled_indices]
    else:
        y_subsampled_tensor_noiseless = pcm_op(im_tensor)

    y_subsampled_tensor = y_subsampled_tensor_noiseless  # No noise

    # noise_rng = torch.Generator(device=device)
    # noise_rng.manual_seed(noise_seed)
    # homoscedastic_noise = y_subsampled_tensor_noiseless.normal_(
    #     mean=0.0, std=noise_std, generator=noise_rng
    # )
    # noise = homoscedastic_noise
    # noise = torch.zeros_like(y_subsampled_tensor_noiseless)
    # assert torch.equal(noise, torch.zeros_like(noise)), "Noise is not zero!"

    # y_subsampled_tensor = y_subsampled_tensor_noiseless + noise

    zero_filled_recon_tensor = (
        pcm_op.pseudo_inv(y_subsampled_tensor).unsqueeze(0).unsqueeze(0)
    )

    recon_tensor = (
        recon_fn(
            pcm_op=pcm_op,
            pcm_measurement=y_subsampled_tensor,
            initial_image=zero_filled_recon_tensor.squeeze(),
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    data_range = (im_tensor.max() - im_tensor.min()).item()
    psnr = PeakSignalNoiseRatio(data_range=data_range).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    psnr_zero_filled = psnr(zero_filled_recon_tensor, im_tensor)
    psnr_recon = psnr(recon_tensor, im_tensor)

    ssim_zero_filled = ssim(zero_filled_recon_tensor, im_tensor)
    ssim_recon = ssim(recon_tensor, im_tensor)

    mse_zero_filled = torch.mean((zero_filled_recon_tensor - im_tensor) ** 2).item()
    mse_recon = torch.mean((recon_tensor - im_tensor) ** 2).item()

    pearson_corr_zero_filled = torch.corrcoef(torch.stack([zero_filled_recon_tensor.flatten(), im_tensor.flatten()]))[0,1].item()
    pearson_corr_recon = torch.corrcoef(torch.stack([recon_tensor.flatten(), im_tensor.flatten()]))[0,1].item()

    csv_path = log_dir / "metrics.csv"
    with csv_path.open("a") as f:
        f.write(
            f"{sampling_percentage},  {coarse_J},  "
            f"{mse_zero_filled},  {psnr_zero_filled},  {ssim_zero_filled},  {pearson_corr_zero_filled},  "
            f"{mse_recon},  {psnr_recon},  {ssim_recon},  {pearson_corr_recon}\n"
        )

    filename = f"{image_name}_{method_name}_sample_{sampling_percentage}_percent_coarse_J={coarse_J}_{randomizing_scheme}_random"
    zero_filled_filename = f"{image_name}_sample_{sampling_percentage}_percent_coarse_J={coarse_J}_{randomizing_scheme}_random"
    recons_dir = log_dir / "recons"
    recons_dir.mkdir(parents=True, exist_ok=True)
    np.save(zero_filled_dir / f"{zero_filled_filename}.npy", zero_filled_recon_tensor.squeeze().cpu().numpy())
    np.save(recons_dir / f"{filename}.npy", recon_tensor.squeeze().cpu().numpy())

    mask_of_masks_np = np.zeros(N * N, dtype=bool)
    mask_of_masks_np[sampled_indices] = True
    np.save(masks_dir / f"sample_{sampling_percentage}_percent_coarse_J={coarse_J}_{randomizing_scheme}_random.npy", mask_of_masks_np.reshape(N, N))

    images_dir = log_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    show_images_with_inset(
        [im_tensor, zero_filled_recon_tensor, recon_tensor],
        fig_filepath=images_dir / f"{filename}.png",
        plot_helper=plot_helper,
        titles=[
            "Original Image",
            f"Inverse WHT (Zero-filled)\nPSNR: {psnr_zero_filled:.2f} dB, SSIM: {ssim_zero_filled:.4f}\nMSE: {mse_zero_filled:.3e}, Pearson Corr.: {pearson_corr_zero_filled:.4f}",
            f"{recon_description}\nPSNR: {psnr_recon:.2f} dB, SSIM: {ssim_recon:.4f}\nMSE: {mse_recon:.3e}, Pearson Corr.: {pearson_corr_recon:.4f}",
        ],
        suptitle=(
            f"PCM Reconstructions, J={J} ({N}x{N} image)\n"
            + f"Sample {sampling_percentage:.2f}%, keep {coarse_J} coarse levels ({in_order_measurements_percentage}% here), the rest: {randomizing_scheme} random"
        ),
        adds_insets=adds_insets,
    )

# %%
denoiser_DRUNet = deepinv.models.DRUNet(
    pretrained="download", in_channels=1, out_channels=1, device=device
)

def run_pnp_admm(
    pcm_op: PhotocurrentMapOp, pcm_measurement: Measurement1D, initial_image: GrayscaleImage2D
) -> GrayscaleImage2D:
    admm_iterations = pnp_admm_iters
    admm_eta = pnp_admm_eta
    cg_max_iter = cg_iters
    _cg_eps = cg_eps
    cg_rel_tol = 0.0

    # print(
    #     f"Running PnP-ADMM reconstruction: {admm_iterations} iterations, cg_max_iter={cg_max_iter}..."
    # )

    if is_out_of_distribution:
        a = max(R_high - R_low, scale_eps)

    # pcm_measurement = (pcm_measurement - R_low) / a if is_out_of_distribution else pcm_measurement

    def denoiser_fn(x: GrayscaleImage2D) -> GrayscaleImage2D:
        with torch.no_grad():
            model_input = (x - R_low) / a if is_out_of_distribution else x
            model_output = (
                denoiser_DRUNet(model_input.unsqueeze(0).unsqueeze(0), sigma=drunet_sigma)
                .squeeze(0)
                .squeeze(0)
            )
            model_output = a * model_output + R_low if is_out_of_distribution else model_output
            return model_output

    pnp = PnP(physics=pcm_op, prior_fn=denoiser_fn, default_algorithm="ADMM")
    recon = pnp.admm_algorithm(
        measurement=pcm_measurement,
        eta=admm_eta,
        max_iter=admm_iterations,
        cg_max_iter=cg_max_iter,
        cg_eps=_cg_eps,
        cg_rel_tol=cg_rel_tol,
        prog_bar=tqdm,
        # cg_prog_bar=tqdm,
    )
    # recon = pnp.forward_backward_splitting(
    #     sino=pcm_measurement,
    # )

    return recon
    # return a * recon + R_low if is_out_of_distribution else recon




# %%
def run_fista_l1(
    pcm_op: PhotocurrentMapOp, pcm_measurement: Measurement1D, initial_image: GrayscaleImage2D
) -> GrayscaleImage2D:

    lam = 10  # Good for Daubechies 4 wavelet transform
    # max_iter = 300
    max_iter = 100
    tol = 1e-5

    debias_max_iter = 10
    debias_support_tol = 1e-5
    debias_tol = 1e-7

    height, width = pcm_op.domain_shape
    # Wavelet transform Psi
    wavelet = Wavelet2D((height, width), wavelet_name="db4", device=device)
    # Composite operator A = Phi Psi^{-1}
    A_op = CompositeOp(wavelet, pcm_op, device=device)

    # print("Running FISTA reconstruction: " f"{max_iter} iterations, lambda={lam}...")
    w_hat = fista_l1(
        op=A_op,
        y=pcm_measurement,
        lam=lam,
        max_iter=max_iter,
        tol=tol,
        L=None,
        verbose=False,
        prog_bar=tqdm,
    )

    # Optional debiasing
    # print(f"Running debiasing: {debias_max_iter} iterations...")
    w_debias = debias_ls(
        op=A_op,
        y=pcm_measurement,
        w=w_hat,
        support_tol=debias_support_tol,
        max_iter=debias_max_iter,
        tol=debias_tol,
        prog_bar=tqdm,
    )

    # Current map reconstruction
    cs_result_tensor = wavelet.inverse(w_debias)

    return cs_result_tensor



# %%
def run_spgl1(
    pcm_op: PhotocurrentMapOp, pcm_measurement: Measurement1D, initial_image: GrayscaleImage2D
) -> GrayscaleImage2D:

    # max_iter = 1000
    # max_iter = 200
    max_iter = 100
    # opt_tol = 1e-4
    # bp_tol = 1e-6
    # opt_tol = 1e-5
    # bp_tol = 1e-7

    debias_max_iter = 10
    # debias_max_iter = 100
    debias_support_tol = 1e-5
    # debias_support_tol = 1e-6
    # debias_support_tol = 1e-7
    debias_tol = 1e-7

    height, width = pcm_op.domain_shape
    # Wavelet transform Psi
    wavelet = Wavelet2D((height, width), wavelet_name="db4", device=device)
    # Composite operator A = Phi Psi^{-1}
    A_op = CompositeOp(wavelet, pcm_op, device=device)

    # rhs_l2_norm = torch.linalg.norm(pcm_measurement).item()
    # # relative_feasibility_tolerance = 1e-6
    # relative_feasibility_tolerance = 1e-9
    # absolute_feasibility_tolerance = relative_feasibility_tolerance * rhs_l2_norm

    pcm_measurement = pcm_measurement * factor  # scale up

    # print("Running SPGL1 reconstruction: " f"{max_iter} iterations ...")
    w_hat, _ = spgl1_torch(
        op=A_op,
        y=pcm_measurement,
        iter_lim=max_iter,
        # opt_tol=absolute_feasibility_tolerance,
        # bp_tol=relative_feasibility_tolerance,
        verbosity=0,
        # opt_tol=opt_tol,
        # bp_tol=bp_tol,
    )
    # Optional debiasing
    # print(f"Running debiasing: {debias_max_iter} iterations...")
    w_debias = debias_ls(
        op=A_op,
        y=pcm_measurement,
        w=w_hat,
        support_tol=debias_support_tol,
        max_iter=debias_max_iter,
        tol=debias_tol,
        prog_bar=tqdm,
    )

    # Current map reconstruction
    cs_result_tensor = wavelet.inverse(w_debias)

    cs_result_tensor = cs_result_tensor / factor  # scale back down

    return cs_result_tensor


def make_test_cases() -> list[tuple[float, int]]:
    min_coarse_J = 0
    # min_coarse_J = 5
    # min_coarse_J = J_order - 3  # keep 1.5625% of in-order measurements at least
    # (sampling_ratio, coarse_J)
    test_cases = []
    num_pixels = 1 << (2 * J_order)  # N*N
    # for sampling_ in range(0, 9, 1):
    #     sampling_ratio = (sampling_ + 1) / 100.0  # from 0.01 to 0.09
    #     num_samples = int(sampling_ratio * num_pixels)
    #     for coarse_J in range(min_coarse_J, J_order):  # from 0 to J_order-1 (not including J_order because that is 100% sampling)
    #         if coarse_J > 0:
    #             prev_num_coarse_samples = 1 << (2 * (coarse_J - 1))
    #             if prev_num_coarse_samples >= num_samples:
    #                 continue
    #         test_cases.append((sampling_ratio, coarse_J))
    for sampling_ in range(1, 10, 1):
        sampling_ratio = sampling_ / 10.0  # from 0.1 to 0.9
        test_cases.append((sampling_ratio, J_order - 2))  # keep the first J-2 coarse levels, i.e. 6.25% in-order measurements
        num_samples = int(sampling_ratio * num_pixels)
        # for coarse_J in range(min_coarse_J, J_order):  # from 0 to J_order-1 (not including J_order because that is 100% sampling)
        #     if coarse_J > 0:
        #         prev_num_coarse_samples = 1 << (2 * (coarse_J - 1))
        #         if prev_num_coarse_samples >= num_samples:
        #             continue
        #     test_cases.append((sampling_ratio, coarse_J))
    test_cases.append((1.0, J_order))  # 100% sampling

    # # sampling_ratios = [0.1]
    # sampling_ratios = [0.2]
    # # # sampling_ratios = [0.25]
    # # sampling_ratios = [0.5]
    # # # sampling_ratios = [0.7]
    # # # coarse_Js = [5]  # keep 2^{coarse_J} x 2^{coarse_J} in-order measurements
    # # # coarse_Js = [7]  # keep 2^{coarse_J} x 2^{coarse_J} in-order measurements
    # coarse_Js = list(range(0, J_order))
    # test_cases = []
    # test_cases += [
    #     (sampling_ratio, coarse_J)
    #     for sampling_ratio in sampling_ratios
    #     for coarse_J in coarse_Js
    # ]
    # test_cases = [
    # #     (0.3, 3),
    #     # (0.2, 2),
    #     (0.2, 6),
    #     # (0.2, 8),
    #     # (0.2, 7),
    #     # (0.5, 6),
    #     # (0.8, 6),
    #     # (1.0, 6),
    #     # (1.0, 7),
    #     # (0.1, 8),
    # ]

    # test_cases = test_cases[80:]

    test_cases.reverse()

    return test_cases


def run_experiments():
    raw_data: GrayscaleImage2D | Measurement1D = np.load(data_dir / data_filename)
    print(f"Raw data shape: {raw_data.shape}")
    print(f"J_order: {J_order}")

    if data_type == "image":
        ground_truth_image: GrayscaleImage2D = torch.tensor(raw_data, dtype=torch.float32, device=device)
        if inverses_sign:
            ground_truth_image = -ground_truth_image
        J_data = int(np.log2(ground_truth_image.shape[0]))
        assert J_data == J_order, f"Data J ({J_data}) does not match expected J_order ({J_order})."
        print(f"Ground truth image shape: {ground_truth_image.shape}")
        measurement_vector = None
    elif data_type == "hadamard_measurement_vector":
        # Reconstruct the image from the Hadamard measurement vector
        J_data = int(np.log2(raw_data.shape[0]) / 2)
        assert J_data == J_order, f"Data J ({J_data}) does not match expected J_order ({J_order})."
        measurement_vector = torch.tensor(raw_data, dtype=torch.float32, device=device)
        if inverses_sign:
            measurement_vector = -measurement_vector

        index_of_max = torch.argmax(measurement_vector).item()
        index_of_min = torch.argmin(measurement_vector).item()
        print(f"Max value in measurement vector: {measurement_vector[index_of_max].item()} at index {index_of_max}")
        print(f"Min value in measurement vector: {measurement_vector[index_of_min].item()} at index {index_of_min}")
        exit()

        pcm_op_full = PhotocurrentMapOp(J=J_order, device=device)
        with torch.no_grad():
            ground_truth_image = pcm_op_full.pseudo_inv(measurement_vector)
        print(f"Reconstructed ground truth image shape from Hadamard measurement vector: {ground_truth_image.shape}")

    elif data_type == "original_measurement_data":
        # Make a 1D array with num_lines//2 elements,
        # where each element is the sum of the measured current multiplied by the pattern index sign.
        num_measurements = raw_data.shape[0]
        assert num_measurements % 2 == 0, "Number of measurements should be even."

        if inverses_sign:
            raw_data[:, 1] = -raw_data[:, 1]

        index_of_max_raw = np.argmax(raw_data[:, 1])
        index_of_min_raw = np.argmin(raw_data[:, 1])
        min_raw_value = raw_data[index_of_min_raw, 1]
        max_raw_value = raw_data[index_of_max_raw, 1]
        print(f"Max value in original measurement data: {max_raw_value} at index {index_of_max_raw}")
        print(f"Min value in original measurement data: {min_raw_value} at index {index_of_min_raw}")
        # exit()

        sign_vector = np.round(np.sign(raw_data[:, 0]))
        sign_vector[:2] = [1.0, -1.0]  # Ensure the first two patterns are +0 and -0

        measurement_vector = torch.tensor((raw_data[:, 1] * sign_vector).reshape(-1, 2).sum(axis=1), dtype=torch.float32, device=device)

        # index_of_max = torch.argmax(measurement_vector).item()
        # index_of_min = torch.argmin(measurement_vector).item()
        # min_value = measurement_vector[index_of_min].item()
        # max_value = measurement_vector[index_of_max].item()
        # print(f"Max value in measurement vector: {max_value} at index {index_of_max}")
        # print(f"Min value in measurement vector: {min_value} at index {index_of_min}")
        # exit()

        pcm_op_full = PhotocurrentMapOp(J=J_order, device=device)
        print(f"pcm_op_full domain shape: {pcm_op_full.domain_shape}, range shape: {pcm_op_full.range_shape}")
        with torch.no_grad():
            ground_truth_image = pcm_op_full.pseudo_inv(measurement_vector)
        print(f"Reconstructed ground truth image shape from original measurement data: {ground_truth_image.shape}")

    if tests_scale_ground_truth:
        gt_min, gt_max = ground_truth_image.min().item(), ground_truth_image.max().item()
        ground_truth_image = (ground_truth_image - gt_min) / (gt_max - gt_min)
        print(f"Normalized ground truth image to [0, 1]. Min: {ground_truth_image.min().item()}, Max: {ground_truth_image.max().item()}")
        measurement_vector = None

    test_cases = make_test_cases()
    # print(f"Total number of test cases: {len(test_cases)}")
    # print(test_cases)
    # for sampling_ratio, coarse_J in test_cases:
    #     sampling_percentage = sampling_ratio * 100
    #     coarse_percentage = (1<<(2*coarse_J))/(1<<(2*J_order))*100
    #     print(f"Sampling: {sampling_percentage}%, coarse_J: {coarse_J} ({coarse_percentage}%)")
    #     assert sampling_percentage >= coarse_percentage, (
    #         "Sampling percentage must be larger than or equal to coarse percentage. "
    #         f"Got sampling {sampling_percentage}% and coarse {coarse_percentage}%."
    #     )
    # return

    # ### Set a directory to save logs and results
    #
    # Each run is stored in a separate subdirectory named with the current date and
    # time, which makes it easier to keep track of different experiments.

    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_log_dir = Path("pcm_demo_output") / f"{current_datetime_str}_{data_name}_{randomizing_scheme}_{num_trials}_trials"
    experiment_log_dir.mkdir(parents=True, exist_ok=True)

    for i_seed in tqdm(range(num_trials_skip, num_trials), desc="Running trials"):
        print(f"\n=== Trial {i_seed} ===")
        log_dir = experiment_log_dir / f"trial_{i_seed}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # %% [markdown]
        # ## First experiment: PnP-ADMM
        #
        # In this section the PCM PnP-ADMM algorithm is tested on the CIGS data.

        # %% [markdown]
        # Define the prior function using a pre-trained DRUNet denoiser and the
        # corresponding Plug-and-Play ADMM solver.
        #
        # DRUNet is a deep convolutional denoiser proposed by:
        #
        # > Kai Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc Van Gool, and
        # > Radu Timofte, "Plug-and-Play Image Restoration with Deep Denoiser Prior,"
        # > IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10),
        # > 6360–6376, 2022.
        #
        # In the PnP-ADMM framework, the proximal step of a regulariser is replaced by
        # an off-the-shelf denoiser. Here DRUNet acts as a powerful learned prior for
        # the PCM inverse problem, while the data fidelity term is handled by ADMM.

        # %%
        if runs_pnp_admm:
            method_name = f"pnp_admm_iters={pnp_admm_iters}_eta={pnp_admm_eta}_cg_iters={cg_iters}_drunet_sigma={drunet_sigma}"
            make_csv(method_name=method_name, log_dir=log_dir)
            # for delta_divided_by, subtract_from_J in tqdm(test_cases, desc="Running PnP-ADMM experiments"):
            for sampling_ratio, coarse_J in tqdm(test_cases, desc="Running PnP-ADMM experiments"):
                run_pcm_demo(
                    recon_description=f"PnP-ADMM ({pnp_admm_iters} iters, η={pnp_admm_eta}, cg_iters={cg_iters}, σ={drunet_sigma})",
                    recon_fn=run_pnp_admm,
                    ground_truth_image=ground_truth_image,
                    method_name=method_name,
                    image_name=data_name,
                    J=J_order,  # image size is 2^J x 2^J
                    sampling_ratio=sampling_ratio,
                    coarse_J=coarse_J,
                    log_dir=log_dir,
                    device=device,
                    seed=i_seed,
                )


        # %% [markdown]
        # The code above runs the PnP-ADMM reconstruction and compares it to the
        # zero-filled pseudo-inverse.
        #
        # Although PnP-ADMM substantially improves PSNR and SSIM, it can smooth out
        # fine-scale structures. In the context of defect detection, these small
        # features can be crucial, so high PSNR and SSIM alone are not sufficient to
        # guarantee that the reconstruction is fit for purpose.
        #
        # In the next sections, two compressed sensing baselines with a wavelet
        # sparsity prior are explored and compared to the PnP-ADMM result.


        # %% [markdown]
        # ## Compressed sensing baseline: FISTA with wavelet sparsity
        #
        # This section applies FISTA with an $\ell_1$-penalty on wavelet coefficients
        # as a classical compressed sensing baseline.
        #
        # Let $\Phi$ denote the PCM forward operator and $\Psi$ a 2D wavelet
        # transform with inverse $\Psi^{-1}$. The composite operator
        # $A = \Phi \Psi^{-1}$ acts on wavelet coefficients $w$.
        # FISTA approximately solves the standard $\ell_1$-regularised problem
        #
        # $$
        # \min_w \frac{1}{2} \lVert A w - y \rVert_2^2
        #     + \lambda \lVert w \rVert_1,
        # $$
        #
        # and the final current map is obtained as $x = \Psi^{-1} w$.
        #
        # An optional debiasing step is included at the end to reduce the bias induced
        # by the $\ell_1$ penalty on the active support.

        # %%
        if runs_fista_l1:
            make_csv(method_name="fista_l1", log_dir=log_dir)
            for sampling_ratio, coarse_J in tqdm(test_cases, desc="Running FISTA-L1 experiments"):
                run_pcm_demo(
                    recon_description="FISTA-L1",
                    recon_fn=run_fista_l1,
                    ground_truth_image=ground_truth_image,
                    method_name="fista_l1",
                    image_name=data_name,
                    J=J_order,  # image size is 2^J x 2^J
                    sampling_ratio=sampling_ratio,
                    coarse_J=coarse_J,
                    log_dir=log_dir,
                    device=device,
                    seed=i_seed,
                )

        # %% [markdown]
        # ## Compressed sensing baseline: SPGL1 with wavelet sparsity
        #
        # This section applies the SPGL1 algorithm as a second compressed sensing
        # baseline, again using a wavelet sparsity prior in the same setting
        # $A = \Phi \Psi^{-1}$.
        #
        # SPGL1 is a spectral projected gradient method that efficiently solves
        # large-scale $\ell_1$-regularised problems and basis pursuit denoising
        # formulations. In this example it is run with default parameters suitable
        # for the PCM problem size, followed by the same optional debiasing step
        # used for FISTA.

        # %%
        if runs_spgl1:
            method_name = f"spgl1_factor={factor}"
            make_csv(method_name=method_name, log_dir=log_dir)
            for sampling_ratio, coarse_J in tqdm(test_cases, desc="Running SPGL1 experiments"):
                run_pcm_demo(
                    recon_description="SPGL1",
                    recon_fn=run_spgl1,
                    ground_truth_image=ground_truth_image,
                    method_name=method_name,
                    image_name=data_name,
                    J=J_order,  # image size is 2^J x 2^J
                    sampling_ratio=sampling_ratio,
                    coarse_J=coarse_J,
                    log_dir=log_dir,
                    device=device,
                    seed=i_seed,
                )

if __name__ == "__main__":
    run_experiments()
