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

from pprint import pprint
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# %% [markdown]
# ### Imports
#
# Import the required libraries, including LION operators and reconstruction algorithms for PCM.

# %%
from datetime import datetime
from pathlib import Path
from typing import Callable

import deepinv
import matplotlib.pyplot as plt
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
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp, Subsampler
from LION.reconstructors.PnP import PnP


# Use tqdm with dynamic column width that adapts to the terminal width
tqdm = partial(std_tqdm, dynamic_ncols=True)

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

data_name = "CIGS"
# data_name = "silion"
# data_name = "organic"
# data_name = "perovskite"
data_filename = f"example_{data_name}_256x256.npy"

assert (data_dir / data_filename).exists(), f"Data file not found in {data_dir}."


runs_pnp_admm = False
runs_fista_l1 = False
runs_spgl1 = True


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
def show_images(
    images: list[torch.Tensor],
    fig_filepath: Path,
    titles: list[str] | None = None,
    suptitle: str | None = None,
) -> None:
    """Plot images."""
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, squeeze=False, figsize=(n_images * 4, 4))
    for i in range(n_images):
        axes[0][i].imshow(images[i].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0][i].axis("off")
        if titles:
            axes[0][i].set_title(titles[i], fontsize=10)
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
            "sampling_percentage,  in_order_measurements_percentage,  "
            "mse_zero_filled,  psnr_zero_filled,  ssim_zero_filled,  pearson_corr_zero_filled,  "
            "mse_recon,  psnr_recon,  ssim_recon,  pearson_corr_recon\n"
        )

# %%
def run_pcm_demo(
    recon_method_name: str,
    recon_fn: Callable[[PhotocurrentMapOp, Measurement1D], GrayscaleImage2D],
    ground_truth_image: GrayscaleImage2D,
    method_name: str,
    image_name: str,
    J: int,  # image size will be 2^J x 2^J
    # subtract_from_J: int = 1,
    sampling_ratio: float,
    in_order_ratio: float,
    # delta_divided_by: int = 4,
    log_dir: Path | str = ".",
    device: torch.device | str = "cuda:0",
):
    zero_filled_dir = Path(log_dir) / "zero_filled"
    zero_filled_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) / method_name
    log_dir.mkdir(parents=True, exist_ok=True)
    N = 1 << J
    im_tensor = torch.tensor(ground_truth_image, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # coarseJ = J - subtract_from_J
    # delta = 1.0 / delta_divided_by

    # sampling_percentage = delta * 100
    sampling_percentage = sampling_ratio * 100
    # in_order_measurements_percentage = 1 / (1 << (subtract_from_J * 2)) * 100
    in_order_measurements_percentage = in_order_ratio * sampling_percentage
    print(f"Sampling rate:         {sampling_percentage:.2f}%")
    print(f"In-order measurements: {in_order_measurements_percentage:.2f}%")

    subsampler = Subsampler(n=N * N, sampling_ratio=sampling_ratio, in_order_ratio=in_order_ratio, rng=np.random.default_rng(0))
    pcm_op = PhotocurrentMapOp(J=J, subsampler=subsampler, device=device)

    y_subsampled_tensor = pcm_op(im_tensor)
    zero_filled_recon_tensor = (
        pcm_op.pseudo_inv(y_subsampled_tensor).unsqueeze(0).unsqueeze(0)
    )

    recon_tensor = (
        recon_fn(
            pcm_op=pcm_op,
            pcm_measurement=y_subsampled_tensor,
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
            f"{sampling_percentage},  {in_order_measurements_percentage},  "
            f"{mse_zero_filled},  {psnr_zero_filled},  {ssim_zero_filled},  {pearson_corr_zero_filled},  "
            f"{mse_recon},  {psnr_recon},  {ssim_recon},  {pearson_corr_recon}\n"
        )

    filename = f"{image_name}_{recon_method_name}_sampling_percentage={sampling_percentage:.2f}_in_order_measurements={in_order_measurements_percentage:.2f}"
    zero_filled_filename = f"{image_name}_sampling_percentage={sampling_percentage:.2f}_in_order_measurements={in_order_measurements_percentage:.2f}"
    recons_dir = log_dir / "recons"
    recons_dir.mkdir(parents=True, exist_ok=True)
    np.save(zero_filled_dir / f"{zero_filled_filename}.npy", zero_filled_recon_tensor.squeeze().cpu().numpy())
    np.save(recons_dir / f"{filename}.npy", recon_tensor.squeeze().cpu().numpy())

    images_dir = log_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    show_images(
        [im_tensor, zero_filled_recon_tensor, recon_tensor],
        fig_filepath=images_dir / f"{filename}.png",
        titles=[
            "Original Image",
            f"Zero-filled Reconstruction\nPSNR: {psnr_zero_filled:.2f} dB, SSIM: {ssim_zero_filled:.4f}\nMSE: {mse_zero_filled:.4e}, Pearson Corr.: {pearson_corr_zero_filled:.4f}",
            f"{recon_method_name} Reconstruction\nPSNR: {psnr_recon:.2f} dB, SSIM: {ssim_recon:.4f}\nMSE: {mse_recon:.4e}, Pearson Corr.: {pearson_corr_recon:.4f}",
        ],
        suptitle=(
            "PCM Reconstructions Comparison\n"
            + f"sampling rate: {sampling_percentage:.2f}%, in-order measurements: {in_order_measurements_percentage:.2f}%"
        ),
    )

# %%
denoiser_DRUNet = deepinv.models.DRUNet(
    pretrained="download", in_channels=1, out_channels=1, device=device
)


def denoiser_fn(x: GrayscaleImage2D) -> GrayscaleImage2D:
    with torch.no_grad():
        return (
            denoiser_DRUNet(x.unsqueeze(0).unsqueeze(0), sigma=0.05)
            .squeeze(0)
            .squeeze(0)
        )


def run_pnp_admm(
    pcm_op: PhotocurrentMapOp, pcm_measurement: Measurement1D
) -> GrayscaleImage2D:
    # admm_iterations = 100
    admm_iterations = 20
    admm_step_size = 1e5
    # cg_max_iter = 100
    cg_max_iter = 20
    cg_tol = 1e-7

    print(
        f"Running PnP-ADMM reconstruction: {admm_iterations} iterations, cg_max_iter={cg_max_iter}..."
    )

    pnp = PnP(physics=pcm_op, prior_fn=denoiser_fn, algorithm="ADMM")
    return pnp.admm_algorithm(
        measurement=pcm_measurement,
        eta=admm_step_size,
        max_iter=admm_iterations,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
        prog_bar=tqdm,
    )




# %%
def run_fista_l1(
    pcm_op: PhotocurrentMapOp, pcm_measurement: Measurement1D
) -> GrayscaleImage2D:

    lam = 10  # Good for Daubechies 4 wavelet transform
    # max_iter = 300
    max_iter = 100
    tol = 1e-5

    debias_max_iter = 10  # TODO: Debiasing seems to not make much difference here
    debias_support_tol = 1e-5
    debias_tol = 1e-7

    height, width = pcm_op.domain_shape
    # Wavelet transform Psi
    wavelet = Wavelet2D((height, width), wavelet_name="db4", device=device)
    # Composite operator A = Phi Psi^{-1}
    A_op = CompositeOp(wavelet, pcm_op, device=device)

    print("Running FISTA reconstruction: " f"{max_iter} iterations, lambda={lam}...")
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
    print(f"Running debiasing: {debias_max_iter} iterations...")
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
    pcm_op: PhotocurrentMapOp, pcm_measurement: Measurement1D
) -> GrayscaleImage2D:

    lam: float = 1e-3
    # max_iter = 1000
    max_iter = 100
    tol: float = 1e-4
    debias_max_iter = 10
    debias_support_tol = 1e-5
    debias_tol = 1e-7

    height, width = pcm_op.domain_shape
    # Wavelet transform Psi
    wavelet = Wavelet2D((height, width), wavelet_name="db4", device=device)
    # Composite operator A = Phi Psi^{-1}
    A_op = CompositeOp(wavelet, pcm_op, device=device)

    print("Running SPGL1 reconstruction: " f"{max_iter} iterations, lambda={lam}...")
    w_hat = spgl1_torch(
        op=A_op,
        y=pcm_measurement,
        iter_lim=max_iter,
        verbosity=0,
        opt_tol=tol,
    )
    # Optional debiasing
    print(f"Running debiasing: {debias_max_iter} iterations...")
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


def make_test_cases() -> list[tuple[float, float]]:
    # (sampling_ratio, in_order_ratio)
    test_cases = []
    for sampling_ in range(0, 9, 1):
        for in_order_ in range(0, sampling_ + 2, 1):
            sampling_ratio = (sampling_ + 1) / 100.0  # from 0.01 to 0.09
            in_order_ratio_abs = (in_order_) / 100.0  # from 0.00 to sampling_ratio
            in_order_ratio_rel = in_order_ratio_abs / sampling_ratio  # relative to sampling ratio
            # print(f"sampling_: {sampling_}, in_order_: {in_order_} => sampling_ratio: {sampling_ratio}, in_order_ratio_abs: {in_order_ratio_abs}, in_order_ratio_rel: {in_order_ratio_rel}")
            assert 0 < sampling_ratio <= 1.0, f"sampling_ratio must be in (0, 1], got {sampling_ratio}"
            assert 0 <= in_order_ratio_abs <= sampling_ratio, f"in_order_ratio_abs must be in [0, sampling_ratio={sampling_ratio}], got {in_order_ratio_abs}"
            assert 0 <= in_order_ratio_rel <= 1.0, f"in_order_ratio_rel must be in [0, 1], got {in_order_ratio_rel}"
            test_cases.append((sampling_ratio, in_order_ratio_rel))
    for sampling_ in range(0, 10, 1):
        for in_order_ in range(0, sampling_ + 2, 1):
            sampling_ratio = (sampling_ + 1) / 10.0  # from 0.1 to 1.0
            in_order_ratio_abs = (in_order_) / 10.0  # from 0.0 to sampling_ratio
            in_order_ratio_rel = in_order_ratio_abs / sampling_ratio  # relative to sampling ratio
            # print(f"sampling_: {sampling_}, in_order_: {in_order_} => sampling_ratio: {sampling_ratio}, in_order_ratio_abs: {in_order_ratio_abs}, in_order_ratio_rel: {in_order_ratio_rel}")
            assert 0 < sampling_ratio <= 1.0, f"sampling_ratio must be in (0, 1], got {sampling_ratio}"
            assert 0 <= in_order_ratio_abs <= sampling_ratio, f"in_order_ratio_abs must be in [0, sampling_ratio={sampling_ratio}], got {in_order_ratio_abs}"
            assert 0 <= in_order_ratio_rel <= 1.0, f"in_order_ratio_rel must be in [0, 1], got {in_order_ratio_rel}"
            test_cases.append((sampling_ratio, in_order_ratio_rel))

    # test_cases = [(0.5, 0.5)]  # 50% sampling, half of which are in-order

    # test_cases = [
    #     # (32, 3),  # 3.125% sampling, 1.5625% in-order (half in-order)
    #     # (32, 2),  # 3.125% sampling, 6.25% in-order (all in-order)
    #     # (16, 3),  # 6.25% sampling, 1.5625% in-order (1/4 of in-order)
    #     # (16, 2),  # 6.25% sampling, 6.25% in-order (all in-order)
    #     # (8, 3),  # 12.5% sampling, 1.5625% in-order (1/8 in-order)
    #     # (8, 2),  # 12.5% sampling, 6.25% in-order (half in-order)
    #     # (8, 1),  # 12.5% sampling, 25% in-order (all in-order)
    #     # (4, 3),  # 25% sampling, 1.5625% in-order (1/16 in-order)
    #     # (4, 2),  # 25% sampling, 6.25% in-order (1/4 in-order)
    #     # (4, 1),  # 25% sampling, 25% in-order (all in-order)
    #     # (2, 2),  # 50% sampling, 6.25% in-order (1/8 in-order)
    #     # (2, 1),  # 50% sampling, 25% in-order (half in-order)
    #     # (2, 0.5),  # 50% sampling, half of which are in-order
    #     (0.5, 0.5),  # 50% sampling, half of which are in-order
    #     # (2, 0),  # 50% sampling, 100% in-order (all in-order)
    #     # (1, 0),  # 100% sampling, 100% in-order (all in-order)
    # ]
    return test_cases


def run_experiments():


    # %% [markdown]
    # ## Experiment

    # %% [markdown]
    # Load the data.
    #
    # The array `cigs_raw_data` is a single 2D image representing the current map
    # of the CIGS device. This will be used as the ground truth image for all
    # reconstruction methods.

    # %%
    raw_data: GrayscaleImage2D = np.load(data_dir / data_filename)
    print(f"Raw data shape: {raw_data.shape}")

    # %% [markdown]
    # In the examples below:
    #
    # - `J = 8` so the image size is $2^J \times 2^J = 256 \times 256$.
    # - `delta_divided_by = 4` corresponds to a sampling rate
    #   $\delta = 1 / 4 = 0.25$, that is, 25% of the total measurements.
    # - `subtract_from_J = 2` keeps a central $2^{J-2} \times 2^{J-2} = 64 \times 64$
    #   block of in-order measurements, corresponding to 6.25% of the total;
    #   the remaining samples are taken in a compressed sensing fashion.

    # %%
    # J = 8  # image size is 2^J x 2^J = 256x256
    # delta_divided_by = 4  # 25% sampling
    # subtract_from_J = (
    #     2  # keep 2^{J-2} x 2^{J-2} = 64x64 in-order measurements, or 6.25% of the total
    # )

    test_cases = make_test_cases()
    # pprint(test_cases)
    # return

    # ### Set a directory to save logs and results
    #
    # Each run is stored in a separate subdirectory named with the current date and
    # time, which makes it easier to keep track of different experiments.

    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("pcm_demo_output") / f"{current_datetime_str}_{data_name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # %% [markdown]
    # ## First experiment: PnP-ADMM on CIGS data
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
        make_csv(method_name="pnp_admm", log_dir=log_dir)
        # for delta_divided_by, subtract_from_J in tqdm(test_cases, desc="Running PnP-ADMM experiments"):
        for sampling_ratio, in_order_ratio in tqdm(test_cases, desc="Running PnP-ADMM experiments"):
            run_pcm_demo(
                recon_method_name="PnP-ADMM",
                recon_fn=run_pnp_admm,
                ground_truth_image=raw_data,
                method_name="pnp_admm",
                image_name="cigs",
                J=8,  # image size is 2^J x 2^J = 256x256
                sampling_ratio=sampling_ratio,
                in_order_ratio=in_order_ratio,
                log_dir=log_dir,
                device=device,
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
        for sampling_ratio, in_order_ratio in tqdm(test_cases, desc="Running FISTA-L1 experiments"):
            run_pcm_demo(
                recon_method_name="FISTA-L1",
                recon_fn=run_fista_l1,
                ground_truth_image=raw_data,
                method_name="fista_l1",
                image_name="cigs",
                J=8,  # image size is 2^J x 2^J = 256x256
                sampling_ratio=sampling_ratio,
                in_order_ratio=in_order_ratio,
                log_dir=log_dir,
                device=device,
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
        make_csv(method_name="spgl1", log_dir=log_dir)
        for sampling_ratio, in_order_ratio in tqdm(test_cases, desc="Running SPGL1 experiments"):
            run_pcm_demo(
                recon_method_name="SPGL1",
                recon_fn=run_spgl1,
                ground_truth_image=raw_data,
                method_name="spgl1",
                image_name="cigs",
                J=8,  # image size is 2^J x 2^J = 256x256
                sampling_ratio=sampling_ratio,
                in_order_ratio=in_order_ratio,
                log_dir=log_dir,
                device=device,
            )

if __name__ == "__main__":
    run_experiments()
