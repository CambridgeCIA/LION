# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: lion_proposed
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Walsh Hadamard Transform (WHT) demo


# %% [markdown]
# ## Imports
#

# %%
import deepinv
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from spyrit.core.torch import fwht_2d
import torch

# Lion imports
from LION.reconstructors.PnPReconstructor import PnPReconstructor
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp, Subsampler


def run_demo(
    dataset: torch.utils.data.Dataset,
    subtract_from_J: int = 1,
    delta_divided_by: int = 4,
):
    device = torch.get_default_device()
    # %%
    J = 9  # 512x512 images
    N = 1 << J

    sino, target = dataset[0]
    im_tensor = target.unsqueeze(0)  # (1,1,H,W)

    coarseJ = J - subtract_from_J
    delta = 1.0 / delta_divided_by

    sampling_rate_percent = delta * 100
    in_order_measurements_percent = 1 / (1 << (subtract_from_J * 2)) * 100

    subsampler = Subsampler(n=N * N, coarseJ=coarseJ, delta=delta)
    op = PhotocurrentMapOp(J=J, subsampler=subsampler)
    y_subsampled_tensor = op(im_tensor)
    im_reconstructed_tensor = op.adjoint(y_subsampled_tensor)

    # %%

    print(f"Running fast WHT")
    measurement_2d = fwht_2d(im_tensor, order=False)
    log_measurement_2d = torch.log1p(torch.abs(measurement_2d))

    def reorder2d_index(Y_std: np.ndarray, perm: np.ndarray) -> np.ndarray:
        N = Y_std.shape[0]
        assert Y_std.shape == (N, N)
        assert perm.shape == (N * N,)
        # perm maps: reordered_index -> standard_flat_index
        src_r = (perm // N).reshape(N, N)  # row indices in standard Y
        src_c = (perm % N).reshape(N, N)  # col indices in standard Y
        return Y_std[src_r, src_c]  # (N,N) reordered view

    measurement_2d_reordered = reorder2d_index(
        measurement_2d.squeeze().detach().cpu().numpy(), op.normal_to_dyadic_perm
    )
    log_measurement_2d_reordered = np.log1p(np.abs(measurement_2d_reordered))

    # plt.imshow(log_measurement_2d_reordered, cmap="magma")
    # # plt.colorbar()
    # plt.axis("off")
    # plt.title("Measurements")
    # plt.show()

    # plt.imshow(op.mask_reordered.astype(np.float32), cmap="gray")
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(
    #     f"Mask of patterns used\nCoarse J=J-{subtract_from_J}, Delta=1/{delta_divided_by}"
    # )
    # plt.show()

    # # plt.figure(figsize=(36, 6))
    # plt.figure(figsize=(36, 9))
    # # plt.figure(figsize=(24, 6))

    # plt.subplot(1, 4, 1)
    # plt.imshow(log_measurement_2d.squeeze().detach().cpu().numpy(), cmap="magma")
    # # plt.colorbar()
    # # plt.axis("off")
    # plt.title("Standard FWHT Measurement (Log Magnitude)", fontsize=16, y=-0.12)

    # plt.subplot(1, 4, 2)
    # plt.imshow(log_measurement_2d_reordered, cmap="magma")
    # # plt.colorbar()
    # # plt.axis("off")
    # plt.title("Reordered FWHT Measurement (Log Magnitude)", fontsize=16, y=-0.12)

    # # plt.subplot(1, 4, 3)
    # # plt.imshow(op.mask_reordered.astype(np.float32), cmap="gray")
    # # # plt.axis("off")
    # # plt.title("Subsampling Mask for Reordered FWHT", fontsize=16, y=-0.12)

    # # plt.subplot(1, 4, 4)
    # # plt.imshow(op.mask_standard, cmap="gray")
    # # # plt.axis("off")
    # # plt.title("Corresponding Mask for Standard FWHT", fontsize=16, y=-0.12)

    # plt.suptitle(
    #     f"FWHT Measurements and Masks\nCoarse J=J-{subtract_from_J}, Delta=1/{delta_divided_by}",
    #     fontsize=22,
    #     y=1.02,
    # )

    # plt.show()

    # %%

    # Test
    denoiser = deepinv.models.DRUNet(pretrained="download", device=device)
    sigma = 25 / 255  # noise level for denoiser

    def denoiser_fn_admm(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.repeat(1, 3, 1, 1)  # grayscale 2D to 4-channel batch
            denoised = denoiser(x, sigma=sigma)
            denoised = torch.mean(denoised, dim=1).squeeze(
                0
            )  # average the channels to get grayscale
        return denoised

    cg_max_iter = 100
    cg_tol = 1e-7

    print(f"Running PnP-ADMM reconstruction...")
    pnp = PnPReconstructor(physics=op, denoiser=denoiser_fn_admm, algorithm="ADMM")
    pnp_admm_result = pnp.admm_algorithm(
        measurement=y_subsampled_tensor,
        eta=1e-4,
        max_iter=10,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
    )
    # plt.imshow(pnp_admm_result_np.clip(0, 1), cmap="gray")
    # plt.title(f"Reconstruction via PnP-ADMM")
    # plt.axis("off")
    # plt.show()

    # %%

    # def run_pnp_forward_backward_splitting(
    #     measurement, A, AT, step_size: float, pnp_admm_iters: int = 10
    # ):
    #     print(f"Shape of measurement: {measurement.shape}")
    #     print(f"Image shape: {im_tensor.shape}")
    #     with torch.no_grad():
    #         pnp_result = pnp_forward_backward_splitting(
    #             measurement=measurement,
    #             image_shape=im_tensor.shape,
    #             denoiser=denoiser,
    #             step_size=step_size,
    #             max_iters=pnp_admm_iters,
    #             A=A,
    #             AT=AT,
    #         )
    #     return pnp_result

    # pnp_fbs_result = run_pnp_forward_backward_splitting(
    #     measurement=y_subsampled_tensor,
    #     A=op.A,
    #     AT=op.AT,
    #     step_size=1,  # 1 is the op norm for our PCM operator
    #     pnp_admm_iters=100,
    # )
    # pnp_fbs_result_np = pnp_fbs_result.squeeze().detach().cpu().numpy()
    # # plt.imshow(pnp_fbs_result_np.clip(0, 1), cmap="gray")
    # # plt.title(f"Reconstruction via PnP-FBS (Forward-Backward Splitting)")
    # # plt.axis("off")
    # # plt.show()

    # # %%
    # def estimate_L(A, AT, image_shape, iters: int = 30, device=None, dtype=None) -> float:
    #     """Estimate L = ||A^T A||_2 via power iteration."""
    #     dtype = dtype or torch.float32
    #     v = torch.randn(image_shape, device=device, dtype=dtype)
    #     v = v / (v.norm() + 1e-12)
    #     lam = None
    #     with torch.no_grad():
    #         for _ in range(iters):
    #             w = AT(A(v))                # apply A^T A
    #             lam = (v.flatten() @ w.flatten()).item()
    #             v = w / (w.norm() + 1e-12)
    #     return float(lam if lam is not None else 1.0)

    # L_estimated = estimate_L(
    #     A=op.A,
    #     AT=op.AT,
    #     image_shape=im_tensor.shape,
    #     device=device,
    # )
    # print(f"Estimated L = ||A^T A||_2: {L_estimated:.6f}")

    # %%
    im_np = im_tensor.squeeze().cpu().numpy()
    im_reconstructed_np = im_reconstructed_tensor.squeeze().cpu().numpy()
    pnp_admm_result_np = pnp_admm_result.squeeze().cpu().numpy()

    data_range = im_np.max() - im_np.min()

    psnr_zf = psnr(im_np, im_reconstructed_np, data_range=data_range)
    psnr_pnp = psnr(im_np, pnp_admm_result_np, data_range=data_range)

    ssim_zf = ssim(im_np, im_reconstructed_np, data_range=data_range)
    ssim_pnp = ssim(im_np, pnp_admm_result_np, data_range=data_range)

    # %%
    n_subplots = 4
    plt.figure(figsize=(n_subplots * 4, 4))

    plt.subplot(1, n_subplots, 1)
    plt.imshow(im_np, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, n_subplots, 2)
    plt.imshow(im_reconstructed_np, cmap="gray")
    plt.title(
        f"Zero-filled Reconstruction\nPSNR: {psnr_zf:.2f} dB, SSIM: {ssim_zf:.4f}"
    )
    plt.axis("off")

    plt.subplot(1, n_subplots, 3)
    plt.imshow(pnp_admm_result_np, cmap="gray")
    plt.title(f"PnP-ADMM Reconstruction\nPSNR: {psnr_pnp:.2f} dB, SSIM: {ssim_pnp:.4f}")
    plt.axis("off")

    plt.suptitle(
        "PCM Reconstructions Comparison\n"
        + f"sampling rate: {delta * 100:.2f}%, in-order measurements: {1 / (1 << (subtract_from_J * 2)) * 100:.2f}%",
        x=0.4,
        y=-0.05,
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig(
        f"pcm_recons_{sampling_rate_percent}_percent_sampling_{in_order_measurements_percent}_percent_in_order_measurements.png",
        dpi=150,
    )
    # plt.show()
