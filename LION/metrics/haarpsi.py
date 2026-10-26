from __future__ import annotations

import torch
import torch.nn as nn

from LION.exceptions.exceptions import ShapeMismatchException


class HAARPsi(nn.Module):
    """HaarPSI: the Haar wavelet-based perceptual similarity index, for
    (potentially) batched pytorch tensors, used like LION's PSNR and SSIM.

    Reisenhofer, Bosse, Kutyniok and Wiegand, "A Haar wavelet-based perceptual
    similarity index for image quality assessment", Signal Processing: Image
    Communication 61 (2018). PyTorch implementation by Sören Dittmer and Clemens
    Karner (through project FWF T1307-N/Anna Breger), from the TensorFlow
    implementation by David Neumann and the MATLAB one by Rafael Reisenhofer.
    It matches ``piq.haarpsi`` for images in [0, data_range].

    Args (constructor):
        preprocess_with_subsampling: subsample the images by 2 first (as in
            the paper). Default True.
        C, a: the algorithm's constants, determined experimentally by the
            authors of HaarPSI (defaults 30 and 4.2, for images in [0, 255]).
    """

    # a similarity score: higher is better
    higher_is_better = True

    def __init__(self, preprocess_with_subsampling: bool = True, C=30.0, a=4.2):
        super().__init__()
        self.preprocess_with_subsampling = preprocess_with_subsampling
        self.C = C
        self.a = a

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        reduce: str | None = None,
        batched=True,
        data_range: float | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input image(s), (B, C, H, W) or (B, H, W) if batched, else
                (C, H, W) or (H, W). C is 1 (grayscale) or 3 (RGB).
            target: target image(s) to compare x against, same shape as x.
            reduce: if batched=False has no effect. Otherwise specifies what
                operation should be done to aggregate the scores along the batch
                axis. Defaults to None, in which case a tensor of shape
                (batch size) is returned.
            batched: whether x and target are batched or not.
            data_range: the range of the image values, for images in
                [0, data_range] (e.g. 1 for images in [0, 1]), as in
                ``piq.haarpsi``. Defaults to None: each pair is mapped with its
                target's minimum and range (max - min) to a target in [0, 1],
                like the range PSNR and SSIM use. Images may have values outside
                the range (e.g. noisy reconstructions).

        Returns:
            The HaarPSI of each sample, in [0, 1], 1 for identical images.
        """
        score = self.maps(x, target, batched=batched, data_range=data_range)[0]
        if not batched:
            return score[0]
        if reduce is None:
            return score
        if reduce == "mean":
            return torch.mean(score)
        raise ValueError(
            f"expected one of 'mean' or None for parameter 'reduce', got {reduce}"
        )

    def maps(self, x, target, batched=True, data_range=None):
        """HaarPSI with its local similarity maps and weights.

        Same arguments as ``forward``. Returns ``(score, local_similarities,
        weights)``: score has shape (B,), the others (B, n_channels, H, W), with
        H and W halved if subsampling and n_channels 2 for grayscale or 3 for
        colour images (always batched).
        """
        if x.shape != target.shape:
            raise ShapeMismatchException(
                f"x (shape {x.shape}) and target (shape {target.shape}) tensors must match to compare with HAARPsi"
            )
        if not batched:
            x, target = x.unsqueeze(0), target.unsqueeze(0)
        if x.dim() == 3:  # (B, H, W)
            x, target = x.unsqueeze(1), target.unsqueeze(1)
        if x.dim() != 4 or x.shape[1] not in {1, 3}:
            raise ValueError(
                "HAARPsi expects grayscale or RGB images: (B, C, H, W) or (B, H, W) "
                f"if batched, else (C, H, W) or (H, W), with C 1 or 3; got {tuple(x.shape)}"
            )
        x, target = x.float(), target.float()
        if data_range is None:  # the target in [0, 1]
            low = torch.amin(target, dim=(1, 2, 3), keepdim=True)
            data_range = torch.amax(target, dim=(1, 2, 3), keepdim=True) - low
            x, target = x - low, target - low
        # the constant C is for images in [0, 255]
        scale = 255.0 / data_range
        score, local_similarities, weights = self._haarpsi(target * scale, x * scale)
        return score[:, 0], local_similarities, weights

    def _haarpsi(self, ref: torch.Tensor, deg: torch.Tensor):
        """HaarPSI of (B, C, H, W) images with values in [0, 255]."""
        is_color_image = deg.shape[1] == 3

        if is_color_image:
            ref_y = (
                0.299 * ref[:, 0, :, :]
                + 0.587 * ref[:, 1, :, :]
                + 0.114 * ref[:, 2, :, :]
            )
            deg_y = (
                0.299 * deg[:, 0, :, :]
                + 0.587 * deg[:, 1, :, :]
                + 0.114 * deg[:, 2, :, :]
            )
            ref_i = (
                0.596 * ref[:, 0, :, :]
                - 0.274 * ref[:, 1, :, :]
                - 0.322 * ref[:, 2, :, :]
            )
            deg_i = (
                0.596 * deg[:, 0, :, :]
                - 0.274 * deg[:, 1, :, :]
                - 0.322 * deg[:, 2, :, :]
            )
            ref_q = (
                0.211 * ref[:, 0, :, :]
                - 0.523 * ref[:, 1, :, :]
                + 0.312 * ref[:, 2, :, :]
            )
            deg_q = (
                0.211 * deg[:, 0, :, :]
                - 0.523 * deg[:, 1, :, :]
                + 0.312 * deg[:, 2, :, :]
            )
            # (B, H, W) -> (B, 1, H, W)
            ref_y = ref_y.unsqueeze(1)
            deg_y = deg_y.unsqueeze(1)
            ref_i = ref_i.unsqueeze(1)
            deg_i = deg_i.unsqueeze(1)
            ref_q = ref_q.unsqueeze(1)
            deg_q = deg_q.unsqueeze(1)
        else:
            ref_y = ref
            deg_y = deg

        if self.preprocess_with_subsampling:
            ref_y = self._subsample(ref_y)
            deg_y = self._subsample(deg_y)
            if is_color_image:
                ref_i = self._subsample(ref_i)
                deg_i = self._subsample(deg_i)
                ref_q = self._subsample(ref_q)
                deg_q = self._subsample(deg_q)

        n_scales = 3
        coeffs_ref_y = self._haar_wavelet_decompose(
            ref_y, n_scales
        )  # n_scales x n_samples x 1 x height x width
        coeffs_deg_y = self._haar_wavelet_decompose(deg_y, n_scales)
        if is_color_image:
            coefficients_ref_i = torch.abs(
                self._convolve2d(ref_i, torch.ones((2, 2), device=ref.device) / 4.0)
            )
            coefficients_deg_i = torch.abs(
                self._convolve2d(deg_i, torch.ones((2, 2), device=ref.device) / 4.0)
            )
            coefficients_ref_q = torch.abs(
                self._convolve2d(ref_q, torch.ones((2, 2), device=ref.device) / 4.0)
            )
            coefficients_deg_q = torch.abs(
                self._convolve2d(deg_q, torch.ones((2, 2), device=ref.device) / 4.0)
            )

        n_samples, _, height, width = ref_y.shape
        if is_color_image:
            n_channels = 3
        else:
            n_channels = 2

        local_similarities = torch.zeros(
            n_channels, n_samples, 1, height, width, device=ref_y.device
        )
        weights = torch.zeros(
            n_channels, n_samples, 1, height, width, device=ref_y.device
        )

        for orientation in [0, 1]:
            weights[orientation] = self._get_weights_for_orientation(
                coeffs_deg_y,
                coeffs_ref_y,
                n_scales,
                orientation,
                n_samples,
                height,
                width,
            )
            local_similarities[
                orientation
            ] = self._get_local_similarity_for_orientation(
                coeffs_deg_y, coeffs_ref_y, n_scales, orientation
            )

        if is_color_image:
            similarity_i = (2 * coefficients_ref_i * coefficients_deg_i + self.C) / (
                coefficients_ref_i**2 + coefficients_deg_i**2 + self.C
            )
            similarity_q = (2 * coefficients_ref_q * coefficients_deg_q + self.C) / (
                coefficients_ref_q**2 + coefficients_deg_q**2 + self.C
            )
            local_similarities[2, :, :, :, :] = (similarity_i + similarity_q) / 2
            weights[2, :, :, :, :] = (
                weights[0, :, :, :, :] + weights[1, :, :, :, :]
            ) / 2

        dims = (0, 3, 4)
        pre_logit = torch.sum(
            torch.sigmoid(self.a * local_similarities) * weights, dim=dims
        ) / torch.sum(weights, dim=dims)
        similarity = self.logit(pre_logit) ** 2

        assert similarity.shape == (n_samples, 1), similarity.shape
        assert local_similarities.shape == (
            n_channels,
            n_samples,
            1,
            height,
            width,
        ), local_similarities.shape
        assert weights.shape == (n_channels, n_samples, 1, height, width), weights.shape
        local_similarities = local_similarities[:, :, 0].permute(1, 0, 2, 3)
        weights = weights[:, :, 0].permute(1, 0, 2, 3)
        assert local_similarities.shape == (
            n_samples,
            n_channels,
            height,
            width,
        ), local_similarities.shape
        assert weights.shape == (n_samples, n_channels, height, width), weights.shape
        return similarity, local_similarities, weights

    def logit(self, value):
        return torch.log(value / (1 - value)) / self.a

    def _get_local_similarity_for_orientation(
        self, coeffs_deg_y, coeffs_ref_y, n_scales, orientation
    ):
        n_samples = coeffs_ref_y.shape[1]
        height, width = coeffs_ref_y.shape[-2:]
        coeffs_ref_y_magnitude = coeffs_ref_y.abs()[
            (orientation * n_scales, 1 + orientation * n_scales), :, :
        ]
        coeffs_deg_y_magnitude = coeffs_deg_y.abs()[
            (orientation * n_scales, 1 + orientation * n_scales), :, :
        ]
        a = 2 * coeffs_ref_y_magnitude * coeffs_deg_y_magnitude + self.C
        b = coeffs_ref_y_magnitude**2 + coeffs_deg_y_magnitude**2 + self.C
        frac = a / b
        assert frac.shape == (2, n_samples, 1, height, width), frac.shape
        local_similarity = (frac[0] + frac[1]) / 2
        assert local_similarity.shape == (
            n_samples,
            1,
            height,
            width,
        ), local_similarity.shape
        return local_similarity

    def _get_weights_for_orientation(
        self,
        coeffs_deg_y,
        coeffs_ref_y,
        n_scales,
        orientation,
        n_samples,
        height,
        width,
    ):
        maxs = torch.maximum(
            coeffs_ref_y[2 + orientation * n_scales].abs(),
            coeffs_deg_y[2 + orientation * n_scales].abs(),
        )
        assert maxs.shape == (n_samples, 1, height, width), maxs.shape
        return maxs

    def _subsample(self, image):
        kernel = torch.ones(2, 2, device=image.device) / 4.0
        subsampled_image = self._convolve2d(image, kernel)
        subsampled_image = subsampled_image[:, :, ::2, ::2]
        return subsampled_image

    def _convolve2d(self, data, kernel):
        width, height = data.shape[-2:]
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        rotated_data = torch.rot90(data, 2, [2, 3])
        padding = (kernel.shape[2] // 2, kernel.shape[3] // 2)
        res = torch.nn.functional.conv2d(rotated_data, kernel, padding=padding)
        res = torch.nn.functional.interpolate(
            res, (width, height), mode="nearest", align_corners=None
        )
        res = torch.rot90(res, 2, [2, 3])
        return res

    def _get_haar_filter(self, scale, device):
        haar_filter = 2**-scale * torch.ones(2**scale, 2**scale, device=device)
        haar_filter[: haar_filter.shape[0] // 2, :] = -haar_filter[
            : haar_filter.shape[0] // 2, :
        ]
        return haar_filter

    def _haar_wavelet_decompose(self, image, number_of_scales):
        coefficients = torch.zeros(
            2 * number_of_scales, *image.shape, device=image.device
        )
        for scale in range(1, number_of_scales + 1):
            haar_filter = self._get_haar_filter(scale, image.device)
            coefficients[scale - 1] = self._convolve2d(image, haar_filter)
            coefficients[scale + number_of_scales - 1] = self._convolve2d(
                image, haar_filter.t()
            )
        return coefficients
