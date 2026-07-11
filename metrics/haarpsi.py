import torch
import torch.nn as nn


class HAARPsi(nn.Module):
    def __init__(self, preprocess_with_subsampling: bool = True, C=30.0, a=4.2):
        """
        This is a Python PyTorch implementation of the HaarPSI algorithm as presented in "A Haar wavelet-based perceptual similarity index for image quality assessment" by Rafael Reisenhofer, Sebastian Bosse, Gitta Kutyniok and Thomas Wiegand.

        Converted by Sören Dittmer and Clemens Karner (through project FWF T1307-N/Anna Breger) from the original Python tensorflow implementation written by David Neumann and from the original MATLAB implementation written by Rafael Reisenhofer.

        Please note that this implementation requires images scaled to [0,1], while the original implementation expects images with image values in [0,255].

        Last update: 28.05.2024

        Parameters:
            Mandatory:
                ref: torch.Tensor
                    The reference image, can be an RGB or grayscale image with image values scaled to [0,1]
                    in one of the following formats:
                        (#samples, #channels, width, height)
                        (#channels, width, height)
                        (width, height)
                deg: torch.Tensor
                    The degraded image, can be an RGB or grayscale image with image values scaled to [0,1]
                    in one of the following formats:
                        (#samples, #channels, width, height)
                        (#channels, width, height)
                        (width, height)
            Optional:
                preprocess_with_subsampling: boolean
                    Default: True
                    Determines if subsampling is performed.
                C: float
                    Default: 30.0
                    A paramter of the algorithm, which has been experimentally determined by the creators of HaarPSI.
                α: float
                    Default: 4.2
                    A paramter of the algorithm, which has been experimentally determined by the creators of HaarPSI.

        Returns:
            (similarity_score, local_similarity, weights): (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        self.preprocess_with_subsampling = preprocess_with_subsampling
        self.C = C
        self.a = a

    def forward(self, ref: torch.Tensor, deg: torch.Tensor):
        assert ref.shape == deg.shape, "The images must have the same dimensions"
        assert len(ref.shape) in {
            2,
            3,
            4,
        }, "Grayscale and color single images and batches are supported"
        assert ref.dtype == torch.float32, "The images must be of type float32"
        assert ref.dtype == deg.dtype, "The images must have the same dtype"
        assert torch.all(ref >= 0) and torch.all(
            ref <= 1
        ), "The images must be in the range [0, 1]"
        assert torch.all(deg >= 0) and torch.all(
            deg <= 1
        ), "The images must be in the range [0, 1]"

        ref = 255 * ref
        deg = 255 * deg

        if len(ref.shape) == 2:
            ref = ref.unsqueeze(0).unsqueeze(0)
            deg = deg.unsqueeze(0).unsqueeze(0)
        elif len(ref.shape) == 3:
            ref = ref.unsqueeze(0)
            deg = deg.unsqueeze(0)

        assert deg.shape[1] in {1, 3}, "The images must be grayscale or color."
        assert (
            ref.shape[1] == deg.shape[1]
        ), "The images must both have the same number of color channels."

        if deg.shape[1] == 3:
            is_color_image = True
        else:
            is_color_image = False

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
            ref_y = ref_y.unsqueeze(0)
            deg_y = deg_y.unsqueeze(0)
            ref_i = ref_i.unsqueeze(0)
            deg_i = deg_i.unsqueeze(0)
            ref_q = ref_q.unsqueeze(0)
            deg_q = deg_q.unsqueeze(0)
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
                self._convolve2d(ref_i, torch.ones((2, 2)) / 4.0)
            )
            coefficients_deg_i = torch.abs(
                self._convolve2d(deg_i, torch.ones((2, 2)) / 4.0)
            )
            coefficients_ref_q = torch.abs(
                self._convolve2d(ref_q, torch.ones((2, 2)) / 4.0)
            )
            coefficients_deg_q = torch.abs(
                self._convolve2d(deg_q, torch.ones((2, 2)) / 4.0)
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
