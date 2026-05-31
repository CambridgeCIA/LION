"""
This module implements the Fourier Slice Theorem-based Radon transform for sparse-view CT reconstruction, which is not mentioned in [Song2022] but is present in the codebase of it (in `score_inverse_problems/cs.py` and its dependencies). This implementation is adapted from the original codebase of [Song2022] and corrected for consistency with LION's CT geometry.

Author: Tianzhen Peng

References
----------
.. [Song2022] Song, Y., Shen, L., Xing, L., & Ermon, S. (2022). 
   "Solving Inverse Problems in Medical Imaging with Score-Based 
   Generative Models." ICLR. https://openreview.net/forum?id=vaRCHVj0uGI
"""

import math
import torch
import torch.fft

def expand_diameter(diameter: int, K: float):
    """
    Compute the expanded k-space diameter using scaling factor K. Guaranteed to be even.

    Args:
        diameter: int. The base diameter of the image grid.
        K: float. The expansion/padding factor.

    Returns:
        int: The computed expanded diameter.
    """
    expanded_diameter = int(diameter * K)
    if expanded_diameter % 2 == 1:
        expanded_diameter += 1
    return expanded_diameter


def kspace_coords(expanded_diameter: int, n_angles: int):
    """
    Generate 2D coordinates that map the 1D Fourier transformed sinograms to their k-space locations, according to the Fourier Slice Theorem. Rounded to the nearest integer.

    Args:
        expanded_diameter: int. The diameter of the expanded k-space grid. Must be even.
        n_angles: int. The number of projection angles.

    Returns:
        kx (torch.Tensor of shape (n_angles, expanded_diameter)): The horizontal indices.
        ky (torch.Tensor of shape (n_angles, expanded_diameter)): The vertical indices.
    """
    r = torch.arange(expanded_diameter, dtype=torch.float64) - expanded_diameter // 2
    a = torch.arange(n_angles, dtype=torch.float64) * (math.pi / n_angles)
    r_grid, a_grid = torch.meshgrid(r, a, indexing='xy')
    
    x = torch.round(r_grid * torch.cos(a_grid)) % expanded_diameter
    y = torch.round(r_grid * torch.sin(a_grid)) % expanded_diameter
    return x.long(), y.long()


def resize(input_tensor: torch.Tensor, oshape: tuple):
    """
    Resize a centered 2D tensor to a target shape via central zero-padding or cropping.

    Args:
        input_tensor: torch.Tensor of shape (..., in_h, in_w). The 2D tensor to be resized.
        oshape: tuple of (out_h, out_w). The target height and width.

    Returns:
        torch.Tensor of shape (..., out_h, out_w): The resized tensor.
    """
    in_h, in_w = input_tensor.shape[-2:]
    out_h, out_w = oshape[0], oshape[1]
    
    if in_h == out_h and in_w == out_w:
        return input_tensor
        
    ishift_h = max(in_h // 2 - out_h // 2, 0)
    ishift_w = max(in_w // 2 - out_w // 2, 0)
    
    oshift_h = max(out_h // 2 - in_h // 2, 0)
    oshift_w = max(out_w // 2 - in_w // 2, 0)
    
    copy_h = min(in_h - ishift_h, out_h - oshift_h)
    copy_w = min(in_w - ishift_w, out_w - oshift_w)
    
    output = torch.zeros(list(input_tensor.shape[:-2]) + [out_h, out_w], dtype=input_tensor.dtype, device=input_tensor.device)
    output[..., oshift_h:oshift_h+copy_h, oshift_w:oshift_w+copy_w] = input_tensor[..., ishift_h:ishift_h+copy_h, ishift_w:ishift_w+copy_w]
    return output


class FSTRadon(torch.nn.Module):
    """
    Fourier Slice Theorem Radon transform.
    """
    def __init__(self, size: int, n_angles: int, expansion: int = 6, device: str = 'cuda'):
        """
        Args:
            size: int. The spatial resolution of the square image (e.g. 512).
            n_angles: int. The number of sparse-view projection angles.
            expansion: int. The expansion factor for k-space.
            device: str. The target device to cache the grids and masks on.
        """
        super().__init__()
        self.size = size
        self.n_angles = n_angles
        self.expansion = expansion
        self.device = torch.device(device)
        
        # Expanded k-space diameter
        self.diameter = int(math.ceil(math.sqrt(2) * size))
        self.expanded_diameter = expand_diameter(self.diameter, expansion)
        
        # Precompute true coordinate grids and place on device during buffer registration
        kx, ky = kspace_coords(self.expanded_diameter, n_angles)
        self.register_buffer('kx', kx.to(self.device))
        self.register_buffer('ky', ky.to(self.device))
        
        # Mask for the k-space locations corresponding to the radial lines of the Fourier transformed sinogram. This corresponds to $Lambda$ in Chapter 3.1 of [Song2022].
        mask = torch.zeros((self.expanded_diameter, self.expanded_diameter), device=self.device)
        mask[self.ky, self.kx] = 1.0
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0))

    def image_to_kspace(self, image: torch.Tensor):
        """
        Map a spatial image to a 2D Cartesian k-space grid. This corresponds to $T$ in Chapter 3.1 of [Song2022].

        Args:
            image: torch.Tensor of shape (batch_size, channels, size, size). The spatial image.

        Returns:
            torch.Tensor of shape (batch_size, channels, expanded_diameter, expanded_diameter): The 2D Fourier representation.
        """
        image_pad = resize(image, (self.diameter, self.diameter))
        image_resized = resize(image_pad, (self.expanded_diameter, self.expanded_diameter))
        image_shifted = torch.fft.ifftshift(image_resized, dim=(-2, -1))
        return torch.fft.fft2(image_shifted, dim=(-2, -1))

    def kspace_to_image(self, kspace: torch.Tensor):
        """
        Reconstruct a spatial image from a 2D Cartesian k-space grid. This corresponds to $T^{-1}$ in Chapter 3.1 of [Song2022] and is the inverse of ``image_to_kspace``.

        Args:
            kspace: torch.Tensor of shape (batch_size, channels, expanded_diameter, expanded_diameter). The 2D Fourier representation.

        Returns:
            torch.Tensor of shape (batch_size, channels, size, size): The reconstructed spatial image.
        """
        image = torch.fft.fftshift(torch.fft.ifft2(kspace, dim=(-2, -1)), dim=(-2, -1))
        image_resized = resize(image, (self.diameter, self.diameter))
        return resize(image_resized.real, (self.size, self.size))

    def sino_to_kspace(self, sino: torch.Tensor):
        """
        Resize a sparse sinogram and plant its 1D Fourier slices onto a 2D Cartesian k-space grid. The result corresponds to $y$ in Chapter 3.1 of [Song2022].

        Args:
            sino: torch.Tensor of shape (batch_size, channels, n_angles, n_detectors). The spatial parallel sinogram.

        Returns:
            torch.Tensor of shape (batch_size, channels, expanded_diameter, expanded_diameter): The planted 2D Cartesian k-space grid.
        """
        sino_resized = resize(sino, (self.n_angles, self.expanded_diameter))
        
        # Core plant operator utilizing vectorized indexing
        sino_shifted = torch.fft.ifftshift(sino_resized, dim=-1)
        slices = torch.fft.fft(sino_shifted, dim=-1)
        slices = torch.fft.fftshift(slices, dim=-1)
        
        kspace_shape = list(sino.shape[:-2]) + [self.expanded_diameter, self.expanded_diameter]
        kspace = torch.zeros(kspace_shape, dtype=torch.complex64, device=sino.device)
        kspace[..., self.ky, self.kx] = slices
        return kspace

    def kspace_to_sino(self, kspace: torch.Tensor, n_detectors: int):
        """
        Extract 1D radial slices from 2D k-space and convert them to a spatial sinogram. This is the inverse of ``sino_to_kspace``.

        Args:
            kspace: torch.Tensor of shape (batch_size, channels, expanded_diameter, expanded_diameter). The Cartesian k-space grid.
            n_detectors: int. The target spatial detector width.

        Returns:
            torch.Tensor of shape (batch_size, channels, n_angles, n_detectors): The reconstructed spatial parallel sinogram.
        """
        slices = kspace[..., self.ky, self.kx]
        slices_shifted = torch.fft.ifftshift(slices, dim=-1)
        sino_shifted = torch.fft.ifft(slices_shifted, dim=-1)
        sino_resized = torch.fft.fftshift(sino_shifted, dim=-1).real
        return resize(sino_resized, (self.n_angles, n_detectors))
