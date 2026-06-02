"""
This module implements the Fourier Slice Theorem-based Radon transform for sparse-view CT reconstruction, which is not mentioned in [Song2022] but is present in the codebase of it (in `score_inverse_problems/cs.py` and its dependencies). This implementation is adapted from the original codebase of [Song2022] and corrected for consistency with LION's parallel-beam CT geometry.

Author: Tianzhen Peng

References
----------
.. [Song2022] Song, Y., Shen, L., Xing, L., & Ermon, S. (2022). 
   "Solving Inverse Problems in Medical Imaging with Score-Based 
   Generative Models." ICLR. https://openreview.net/forum?id=vaRCHVj0uGI
"""

import math
import torch
import torch.fft as fft
from LION.CTtools.ct_geometry import Geometry
import numpy as np

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


def kspace_coords(expanded_diameter: int, angles: torch.Tensor):
    """
    Generate 2D coordinates that map the 1D Fourier transformed sinograms to their k-space locations, according to the Fourier Slice Theorem. Rounded to the nearest integer.

    Args:
        expanded_diameter: int. The diameter of the expanded k-space grid. Must be even.
        angles: torch.Tensor of shape (n_angles,). The projection angles. Must be in the range [0, pi).

    Returns:
        kx (torch.Tensor of shape (n_angles, expanded_diameter)): The horizontal indices.
        ky (torch.Tensor of shape (n_angles, expanded_diameter)): The vertical indices.
    """
    assert expanded_diameter % 2 == 0, "Expanded diameter must be even."
    assert angles.min() >= 0 and angles.max() < math.pi, "Angles must be in the range [0, pi)."
    r = torch.arange(expanded_diameter, dtype=torch.float64, device=angles.device) - expanded_diameter // 2
    angles = angles.to(r.dtype)
    r_grid, a_grid = torch.meshgrid(r, angles, indexing='xy')
    
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


class FSTRadon():
    """
    Fourier Slice Theorem Radon transform. Accepts parallel-beam geometry, but image_size and detector_size are ignored and treated as equal to image_shape and detector_shape, respectively. Also, image_pos has to be at the origin.
    """
    def __init__(self, geo: Geometry, expansion: float = 6, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the FST Radon transform module. 
        
        Args:
            geo: Geometry. The geometry object containing the image and projection parameters. The angles in the geometry must be in the range [0, pi) and the number of detectors must be at least sqrt(2) times the max of image size.
            expansion: float. The expansion factor for k-space.
            device: str. The target device to cache the grids and masks on.
        
        Attributes:
            geo: Geometry. The geometry object containing the image and projection parameters.
            shape: tuple. The spatial image size (height, width).
            device: torch.device. The target device for computation.
            n_detectors: int. The number of detectors in the sinogram.
            expanded_diameter: int. The diameter of the expanded k-space grid.
        """
        assert geo.mode == "parallel", "FSTRadon currently only supports parallel-beam geometry."
        assert geo.image_pos is None or np.allclose(geo.image_pos[-2:], 0.0), "FSTRadon currently only supports images centered at the origin."
        self.geo = geo
        self.shape = geo.image_shape[-2:]
        self.device = torch.device(device)
        
        # Expanded k-space diameter
        self.n_detectors = geo.detector_shape[-1]
        assert self.n_detectors >= max(self.shape) * math.sqrt(2), "The number of detectors must be at least sqrt(2) times the max of image size."
        self.expanded_diameter = expand_diameter(self.n_detectors, expansion)
        
        # Precompute true coordinate grids and place on device during buffer registration
        kx, ky = kspace_coords(self.expanded_diameter, torch.from_numpy(geo.angles))
        self.kx = kx.to(self.device)
        self.ky = ky.to(self.device)
        
        # Mask for the k-space locations corresponding to the radial lines of the Fourier transformed sinogram. This corresponds to $Lambda$ in Chapter 3.1 of [Song2022].
        self.mask = torch.zeros((self.expanded_diameter, self.expanded_diameter), device=self.device)
        self.mask[self.ky, self.kx] = 1.0

    def image_to_kspace(self, image: torch.Tensor):
        """
        Map a spatial image to a 2D Cartesian k-space grid. This corresponds to $T$ in Chapter 3.1 of [Song2022].

        Args:
            image: torch.Tensor of shape (batch_size, channels, shape[0], shape[1]). The spatial image.

        Returns:
            torch.Tensor of shape (batch_size, channels, expanded_diameter, expanded_diameter): The 2D Fourier representation.
        """
        image_padded = resize(image, (self.n_detectors, self.n_detectors))
        image_expanded = resize(image_padded, (self.expanded_diameter, self.expanded_diameter))
        return fft.fft2(fft.ifftshift(image_expanded, dim=(-2, -1)), dim=(-2, -1))

    def kspace_to_image(self, kspace: torch.Tensor):
        """
        Reconstruct a spatial image from a 2D Cartesian k-space grid. This corresponds to $T^{-1}$ in Chapter 3.1 of [Song2022] and is the inverse of ``image_to_kspace``.

        Args:
            kspace: torch.Tensor of shape (batch_size, channels, expanded_diameter, expanded_diameter). The 2D Fourier representation.

        Returns:
            torch.Tensor of shape (batch_size, channels, shape[0], shape[1]): The reconstructed spatial image.
        """
        image_expanded = fft.fftshift(fft.ifft2(kspace, dim=(-2, -1)), dim=(-2, -1))
        image_padded = resize(image_expanded, (self.n_detectors, self.n_detectors))
        return resize(image_padded.real, self.shape)

    def sino_to_kspace(self, sino: torch.Tensor):
        """
        Resize a sparse sinogram and plant its 1D Fourier slices onto a 2D Cartesian k-space grid. The result corresponds to $y$ in Chapter 3.1 of [Song2022].

        Args:
            sino: torch.Tensor of shape (batch_size, channels, n_angles, n_detectors). The spatial parallel sinogram.

        Returns:
            torch.Tensor of shape (batch_size, channels, expanded_diameter, expanded_diameter): The planted 2D Cartesian k-space grid.
        """
        sino_expanded = resize(sino, (sino.shape[-2], self.expanded_diameter))
        sino_ft = fft.fftshift(fft.fft(fft.ifftshift(sino_expanded, dim=-1), dim=-1), dim=-1)
        
        kspace_shape = list(sino.shape[:-2]) + [self.expanded_diameter, self.expanded_diameter]
        kspace = torch.zeros(kspace_shape, dtype=torch.complex64, device=sino.device)
        kspace[..., self.ky, self.kx] = sino_ft
        return kspace

    def kspace_to_sino(self, kspace: torch.Tensor):
        """
        Extract 1D radial slices from 2D k-space and convert them to a spatial sinogram. This is the inverse of ``sino_to_kspace``.

        Args:
            kspace: torch.Tensor of shape (batch_size, channels, expanded_diameter, expanded_diameter). The Cartesian k-space grid.

        Returns:
            torch.Tensor of shape (batch_size, channels, n_angles, n_detectors): The reconstructed spatial parallel sinogram.
        """
        sino_ft = kspace[..., self.ky, self.kx]
        sino_expanded = fft.fftshift(fft.ifft(fft.ifftshift(sino_ft, dim=-1), dim=-1), dim=-1).real
        return resize(sino_expanded, (sino_expanded.shape[-2], self.n_detectors))