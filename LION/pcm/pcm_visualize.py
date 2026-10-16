"""Shared visualization utilities for PCM reconstructions."""

from __future__ import annotations

import torch


def absolute_error(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the pointwise absolute reconstruction error."""
    return torch.abs(target - recon)
