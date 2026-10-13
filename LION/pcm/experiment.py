from __future__ import annotations

import torch


def resolve_device(device_name: str) -> torch.device:
    """Resolve the torch device from a string.

    Args:
        device_name: Requested device string, or ``'auto'``.

    Returns:
        device: Resolved torch device.
    """
    if device_name != "auto":
        return torch.device(device_name)

    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
