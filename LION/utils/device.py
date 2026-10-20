"""The device LION uses when none is given.

Pick the GPU once, at the top of a script, and LION follows it:

    torch.cuda.set_device(2)          # or
    torch.set_default_device("cuda:2")

Anything given an explicit ``device=`` uses that instead. On machines with many
GPUs where only one may be used, the most robust option is to make only that GPU
visible to the process: ``CUDA_VISIBLE_DEVICES=2 python script.py``.
"""

from __future__ import annotations

import torch


def default_device() -> torch.device:
    """Return the device LION uses when no device is given.

    In order of priority:

    1. ``torch.set_default_device(...)``, if set to anything other than the CPU;
    2. the current CUDA device (``torch.cuda.set_device(k)``, cuda:0 by default),
       if CUDA is available;
    3. the CPU.
    """
    get_default = getattr(torch, "get_default_device", None)  # torch >= 2.3
    if get_default is not None:
        device = get_default()
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type != "cpu":
            return device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")
