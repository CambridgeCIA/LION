# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Times MSDNet on a GPU: one forward and backward pass, by implementation,
depth and image size, with the peak GPU memory. To decide how to make MSDNet
fast (removed before merging).

    CUDA_VISIBLE_DEVICES=3 python scripts/msd_timing.py

Implementations: "triton" (the Triton kernels) and "torch, groups of 16"
(plain torch, 16 input channels at a time). The forward pass is also timed
alone, to see what the backward costs.
"""

import time

import torch

from LION.models.CNNs import _msd_triton
from LION.models.CNNs import MSDNet as module
from LION.models.CNNs.MSDNet import MSDNet

triton_available = _msd_triton.available
IMPLEMENTATIONS = {
    "triton": (triton_available, 16),
    "torch, groups of 16": (lambda x: False, 16),
}
SIZES = {"2DeteCT sinogram 3600x956": (3600, 956), "image 1024x1024": (1024, 1024)}
DEPTHS = (10, 50, 100)
REPEATS = 3

device = torch.device("cuda")
print(torch.cuda.get_device_name(device), torch.__version__)
print(
    "| size | depth | implementation | forward (s) | forward+backward (s) | peak memory (GB) |"
)
print("|---|---|---|---|---|---|")
for size_name, shape in SIZES.items():
    for depth in DEPTHS:
        parameters = MSDNet.default_parameters()
        parameters.depth = depth
        model = MSDNet(parameters).to(device)
        x = torch.randn(1, 1, *shape, device=device)
        for name, (available, group) in IMPLEMENTATIONS.items():
            _msd_triton.available = available
            module.CHANNELS_AT_A_TIME = group
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            try:
                model(x).sum().backward()  # warm up (Triton compiles once)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(REPEATS):
                    model.zero_grad()
                    model(x).sum().backward()
                torch.cuda.synchronize()
                seconds = (time.perf_counter() - start) / REPEATS
                memory = torch.cuda.max_memory_allocated(device) / 1e9
                with torch.no_grad():
                    start = time.perf_counter()
                    for _ in range(REPEATS):
                        model(x)
                    torch.cuda.synchronize()
                forward = (time.perf_counter() - start) / REPEATS
                print(
                    f"| {size_name} | {depth} | {name} | {forward:.3f} "
                    f"| {seconds:.3f} | {memory:.2f} |"
                )
            except torch.OutOfMemoryError:
                print(f"| {size_name} | {depth} | {name} | | out of memory | |")
            model.zero_grad()
            torch.cuda.empty_cache()
