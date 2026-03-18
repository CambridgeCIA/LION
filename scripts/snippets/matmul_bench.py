from functools import partial
import time
import torch
from tqdm import tqdm as std_tqdm

# Use tqdm with dynamic column width that adapts to the terminal width
tqdm = partial(std_tqdm, dynamic_ncols=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device, "torch:", torch.__version__)

x = torch.randn(4096, 4096, device=device)
w = torch.randn(4096, 4096, device=device)

if device.type == "mps":
    torch.mps.synchronize()

t0 = time.time()
for _ in tqdm(range(2000)):
    y = x @ w

if device.type == "mps":
    torch.mps.synchronize()

print("seconds:", time.time() - t0, "y.device:", y.device)
