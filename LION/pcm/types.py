from __future__ import annotations

import torch
from jaxtyping import Float

GrayscaleImage2D = Float[torch.Tensor, "height width"]
Measurement1D = Float[torch.Tensor, "num_measurements"]
