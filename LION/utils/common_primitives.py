# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Thanh Trung (Troy) Vu
# =============================================================================
from typing import TypeAlias

import torch

DeviceLikeType: TypeAlias = str | torch.device | int
