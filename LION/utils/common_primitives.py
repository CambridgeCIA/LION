# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  :
# =============================================================================
from typing import TypeAlias

import torch

DeviceLikeType: TypeAlias = str | torch.device | int
