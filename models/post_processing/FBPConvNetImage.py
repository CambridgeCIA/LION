# This file is part of LION library
# License : BSD-3
#
# Author  : Emilia Zabrzanska
# Modifications: -
# =============================================================================

import torch
from LION.models.post_processing.FBPConvNet import FBPConvNet


class FBPConvNetImage(FBPConvNet):
    """
    FBPConvNet that takes a pre-computed FBP image as input instead of calling FDK inside the forward pass.
    """

    def forward(self, image):

        block_1_res = self.block_1_down(image)
        block_2_res = self.block_2_down(self.down_1(block_1_res))
        block_3_res = self.block_3_down(self.down_2(block_2_res))
        block_4_res = self.block_4_down(self.down_3(block_3_res))

        res = self.block_bottom(self.down_4(block_4_res))
        res = self.block_1_up(torch.cat((block_4_res, self.up_1(res)), dim=1))
        res = self.block_2_up(torch.cat((block_3_res, self.up_2(res)), dim=1))
        res = self.block_3_up(torch.cat((block_2_res, self.up_3(res)), dim=1))
        res = self.block_4_up(torch.cat((block_1_res, self.up_4(res)), dim=1))
        res = self.block_last(res)

        return image + res
