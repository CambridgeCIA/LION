import torch.nn as nn
import torch
from typing import Optional
from LION.utils.parameter import LIONParameter
from LION.models.LIONmodel import LIONmodel


# TODO: add option to change activation function
class MSD_Net(LIONmodel):
    def __init__(
        self,
        model_parameters: Optional[LIONParameter] = None,
    ) -> None:
        """_summary_

        Args:
            width (int): _description_
            depth (int): _description_
            dilations (Collection[int]): Vectorized matrix of dilations s_ij. i from 0 to width, j from 0 to depth.
                e.g first (width) entries correspond to dilations for first layer.
            look_back_depth: -1 = no limit

        Raises:
            ValueError: _description_
        """
        super().__init__(model_parameters)

        self.width = self.model_parameters.width
        self.depth = self.model_parameters.depth
        self.dilations = self.model_parameters.dilations
        self.look_back_depth = self.model_parameters.look_back_depth
        self.final_look_back = self.model_parameters.final_look_back
        self.activation = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU()) # implement choosing activation

        # total there should be width * depth distinct convolutions
        # so expect the same number of dilations to be given
        if len(self.dilations) != self.width * self.depth:
            raise ValueError(
                f"Expected {self.width*self.depth} dilation sizes to be given, instead given {len(self.dilations)}"
            )

        convs = nn.ModuleDict()
        for i, d in enumerate(self.dilations):
            layer = i // self.width
            channel = i % self.width
            # number of input channels depends on whether we use original input to construct this layer
            # s_ij = dilations used to calculate layer i channel j
            # TODO: maybe precalculate all of this, doesn't need to be done dynamically
            # in_channels = width * look_back_depth if i < look_back_depth else width * i + 1
            # bias set to false since we perform a BatchNorm after each Conv, so bias would be cancelled out anyway
            convs.add_module(
                f"{layer}{channel}",
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation=d, padding='same', padding_mode='reflect', bias=False),
            )

        self.convs = convs
        
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self, x):
        # maybe just combine these into a multi-channel tensor?
        lookbacks = [x]

        for layer in range(self.depth):
            if self.look_back_depth != -1 and len(lookbacks) > self.look_back_depth:
                # truncate lookbacks list to only contain layers we should use
                lookbacks = lookbacks[layer - self.look_back_depth :]

            for channel in range(self.width):
                # apply convolution to all lookback layers and sum over them
                z_ij = torch.zeros(x.shape)

                for prev in lookbacks:
                    next = self.convs[f"{layer}{channel}"](prev)
                    z_ij = torch.add(z_ij.to(next.device), next)
                
                z_ij = self.activation(z_ij)
                # add newly calculated channel to lookbacks to use in next layers
                lookbacks.append(z_ij)
        out = lookbacks[0]
        for prev in lookbacks[1:]:
            out = torch.add(out, prev)

        return self.activation(out)

    def default_parameters(self):
        width, depth = 1, 50
        dilations = []
        for i in range(depth):
            for j in range(width):
                dilations.append((((i * width) + j) % 10) + 1)
        params = LIONParameter(
            width=width,
            depth=depth,
            dilations=dilations,
            look_back_depth=-1,
            final_look_back=-1,
            activation="ReLU",
        )
        return params
