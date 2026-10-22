# This file is part of LION library
# License : BSD-3
#
# Author  : Oliver Coughlan
# Modifications: Ander Biguri, Alexander Huang
# =============================================================================

import torch
import torch.nn as nn

from collections import OrderedDict

from LION.utils.parameter import LIONParameter
from LION.models import LIONmodel

##Code for UNet
class REDCNN(LIONmodel.LIONmodel):
    def __init__(self, model_parameters=None):
        # set properties of UNet
        torch.autograd.set_detect_anomaly(True)
        if model_parameters is None:
            model_parameters = REDCNN.default_parameters()
        super().__init__(model_parameters)

        self.encoder1 = self.blockdown(
            model_parameters.inChan,
            model_parameters.baseDim,#96
            24,
            model_parameters.dropFac,
            "enc1"
        )
        self.encoder2 = self.blockdown(
            model_parameters.baseDim,
            model_parameters.baseDim-4,#92
            23,
            model_parameters.dropFac,
            "enc2"
        )
        self.encoder3 = self.blockdown(
            model_parameters.baseDim-4,#92
            model_parameters.baseDim-4*2,#88
            22,#32
            model_parameters.dropFac,
            "enc3"
        )
        self.encoder4 = self.blockdown(
            model_parameters.baseDim-4*2,#88
            model_parameters.baseDim-4*3,#84
            21,#32
            model_parameters.dropFac,
            "enc4"
        )
        self.encoder5 = self.blockdown(
            model_parameters.baseDim-4*3,#88
            model_parameters.baseDim-4*4,#80
            20,#32
            model_parameters.dropFac,
            "enc5"
        )
        # self.base = self.blockdown(
        #     model_parameters.baseDim ,
        #     model_parameters.baseDim ,
        #     model_parameters.noGrp,
        #     model_parameters.dropFac,
        #     name="base",
        # )
        # self.upconv1 = nn.ConvTranspose2d(
        #     model_parameters.baseDim ,
        #     model_parameters.baseDim ,
        #     model_parameters.kerSiz,
        #     stride=2,
        # )
       
        # self.encoder1 = "enc1", nn.Conv2d(1, model_parameters.baseDim, kernel_size=5, stride=1, padding=0)
        # self.encoder2 = "enc2",nn.Conv2d(model_parameters.baseDim, model_parameters.baseDim, kernel_size=5, stride=1, padding=0)
        # self.encoder3 = nn.Sequential(OrderedDict(["enc3",nn.Conv2d(model_parameters.baseDim, model_parameters.baseDim, kernel_size=5, stride=1, padding=0)]))
        # self.encoder4 = nn.Sequential(OrderedDict(["enc4",nn.Conv2d(model_parameters.baseDim, model_parameters.baseDim, kernel_size=5, stride=1, padding=0)]))
        # self.encoder5 = nn.Sequential(OrderedDict(["enc5",nn.Conv2d(model_parameters.baseDim, model_parameters.baseDim, kernel_size=5, stride=1, padding=0)]))

        # self.decoder1 = nn.Sequential(OrderedDict(["dec1", nn.ConvTranspose2d(model_parameters.baseDim, model_parameters.baseDim, kernel_size=5, stride=1, padding=0)]))
        # self.decoder2 = nn.Sequential(OrderedDict(["dec2",nn.ConvTranspose2d(model_parameters.baseDim, model_parameters.baseDim, kernel_size=5, stride=1, padding=0)]))
        # self.decoder3 = nn.Sequential(OrderedDict(["dec3",nn.ConvTranspose2d(model_parameters.baseDim, model_parameters.baseDim, kernel_size=5, stride=1, padding=0)]))
        # self.decoder4 = nn.Sequential(OrderedDict(["dec4",nn.ConvTranspose2d(model_parameters.baseDim, model_parameters.baseDim, kernel_size=5, stride=1, padding=0)]))
        # self.decoder5 = nn.Sequential(OrderedDict(["dec5",nn.ConvTranspose2d(model_parameters.baseDim, 1, kernel_size=5, stride=1, padding=0)]))
        # self.relu = "relu1",nn.ReLU()
        
        # param.inChan = 1
        # param.outChan = 1
        # param.baseDim = 96
        # param.dropFac = 0
        
        
        self.decoder1 = self.blockup(
            model_parameters.baseDim-4*4,
            model_parameters.baseDim-4*3, #this needs to be dvisible by
            21,#this
            model_parameters.dropFac,
            name="dec1",
        )
        self.decoder2 = self.blockup(
            model_parameters.baseDim-4*3,
            model_parameters.baseDim-4*2,
            22,
            model_parameters.dropFac,
            name="dec2",
        )
        self.decoder3 = self.blockup(
            model_parameters.baseDim-4*2,
            model_parameters.baseDim-4,
            23,
            model_parameters.dropFac,
            name="dec3",
        )

        self.decoder4 = self.blockup(
            model_parameters.baseDim-4,
            model_parameters.baseDim,
            24,
            model_parameters.dropFac,
            name="dec4",
        )

        self.decoder5 = self.blockup(
            model_parameters.baseDim,
            model_parameters.outChan,
            1,
            model_parameters.dropFac,
            # group = 1, #SUSSY CODE PLEASE CHECK
            name="dec5",
        )

        self.relu = nn.ReLU(True)


        # self.conv = nn.Conv2d(
        #     model_parameters.baseDim, model_parameters.outChan, kernel_size=1
        # )
    
    def forward(self, x):

        jump1 = x
        enc1 = self.relu(self.encoder1(x))
        enc2 = self.relu(self.encoder2(enc1))

        jump2 = enc2

        enc3 = self.relu(self.encoder3(enc2))

        enc4 = self.relu(self.encoder4(enc3))

        jump3 = enc4
        enc5 = self.relu(self.encoder5(enc4))

       
        dec1 = self.decoder1(enc5)
        dec1 = dec1.add(jump3)
        dec2 = self.relu(self.decoder2(self.relu(dec1)))



        dec3 = self.decoder3(dec2)
        dec3 = dec3.add(jump2)
        dec4 = self.relu(self.decoder4(self.relu(dec3)))

        dec5 = self.decoder5(dec4)
        dec5 = self.relu(dec5.add(jump1))

        return dec5
#use relu after convtranspose for upconv


#padding of size 2,2 and 4,0 run
    def blockdown(self, inChan, outChan, noGrp, dropFac, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(inChan, outChan, kernel_size=5, stride = 1, padding=0, bias=True),
                    ),
                    (name + "gn1", nn.GroupNorm(noGrp, outChan)),
                    # (name + "relu1", nn.ReLU(True)),
                    (name + "dr1", nn.Dropout(p=dropFac)),
                ]
            )
        )
    
    def blockup(self, inChan, outChan, noGrp, dropFac, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "deconv1",
                        nn.ConvTranspose2d(inChan, outChan, kernel_size=5, stride = 1, padding=0, bias=True),
                    ),
                    (name + "gn1", nn.GroupNorm(noGrp, outChan)),
                    # (name + "relu1", nn.ReLU(True)),
                    (name + "dr1", nn.Dropout(p=dropFac)),
                ]
            )
        )
#self, in_channels=1, out_channels=96, num_layers=10, kernel_size=5, padding=0
    @staticmethod
    def default_parameters():
        param = LIONParameter()
        param.inChan = 1
        param.outChan = 1
        param.baseDim = 96
        param.dropFac = 0
        param.kerSiz = 5
        param.noGrp = 32
        param.model_input_type = LIONmodel.ModelInputType.IMAGE

        return param
