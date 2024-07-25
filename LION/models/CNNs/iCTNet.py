from typing import Tuple
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from LION.CTtools.ct_geometry import Geometry
from LION.models.LIONmodel import LIONmodel, ModelInputType, ModelParams


class HardThreshold(nn.Module):
    def __init__(self, lamb: float) -> None:
        super().__init__()
        self.lamb = lamb
        self.shrink = nn.Hardshrink(lamb)

    def forward(self, x):
        return self.shrink(x)


class iCTNetParams(ModelParams):
    def __init__(
        self,
        out_shape: Tuple[int, int],
        lambds: Tuple[float, ...],
        alphas: Tuple[int, int],
        beta: int,
        phi: float
    ):
        super().__init__(model_input_type=ModelInputType.SINOGRAM)

        # check to make sure lambds is valid

        self.out_shape = out_shape
        self.lambds = lambds
        self.alphas = alphas
        self.beta = beta
        self.phi = phi


class iCTNet(LIONmodel):
    def __init__(
        self,
        geometry_parameters: Geometry,
        model_parameters: iCTNetParams | None = None
    ):
        super().__init__(model_parameters, geometry_parameters)
        assert isinstance(self.model_parameters, iCTNetParams)
        assert self.geo is not None, "iCTNet requires a geo to initialize"

        # sinogram have shape B, C (1), Nv, Nc
        lamb1, lamb2, lamb3, lamb4, lamb5 = self.model_parameters.lambds
        alph1, alph2 = self.model_parameters.alphas
        in_channels, Nc, Nv = *self.geo.detector_shape, len(self.geo.angles)

        beta = self.model_parameters.beta
        outw, outh = self.model_parameters.out_shape

        # layers
        # 1 - 3 act on detector element dimension
        self.l1 = nn.Sequential(nn.Conv2d(in_channels, 64, (3, 1), 1, "same"), HardThreshold(lamb1))
        self.l2 = nn.Sequential(nn.Conv2d(64, 64, (3, 1), 1, "same"), HardThreshold(lamb2))
        self.l3 = nn.Sequential(nn.Conv2d(129, 1, (3, 1), 1, "same"), HardThreshold(lamb3))

        # 4 - 5 act on view angle dimension
        self.l4 = nn.Sequential(nn.Conv2d(Nv, alph1 * Nv, (1, 1), 1, "same"), HardThreshold(lamb4))
        self.l5 = nn.Sequential(nn.Conv2d(alph1 * Nv, alph2 * Nv, (1, 1), 1, "same"), HardThreshold(lamb5))

        self.l6 = nn.Conv2d(1, 1, (Nc, alph2 * Nv) , 1, "same", bias=False)
        self.l7 = nn.Sequential(nn.Conv2d(1, 16, (beta, 1), 1, "same"), nn.Tanh())
        self.l8 = nn.Sequential(nn.Conv2d(16, 1, (beta, 1), 1, "same"), nn.Tanh())
        self.l9 = nn.Sequential(nn.Conv2d(Nc, Nc, (1, 1), 1, "same"), nn.Tanh())
        self.l10 = nn.Conv2d(Nc, outw * outh, (1, 1), 1, "same", bias=False)

        # figure out what's going on with l11


        self.l12 = nn.Conv2d(alph2 * Nv, 1, (1, 1), 1, "same", bias=False)

    
    def forward(self, x):
        # print("x:", x.shape)
        l1in = torch.permute(x, (0, 1, 3, 2))
        # print("l1in:", l1in.shape)
        l1out = self.l1(l1in)
        # print("l1out:", l1out.shape)
        l2out = self.l2(l1out)
        # print("l2out:", l2out.shape)
        l3in = torch.cat((l1in, l1out, l2out), dim=1)
        # print("l3in:", l3in.shape)
        l3out = self.l3(l3in)
        # print("l3out:", l3out.shape)
        l4in = torch.permute(l3out, (0, 3, 2, 1))
        # print("l4in:", l4in.shape)
        l4out = self.l4(l4in)
        # print("l4out:", l4out.shape)
        l5out = self.l5(l4out)
        # print("l5out:", l5out.shape)
        l6in = torch.permute(l5out, (0, 3, 2, 1))
        # print("l6in:", l6in.shape)
        l6out = self.l6(l6in)
        # print("l6out:", l6out.shape)
        l7out = self.l7(l6out)
        # print("l7out", l7out.shape)
        l8out = self.l8(l7out)
        # print("l8out:", l8out.shape)
        l9in = torch.permute(l8out, (0, 2, 1, 3))
        # print("l9in:", l9in.shape)
        l9out = self.l9(l9in)
        # print("l9out", l9out.shape)
        l10out = self.l10(l9out)
        # print("l10out:", l10out.shape)

        assert isinstance(self.model_parameters, iCTNetParams) # they are I promise
        reshaped = torch.reshape(l10out, (l10out.shape[0], *self.model_parameters.out_shape, *l10out.shape[2:]))
        reshaped = torch.permute(reshaped, (0, 4, 1, 2, 3))
        # print("reshaped:", reshaped.shape)
        rotated_mats = []
        for i in range(reshaped.shape[1]):
            sub_mat = reshaped[:, i, :, :, :]
            # print("sub_mat:", sub_mat.shape)
            sub_mat = torch.permute(sub_mat, (0, 3, 1, 2))
            # print("sub_mat:", sub_mat.shape)
            rot_angle = self.model_parameters.phi * (reshaped.shape[1] - i)
            rotated_sub_mat = TF.rotate(sub_mat, rot_angle)
            # print("rotated_sub_mat:", rotated_sub_mat.shape)
            rotated_sub_mat = F.interpolate(rotated_sub_mat, self.model_parameters.out_shape, mode="bilinear")
            # print("interpolated:", rotated_sub_mat.shape)
            re_vectorised = torch.reshape(rotated_sub_mat, (*rotated_sub_mat.shape[:2], l10out.shape[1]))
            # print("re_vectorised:", re_vectorised.shape)
            rotated_mats.append(re_vectorised)

        l12in = torch.stack(rotated_mats, dim=1)
        # print("l12in:", l12in.shape)
        l12out = self.l12(l12in)
        # print("l12out:", l12out.shape)
        output = torch.reshape(l12out, (l12out.shape[0], l12out.shape[1], *self.model_parameters.out_shape))
        # print("output:", output.shape)
        
        return output

    @staticmethod
    def default_parameters(mode="sparse") -> ModelParams:

        if mode == "sparse":
            alphas = (2, 4)
        elif mode == "dense":
            alphas = (1, 1)
        else:
            raise ValueError(f"iCTNet mode {mode} not recognised. Expected 'non-interior' or 'interior'")

        return iCTNetParams(
                (512, 512), (1e-5,) * 3 + (1e-8,) * 2, alphas, 1, torch.pi / 492
            )




