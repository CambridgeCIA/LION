import numpy as np
import torch
import torch.nn as nn

from ts_algorithms import fdk
from collections import OrderedDict

import LION.CTtools.ct_geometry as ct
from tomosipo.torch_support import to_autograd
from LION.utils.parameter import LIONParameter
from LION.models import LIONmodel

from LION.models.CNNs.UNets.Unet import UNet

import scipy.sparse

torch.autograd.set_detect_anomaly(True)


class ItNetLBFGSTV(LIONmodel.LIONmodel):
    def __init__(self, geometry: ct.Geometry, model_parameters: LIONParameter = None):
        if geometry is None:
            raise ValueError("Geometry parameters required. ")

        super().__init__(model_parameters, geometry)

        # Create layers per iteration
        for i in range(self.model_parameters.n_iters):
            self.add_module(f"Unet_{i}", UNet(self.model_parameters.Unet_params))

        # Create pytorch compatible operators and send them to autograd
        self._make_operator()

        #need to fix the operators
        self.A = to_autograd(self.op, num_extra_dims = 0)
        self.AT = to_autograd(self.op.T, num_extra_dims = 0)


        # Define step size
        if self.model_parameters.step_size is None:
            # compute step size
            self.model_parameters.step_size = np.array(
                [1] * self.model_parameters.n_iters
            )

        elif not hasattr(self.model_parameters.step_size, "__len__"):
            self.model_parameters.step_size = np.array(
                self.model_parameters.step_size * self.model_parameters.n_iters
            )
        elif len(self.model_parameters.step_size) == self.model_parameters.n_iters:
            self.model_parameters.step_size = np.array(self.model_parameters.step_size)
        else:
            raise ValueError("Step size not understood")

        #  Are we learning the step? (with the above initialization)
        if self.model_parameters.learned_step:
            # Enforce positivity by making it 10^step
            if self.model_parameters.step_positive:
                self.step_size = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.ones(1)
                            * 10 ** np.log10(self.model_parameters.step_size[i])
                        )
                        for i in range(self.model_parameters.n_iters)
                    ]
                )
            # Negatives OK
            else:
                self.step_size = nn.ParameterList(
                    [
                     	nn.Parameter(torch.ones(1) * self.model_parameters.step_size[i])
                        for i in range(self.model_parameters.n_iters)
                    ]
                )
        else:
            self.step_size = (
                torch.ones(self.model_parameters.n_iters)
                * self.model_parameters.step_size
            )
        self.tv_step_size = nn.ParameterList(
            [nn.Parameter(torch.ones(1) * 1e-4) for i in range(self.model_parameters.n_iters)]
        )

    @staticmethod
    def default_parameters():
        param = LIONParameter()
        param.learned_step = True
        param.step_positive = False
        param.step_size = [1.1183, 1.3568, 1.4271, 0.0808]
        param.n_iters = 4
        param.Unet_params = UNet.default_parameters()
        param.mode = "ct"
        return param

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Martin Genzel, Ingo Guhring, Jan Macdonald, and Maximilian MÃ¤rz. ")
            print(
                '""Near-exact recovery for tomographic inverse problems via deep learning." '
            )
            print("\x1B[3m ICML 2022 \x1B[0m")
            print("(pp. 7368-7381). PMLR")

        elif cite_format == "bib":
            string = """
            @inproceedings{genzel2022near,
            title={Near-exact recovery for tomographic inverse problems via deep learning},
            author={Genzel, Martin and G{\"u}hring, Ingo and Macdonald, Jan and M{\"a}rz, Maximilian},
            booktitle={International Conference on Machine Learning},
            pages={7368--7381},
            year={2022},
            organization={PMLR}
            }"""
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )
    def gradientTVnorm2D(self, f):
        Gx = torch.diff(f, dim=2)
        Gy = torch.diff(f, dim=3)
        tvg = torch.zeros_like(f)

        Gx = torch.cat(
            (torch.zeros_like(Gx[:, :, :1, :]), Gx), dim=2
        )  # Pad Gx with zeros
        Gy = torch.cat(
            (torch.zeros_like(Gy[:, :, :, :1]), Gy), dim=3
        )  # Pad Gy with zeros

        nrm = torch.sqrt(Gx**2 + Gy**2 + 1e-6)

        tvg[:, :, :, :] = (
            tvg[:, :, :, :] + (Gx[:, :, :, :] + Gy[:, :, :, :]) / nrm[:, :, :, :]
        )
        tvg[:, :, :-1, :] = tvg[:, :, :-1, :] - Gx[:, :, 1:, :] / nrm[:, :, 1:, :]
        tvg[:, :, :, :-1] = tvg[:, :, :, :-1] - Gy[:, :, :, 1:] / nrm[:, :, :, 1:]

        return tvg

    def forward(self, sino):
        B, C, W, H = sino.shape

        img = sino.new_zeros(B, 1, *self.geometry.image_shape[1:])
        update = sino.new_zeros(B, 1, *self.geometry.image_shape[1:])
        del_S = sino.new_zeros(B, 1, *self.geometry.image_shape[1:])
        # Start from FDK
        for i in range(B):
            img[i] = fdk(self.op, sino[i])
        
        past_list = [[] for num in range(B)] #this allows each image to have its own list in a batch
        #alphas = [[] for num in range(B)] #should be reset each time forward is called?

        for i in range(self.model_parameters.n_iters): #number of iterations (5) for ONE EPOCH
            unet = getattr(self, f"Unet_{i}")
            img = unet(img)

            for j in range(img.shape[0]): #updating each image in batch
                del_S[j] = self.tv_step_size[i] * self.gradientTVnorm2D(img[j].unsqueeze(0))
                #compute gradient before we call the function so we dont have to recompute when s_k is zero
                g_k = self.AT(self.A(img[j]) - sino[j])

                #direction is the H approximation * gradient term
                direction = self.get_update(img[j], sino[j], past_list[j], g_k)

                #update is step size * direction
                update[j] = -self.step_size[i].squeeze() * direction
                
                # img[j] = img[j] - update[j] !!!we can not do this because we need that same img for back prop!!!
               
                #s_k is the step size of the last iteration
                s_k = update[j].clone()
                # ??? look at zach's notes
                #y_k is the difference of the gradients (grad f_k+1 - grad f_k), gives a better approx of gradient
                y_k = self.AT(self.A(s_k)).clone()
                #rho is the inverse curvature
                rho = 1 / (torch.dot(torch.flatten(y_k),torch.flatten(s_k))+ 1e-8)
                
                #add these values to the past list to get the next updates for the next iteration
                past_list[j].append([s_k,y_k,rho])
            
            img = img - update -del_S #this will keep our images safe and return them in a batch for the network

        return img
    
    def get_update(self, img, sino, past_list, g_k):
        alphas = [0] * len(past_list)
        #first iteration: just return the gradient as the direction
        if len(past_list)==0:
            return g_k
        
        #get the first hessian approximation
        q = g_k
        for i in range(len(past_list)-1,-1,-1):
            rho = past_list[i][2]
            alpha = rho * torch.dot(torch.flatten(past_list[i][0]), torch.flatten(q))
            alphas[i]=alpha
            q = q - (alpha * past_list[i][1])

        s_k = past_list[-1][0]
        y_k = past_list[-1][1]
        gamma = torch.dot(torch.flatten(s_k), torch.flatten(y_k)) / (torch.dot(torch.flatten(y_k), torch.flatten(y_k))+ 1e-8)
        r = gamma * q
        
        #r is the initial approximation H_0 * q
        # r = self.get_past_curv(past_list, alphas, g_k)
        
        for i in range(len(past_list)):
            beta = past_list[i][2] * torch.dot(torch.flatten(past_list[i][1]), torch.flatten(r))
            r = r + (alphas[i]-beta) * past_list[i][0]
        
        return r
 
