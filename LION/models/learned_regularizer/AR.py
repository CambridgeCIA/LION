# This file is part of LION library
# License : BSD-3
#
# Author  : Zakhar Shumaylov, Subhadip Mukherjee
# Modifications: Ander Biguri, Zakhar Shumaylov
# =============================================================================


import torch
import torch.nn as nn
from LION.models import LIONmodel
import LION.CTtools.ct_geometry as ct
from LION.utils.parameter import LIONParameter
import torch.nn.utils.parametrize as P
from ts_algorithms import fdk
from tqdm import tqdm
import numpy as np
import wandb
from LION.utils.math import power_method

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
# Just a temporary SSIM that takes troch tensors (will be added to LION at some point)
def my_ssim(x: torch.tensor, y: torch.tensor):
    if x.shape[0]==1:
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        return ssim(x, y, data_range=x.max() - x.min())
    else: 
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        vals=[]
        for i in range(x.shape[0]):
            vals.append(ssim(x[i], y[i], data_range=x[i].max() - x[i].min()))
        return np.array(vals)

def my_psnr(x: torch.tensor, y: torch.tensor):
    if x.shape[0]==1:
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        return psnr(x, y, data_range=x.max() - x.min())
    else: 
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        vals=[]
        for i in range(x.shape[0]):
            vals.append(psnr(x[i], y[i], data_range=x[i].max() - x[i].min()))
        return np.array(vals)


class network(nn.Module):
    def __init__(self,n_chan=1):
        super(network, self).__init__()

        self.leaky_relu = nn.LeakyReLU()
        self.convnet = nn.Sequential(
            nn.Conv2d(n_chan, 16, kernel_size=(5, 5),padding=2),
            self.leaky_relu,
            nn.Conv2d(16, 32, kernel_size=(5, 5),padding=2),
            self.leaky_relu,
            nn.Conv2d(32, 32, kernel_size=(5, 5),padding=2,stride=2),
            self.leaky_relu,
            nn.Conv2d(32, 64, kernel_size=(5, 5),padding=2,stride=2),
            self.leaky_relu,
            nn.Conv2d(64, 64, kernel_size=(5, 5),padding=2,stride=2),
            self.leaky_relu,
            nn.Conv2d(64, 128, kernel_size=(5, 5),padding=2,stride=2),
            self.leaky_relu
        )
        size=1024
        self.fc = nn.Sequential(
            nn.Linear(128*(size//2**4)**2, 256),
            self.leaky_relu,
            nn.Linear(256, 1)
        )

    def forward(self, image):
        output = self.convnet(image)
        output = output.view(image.size(0), -1)
        output = self.fc(output)
        return output

class AR(LIONmodel.LIONmodel):
    def __init__(self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None):

        super().__init__(model_parameters,geometry_parameters)
        self._make_operator()
        
        self.network = network()
        # First Conv
        self.estimate_lambda()
        self.step_amounts = torch.tensor([150.0])
        self.op_norm = power_method(self.op)
        self.model_parameters.step_size = 0.2/(self.op_norm)**2

    def forward(self, x):
        # x = fdk(self.op, x)
        x = self.normalise(x)
        # print(self.pool(z).mean(),self.L2(z).mean())
        return self.network(x).reshape(-1,1)# + self.L2(z)
        
    def estimate_lambda(self,dataset=None):
        self.lamb=1.0
        if dataset is None: self.lamb=1.0
        else: 
            residual = 0.0
            for index, (data, target) in enumerate(dataset):
                residual += torch.norm(self.AT(self.A(target) - data),dim=(2,3)).mean()
                # residual += torch.sqrt(((self.AT(self.A(target) - data))**2).sum())
            self.lamb = residual.mean()/len(dataset)
        print('Estimated lambda: ' + str(self.lamb))
    
       
       
    
    # def output(self, x):
        # return self.AT(x)
    
    def var_energy(self,x,y):
        # return torch.norm(x) + 0.5*(torch.norm(self.A(x)-y,dim=(2,3))**2).sum()#self.lamb * self.forward(x).sum()
        return 0.5*((self.A(x)-y)**2).sum() + self.lamb * self.forward(x).sum()
      ### What is the difference between .sum() and .mean()??? idfk but PSNR is lower when I do .sum
      
    def output(self,y,truth=None):
        # wandb.log({'Eearly_stopping_steps': self.step_amounts.mean().item(), 'Eearly_stopping_steps_std': self.step_amounts.std().item()})
        x0=[]
        device = torch.cuda.current_device()
        for i in range(y.shape[0]):
            x0.append(fdk(self.op, y[i]))
        x = torch.stack(x0)
        # print(x.shape)
        # print(x.min(),x.max())
        # print(my_psnr(truth.detach().to(device),x.detach()).mean(),my_ssim(truth.detach().to(device),x.detach()).mean())
        x=torch.nn.Parameter(x)#.requires_grad_(True)

        optimizer = torch.optim.SGD([x], lr=self.model_parameters.step_size, momentum=0.5)#self.model_parameters.momentum)
        lr = self.model_parameters.step_size
        prevpsn=0
        curpsn=0
        for j in range(self.model_parameters.no_steps):
            # print(x.min(),x.max())
            # data_misfit=self.A(x)-y
            # data_misfit_grad = self.AT(data_misfit)
            
            optimizer.zero_grad()
            # reg_func=self.lamb * self.forward(x).mean()
            # reg_func.backward()
            # print(x.requires_grad, reg_func.requires_grad)
            energy = self.var_energy(x,y)
            energy.backward()
            while(self.var_energy(x-x.grad*lr,y) > energy - 0.5*lr*(x.grad.norm(dim=(2,3))**2).mean()):
                lr=self.model_parameters.beta_rate*lr
                # print('decay')
            for g in optimizer.param_groups:
                g['lr'] = lr
            # x.grad+=data_misfit_grad
            if(truth is not None):
                loss = torch.nn.MSELoss()(x.detach(),truth.detach().to(device))
                psnr_val = my_psnr(truth.detach().to(device),x.detach()).mean()
                ssim_val = my_ssim(truth.detach().to(device),x.detach()).mean()
                # wandb.log({'MSE Loss': loss.item(),'SSIM':ssim_val,'PSNR':psnr_val})
                # wandb.log({'MSE Loss'+str(self.model_parameters.step_size): loss.item(),'SSIM'+str(self.model_parameters.step_size):ssim_val,'PSNR'+str(self.model_parameters.step_size):psnr_val})
                print(f"{j}: SSIM: {my_ssim(truth.to(device).detach(),x.detach())}, PSNR: {my_psnr(truth.to(device).detach(),x.detach())}, Energy: {energy.detach().item()}")
        
            #     if(self.args.outp):
            #         print(j)
                prevpsn=curpsn
                curpsn=psnr_val
                # if(curpsn<prevpsn):
                #     self.step_amounts = torch.cat((self.step_amounts,torch.tensor([j*1.0])))
                #     return x.detach()
            elif(j > self.step_amounts.mean().item()): 
                # print('only for testing')
                x.clamp(min=0.0)
                return x.detach()
            elif(lr * self.op_norm**2 < 1e-3):
                x.clamp(min=0.0)
                return x.detach()
            optimizer.step()
            x.clamp(min=0.0)
        return x.detach()
    
    def normalise(self,x):
        return (x - self.model_parameters.xmin) / (self.model_parameters.xmax - self.model_parameters.xmin)
    def unnormalise(self,x):
        return x * (self.model_parameters.xmax - self.model_parameters.xmin) + self.model_parameters.xmin

    @staticmethod
    def default_parameters():
        param = LIONParameter()
        param.channels = 16
        param.kernel_size = 5
        param.stride = 1
        param.relu_type = "LeakyReLU"
        param.layers = 5
        param.early_stopping = False
        param.no_steps = 150
        param.step_size = 1e-6
        param.momentum = 0.5
        param.beta_rate=0.95
        param.xmin = 0.
        param.xmax = 1.
        return param

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Mukherjee, Subhadip, et al.")
            print('"Data-Driven Convex Regularizers for Inverse Problems."')
            print("ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024")
            print("arXiv:2008.02839 (2020).")
        elif cite_format == "bib":
            string = """
            @inproceedings{mukherjee2024data,
            title={Data-Driven Convex Regularizers for Inverse Problems},
            author={Mukherjee, S and Dittmer, S and Shumaylov, Z and Lunz, S and {\"O}ktem, O and Sch{\"o}nlieb, C-B},
            booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
            pages={13386--13390},
            year={2024},
            organization={IEEE}
            }
            """
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )
