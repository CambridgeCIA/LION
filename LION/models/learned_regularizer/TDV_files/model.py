import torch
import torch.utils.checkpoint as cp

import numpy as np

from .ddr import TDV


class Dataterm(torch.nn.Module):
    """
    Basic dataterm function
    """

    def __init__(self, config):
        super(Dataterm, self).__init__()

    def forward(self, x, *args):
        raise NotImplementedError

    def energy(self):
        raise NotImplementedError

    def prox(self, x, *args):
        raise NotImplementedError

    def grad(self, x, *args):
        raise NotImplementedError


class L2DenoiseDataterm_noise(Dataterm):
    def __init__(self, config):
        super(L2DenoiseDataterm, self).__init__(config)

    def energy(self, x, z):
        return 0.5*(x-z)**2

    def prox(self, x, z, tau):
        return (x + tau * z) / (1 + tau) 

    def grad(self, x, z):
        return x-z

class L2DenoiseDataterm(Dataterm):
    def __init__(self, config, A, AT):
        super(L2DenoiseDataterm, self).__init__(config)
        self.A=A
        self.AT=AT

    def energy(self, x, z):
        return (0.5*(self.A(x)-z)**2).mean()

    def prox(self, x, z, tau):
        return 0#(x + tau * z) / (1 + tau) 

    def grad(self, x, z):
        return self.AT(self.A(x)-z)
    
class VNet(torch.nn.Module):
    """
    Variational Network
    """

    def __init__(self, config,A,AT,op_norm, efficient=False):
        super(VNet, self).__init__()

        self.efficient = efficient
        
        self.S = config['S']
        self.op_norm = op_norm
        # setup the stopping time
        if config['T_mode'] == 'fixed':
            self.register_buffer('T', torch.tensor(config['T']['init']))
        elif config['T_mode'] == 'learned':
            self.T = torch.nn.Parameter(torch.Tensor(1))
            self.reset_scalar(self.T, **config["T"])
            self.T.L_init = 1e-6
        else:
            raise RuntimeError('T_mode unknown!')

        if config['lambda_mode'] == 'fixed':
            self.register_buffer('lmbda', torch.tensor(config['lambda']['init']))
        elif config['lambda_mode'] == 'learned':
            self.lmbda = torch.nn.Parameter(torch.Tensor(1))
            self.reset_scalar(self.lmbda, **config["lambda"])
            self.lmbda.L_init = 1e-6/op_norm**2
        else:
            raise RuntimeError('lambda_mode unknown!')

        # setup the regularization
        R_types = {
            'tdv': TDV,
        }
        self.R = R_types[config['R']['type']](config['R']['config'])

        # setup the dataterm
        self.use_prox = config['D']['config']['use_prox']
        D_types = {
            'denoise': L2DenoiseDataterm,
        }
        self.D = D_types[config['D']['type']](config['D']['config'],A,AT)
        print(self.lmbda)

    def reset_scalar(self, scalar, init=1., min=0, max=1000):
        scalar.data = torch.tensor(init, dtype=scalar.dtype)
        # add a positivity constraint
        scalar.proj = lambda: scalar.data.clamp_(min, max)

    def forward(self, x, z, get_grad_R=False):

        x_all = x.new_empty((self.S+1,*x.shape))
        x_all[0] = x
        if get_grad_R:
            grad_R_all = x.new_empty((self.S, *x.shape))

        # define the step size
        tau = self.T / self.S
        for s in range(1,self.S+1):
            # compute a single step
            if self.efficient and x.requires_grad:
                grad_R = cp.checkpoint(self.R.grad, x)
            else:
                grad_R = self.R.grad(x)
            if self.use_prox:
                x = self.D.prox(x - tau * grad_R, z, self.lmbda / self.S)
            else:
                # print()
                # print(self.D.energy(x,z))
                # print(grad_R.isnan().any(),self.D.grad(x, z).isnan().any())
                # print(grad_R.norm().item(),self.D.grad(x, z).norm().item())
                x = x - tau * grad_R - (self.lmbda/self.S) * self.D.grad(x, z) /(self.op_norm**2)
            if get_grad_R:
                grad_R_all[s-1] = grad_R
            x_all[s] = x
        
        if get_grad_R:
            return x_all, grad_R_all
        else:
            return x_all

    def set_end(self, s):
        assert 0 < s
        self.S = s

    def extra_repr(self):
        s = "S={S}"
        return s.format(**self.__dict__)
