import torch
import torch.nn as nn
import numpy as np

from LION.classical_algorithms.fdk import fdk
from LION.optimizers.LIONloss import LIONloss2


class WGAN_gradient_penalty_loss(LIONloss2):
    def __init__(self, model, op, mu=10.0 * 1e-2):
        self.mu = mu
        self.op = op
        super().__init__(model)

    def forward(self, sino, gt):
        """Calculates the gradient penalty loss for WGAN GP"""
        bad_recon = fdk(sino, self.op)
        alpha = torch.Tensor(np.random.random((gt.shape[0], 1, 1, 1))).type_as(gt)
        interpolates = (alpha * gt + ((1 - alpha) * bad_recon)).requires_grad_(True)
        net_interpolates = self.model(interpolates)
        fake = torch.Tensor(net_interpolates.shape).fill_(1.0).type_as(gt).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=net_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.shape[0], -1)
        # print(model(real_samples).mean()-model(fake_samples).mean(),self.mu*(((gradients.norm(2, dim=1) - 1)) ** 2).mean())
        loss = (
            self.model(gt).mean()
            - self.model(bad_recon).mean()
            + self.mu * (((gradients.norm(2, dim=1) - 1)) ** 2).mean()
        )
        return loss
