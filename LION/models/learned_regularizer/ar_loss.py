import torch
import torch.nn as nn
import numpy as np



class WGAN_gradient_penalty_loss(nn.Module):
    def __init__(self, mu=10.0 * 1e-2):
        self.mu=mu
        super().__init__()


    def forward(self,model, data_marginal_noisy,data_marginal_real):
        """Calculates the gradient penalty loss for WGAN GP"""
        real_samples=data_marginal_real
        fake_samples=data_marginal_noisy
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).type_as(real_samples)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        net_interpolates = model(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).type_as(real_samples).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=net_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        # print(model(real_samples).mean()-model(fake_samples).mean(),self.mu*(((gradients.norm(2, dim=1) - 1)) ** 2).mean())
        loss = model(real_samples).mean()-model(fake_samples).mean()+self.mu*(((gradients.norm(2, dim=1) - 1)) ** 2).mean()
        return loss