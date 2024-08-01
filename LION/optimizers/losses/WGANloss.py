import numpy as np
import torch
import torch.nn as nn
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import ModelInputType


class WGANloss(nn.Module):
    def __init__(self, mu=10e-2) -> None:
        super().__init__()
        self.mu = mu

    def forward(self, sino, gt):
        assert (
            self.model is not None
        ), "Model required but not set. Please call set_model"
        if self.model.get_input_type() == ModelInputType.IMAGE:
            bad_recon = fdk(sino, self.model.op)

            if hasattr(self, "do_normalize") and self.do_normalize is not None:
                assert (
                    hasattr(self, "normalize") and self.normalize is not None
                ), "do_normalize True but no normalization function no set"
                model_in_data = self.normalize(bad_recon)

        else:
            model_in_data = sino
            bad_recon = fdk(sino, self.model.op)

        alpha = torch.Tensor(np.random.random((gt.shape[0], 1, 1, 1))).type_as(gt)
        interpolates = (alpha * gt + ((1 - alpha) * bad_recon)).requires_grad_(True)

        net_interpolates = self.model(interpolates)

        fake = (
            torch.Tensor(net_interpolates.shape)
            .fill_(1.0)
            .type_as(gt)
            .requires_grad_(False)
        )
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
            - self.model(model_in_data).mean()
            + self.mu * (((gradients.norm(2, dim=1) - 1)) ** 2).mean()
        )
        return loss
