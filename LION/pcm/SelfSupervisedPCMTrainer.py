"""Dataset self-supervised SSDU SAPM training for PCM reconstruction."""

from __future__ import annotations

import numpy as np
import torch

from LION.pcm.pcm_config import SAPMSelfSupervisedPCMConfig
from LION.pcm.pcm_ssdu import PCMSSDUSplit, make_pcm_ssdu_split
from LION.pcm.SupervisedPCMTrainer import SupervisedPCMTrainer


class SelfSupervisedPCMTrainer(SupervisedPCMTrainer):
    """Dataset self-supervised SSDU trainer for unrolled TV/TGV PCM reconstruction."""

    def make_ssdu_data(
        self,
        target_image: torch.Tensor,
        *,
        key: int,
        rng: np.random.Generator,
        device: torch.device | str,
        p_target: float,
    ) -> PCMSSDUSplit:
        sampling_ratio, coarse_j = self.cases[key]
        acquisition = self.make_acquisition(
            target_image, key=key, rng=rng, device=device
        )
        return make_pcm_ssdu_split(
            config=self.config,
            source_indices=acquisition.sampled_indices,
            source_measurement=acquisition.measurement,
            p_target=p_target,
            rng=rng,
            key=key,
            sampling_ratio=sampling_ratio,
            coarse_j=coarse_j,
            device=device,
        )

    def make_train_forward_data(
        self,
        target_image: torch.Tensor,
        *,
        key: int,
        rng: np.random.Generator,
        device: torch.device | str,
    ) -> PCMSSDUSplit:
        config: SAPMSelfSupervisedPCMConfig = self.config
        return self.make_ssdu_data(
            target_image,
            key=key,
            rng=rng,
            device=device,
            p_target=config.ssdu.p_target_train,
        )

    def make_val_forward_data(
        self,
        target_image: torch.Tensor,
        *,
        key: int,
        rng: np.random.Generator,
        device: torch.device | str,
    ) -> PCMSSDUSplit:
        config: SAPMSelfSupervisedPCMConfig = self.config
        return self.make_ssdu_data(
            target_image,
            key=key,
            rng=rng,
            device=device,
            p_target=config.ssdu.p_target_val,
        )

    def forward_step(self, data: PCMSSDUSplit, *, skip_grad_checkpoint: bool):
        return self.forward_runner(
            model=self.model,
            data=data.forward_data,
            config=self.config,
            skip_grad_checkpoint=skip_grad_checkpoint,
        )

    def forward_backward_train(
        self, data: PCMSSDUSplit, *, target_image: torch.Tensor
    ) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        result = self.forward_step(data, skip_grad_checkpoint=False)
        loss = torch.nn.functional.mse_loss(
            data.target_op(result.reconstruction), data.target
        )
        return self.optimizer_step_from_loss(loss)

    @torch.no_grad()
    def forward_val(
        self, data: PCMSSDUSplit, *, target_image: torch.Tensor
    ) -> tuple[float, torch.Tensor]:
        result = self.forward_step(data, skip_grad_checkpoint=True)
        loss = torch.nn.functional.mse_loss(
            data.target_op(result.reconstruction), data.target
        )
        return loss.item(), result.reconstruction.detach().cpu()
