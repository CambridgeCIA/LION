"""Run dataset self-supervised SSDU SAPM training for PCM reconstruction."""

from LION.pcm.pcm_config import (
    SAPMSelfSupervisedPCMConfig,
    load_pcm_sapm_self_supervised_config,
)
from LION.pcm.pcm_sapm import forward_func_sapm_pcm, load_sapm_pcm_model
from LION.pcm.SelfSupervisedPCMTrainer import SelfSupervisedPCMTrainer
from LION.pcm.SupervisedPCMTrainer import run_pcm_dataset_train_cli
from LION.train.sapm import make_param_groups_sapm_common

if __name__ == "__main__":
    run_pcm_dataset_train_cli(
        config_cls=SAPMSelfSupervisedPCMConfig,
        load_config=load_pcm_sapm_self_supervised_config,
        make_param_groups=make_param_groups_sapm_common,
        load_model=load_sapm_pcm_model,
        forward_func=forward_func_sapm_pcm,
        trainer_cls=SelfSupervisedPCMTrainer,
    )
