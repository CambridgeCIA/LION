"""Run zero-shot SAPM training for PCM reconstruction."""

from LION.pcm.pcm_config import SAPMZeroShotPCMConfig, load_pcm_sapm_zero_shot_config
from LION.pcm.pcm_sapm import forward_func_sapm_pcm, load_sapm_pcm_model
from LION.pcm.ZeroShotPCMTrainer import run_pcm_zero_shot_train_cli
from LION.train.sapm import make_param_groups_sapm_common

if __name__ == "__main__":
    run_pcm_zero_shot_train_cli(
        config_cls=SAPMZeroShotPCMConfig,
        load_config=load_pcm_sapm_zero_shot_config,
        make_param_groups=make_param_groups_sapm_common,
        load_model=load_sapm_pcm_model,
        forward_func=forward_func_sapm_pcm,
    )
