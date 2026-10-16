"""Run supervised SAPM training for Cartesian MRI reconstruction."""

from __future__ import annotations

from zsu.models.model_loader import load_sapm_mri_model
from zsu.mri.mri_config import SAPMSupervisedMRIConfig, load_mri_sapm_supervised_config
from zsu.mri.mri_sapm import forward_func_sapm_mri
from zsu.mri.mri_supervised_train import run_mri_supervised_train_cli
from zsu.scripts_utils.sapm import make_param_groups_sapm_common

if __name__ == "__main__":
    run_mri_supervised_train_cli(
        config_cls=SAPMSupervisedMRIConfig,
        load_config=load_mri_sapm_supervised_config,
        make_param_groups=make_param_groups_sapm_common,
        load_model=load_sapm_mri_model,
        forward_func=forward_func_sapm_mri,
    )
