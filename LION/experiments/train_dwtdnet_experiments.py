from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from LION.experiments.config.training_config import (
    AdamParameter,
    CosineAnnealingLRParameter,
)
from LION.utils.parameter import LIONParameter


@dataclass(kw_only=True)
class DWTDNetTrainParameter(LIONParameter):
    """Configuration for DWTDNet training.

    Attributes:
        sigma: A float for single noise level, or pair of floats for noise level range.
    """

    optim: AdamParameter
    scheduler: CosineAnnealingLRParameter | None
    epochs: int = 10
    start_epoch: int = 1
    sigma: float | list[float] = 0.05
    model_size: Literal["tiny", "small", "regular"] | None = None

    @staticmethod
    def dwtdnet_tiny() -> DWTDNetTrainParameter:
        """Return default training parameters for DWTDNet."""
        return DWTDNetTrainParameter(
            optim=AdamParameter(
                name="Adam",
                learning_rate=1e-4,
                betas=[0.9, 0.99],
                eps=1e-8,
                weight_decay=0.0,
            ),
            scheduler=CosineAnnealingLRParameter(T_max=None, eta_min=1e-6),
            epochs=10,
            start_epoch=1,
            sigma=0.05,
            model_size="tiny",
        )

    @staticmethod
    def gs_dwtdnet_tiny() -> DWTDNetTrainParameter:
        return DWTDNetTrainParameter(
            optim=AdamParameter(
                name="Adam",
                # learning_rate=1e-4,
                # learning_rate=1e-5,
                learning_rate=3e-6,
                # learning_rate=2e-6,
                # learning_rate=1e-6,
                betas=[0.9, 0.99],
                eps=1e-8,
                weight_decay=0.0,
            ),
            scheduler=CosineAnnealingLRParameter(T_max=25, eta_min=1e-6),
            # scheduler=None,
            # epochs=10,
            epochs=1000,
            start_epoch=1,
            sigma=0.05,
            model_size="tiny",
        )

    @staticmethod
    def gs_dwtdnet_small() -> DWTDNetTrainParameter:
        return DWTDNetTrainParameter(
            optim=AdamParameter(
                name="Adam",
                # learning_rate=1e-4,
                # learning_rate=1e-5,
                # learning_rate=3e-6,
                # learning_rate=3e-6,
                # learning_rate=2e-6,
                learning_rate=5e-7,
                betas=[0.9, 0.99],
                eps=1e-8,
                weight_decay=0.0,
            ),
            scheduler=CosineAnnealingLRParameter(
                T_max=None,
                #  eta_min=1e-6,
                # eta_min=2e-7,
                eta_min=1e-7,
            ),
            # scheduler=None,
            # epochs=10,
            # epochs=1000,
            epochs=25,
            start_epoch=1,
            sigma=0.05,
            model_size="small",
        )

    @staticmethod
    def gs_dwtdnet_regular() -> DWTDNetTrainParameter:
        return DWTDNetTrainParameter(
            optim=AdamParameter(
                name="Adam",
                # learning_rate=1e-4,
                # learning_rate=1e-5,
                # learning_rate=3e-6,
                learning_rate=3e-6,
                # learning_rate=2e-6,
                # learning_rate=1e-6,
                betas=[0.9, 0.99],
                eps=1e-8,
                weight_decay=0.0,
            ),
            scheduler=CosineAnnealingLRParameter(T_max=None, eta_min=1e-6),
            # scheduler=None,
            # epochs=10,
            epochs=100,
            start_epoch=1,
            sigma=0.05,
            model_size="regular",
        )

    @staticmethod
    def gs_drunet() -> DWTDNetTrainParameter:
        return DWTDNetTrainParameter(
            optim=AdamParameter(
                name="Adam",
                # learning_rate=1e-4,
                # learning_rate=1e-5,
                learning_rate=1e-3,
                betas=[0.9, 0.99],
                eps=1e-8,
                weight_decay=0.0,
            ),
            scheduler=CosineAnnealingLRParameter(T_max=None, eta_min=1e-6),
            # scheduler=None,
            epochs=10,
            start_epoch=1,
            sigma=0.05,
        )
