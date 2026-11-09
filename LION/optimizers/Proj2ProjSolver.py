# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Alvaro Exposito Martinez, Ander Biguri
# =============================================================================
from typing import Optional
from LION.models.LIONmodel import LIONmodel
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from LION.optimizers.LIONsolver import LIONsolver, warn_if_not_mse
from LION.utils.parameter import LIONParameter, OperatorParameters
from LION.utils.citation import citation


class Proj2ProjSolver(LIONsolver):
    """Proj2Proj.

    The training loss is the mean squared error: the masked self-supervised loss
    equals the supervised loss up to a constant only for the MSE. ``loss_fn`` is
    only used as the default validation and testing function.
    """

    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn,
        operator_params: OperatorParameters = None,
        solver_params: Optional[LIONParameter] = None,
        model_regularization=None,
        scheduler=None,
        scheduler_step: str = "epoch",
        verbose: int = 1,
        device: torch.device = None,
        seed: Optional[int] = 0,
        deterministic: bool = False,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            operator_params=operator_params,
            solver_params=solver_params,
            model_regularization=model_regularization,
            scheduler=scheduler,
            scheduler_step=scheduler_step,
            verbose=verbose,
            device=device,
            seed=seed,
            deterministic=deterministic,
        )
        warn_if_not_mse(loss_fn, "Proj2ProjSolver")

        self.op_params = self.operator_params
        self.projector, _ = self.op.autograd()
        self.global_step = 0

    def get_mask(self, shape, step):
        """The masks of a batch of measurements of ``shape`` (B, C, H, W): 0
        on one cell of every ``grid_size`` x ``grid_size`` block, the cell
        cycling with ``step`` (the samples seen so far), 1 elsewhere."""
        # shape: (B, C, H, W)
        mask = torch.ones(shape, device=self.device)
        grid = self.solver_params.grid_size

        for b in range(shape[0]):
            idx = (step + b) % (grid * grid)
            r = idx // grid
            c = idx % grid

            mask[b, :, r::grid, c::grid] = 0
        return mask

    def fill_mean(self, y, mask):
        """``y`` with its masked values (``mask`` 0) replaced by the mean of
        their four neighbours."""
        kernel = (
            torch.tensor(
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                dtype=torch.float32,
                device=self.device,
            )
            / 4.0
        )

        kernel = kernel.view(1, 1, 3, 3)
        local_mean_sino = F.conv2d(y, kernel, padding=1)

        filled_sinos = (y * mask) + (local_mean_sino * (1 - mask))
        return filled_sinos

    @staticmethod
    def default_parameters() -> LIONParameter:
        """``grid_size``: the masks' grid (4: one measurement in 16 masked);
        ``recon_fn``: how measurements are reconstructed ("default": the
        operator's default reconstruction)."""
        params = LIONParameter()
        params.grid_size = 4
        # How measurements are reconstructed, recon_fn(y, op). "default": the
        # operator's default reconstruction (FDK for CT), replaced by that
        # function when the solver is made.
        params.recon_fn = "default"
        return params

    def mini_batch_step(self, y, targets):
        """The loss: the masked measurements filled in, reconstructed, the
        model's output projected back, against the measurements on the mask
        (mean squared error; ``targets`` aren't used: self-supervised)."""
        mask = self.get_mask(y.shape, self.global_step)
        self.global_step += y.shape[0]
        input_sino = self.fill_mean(y, mask)

        input_recon = self.reconstruct_measurements(input_sino, self.op)
        output_recon = self.model(input_recon)
        output_sino = self.projector(output_recon)

        output_sino_mask = output_sino * (1 - mask)
        target_sino = y * (1 - mask)

        batch_loss = ((output_sino_mask - target_sino) ** 2).mean()
        return self._regularized(batch_loss, input_recon, output_recon)

    def _forward(self, data):
        # validation and testing evaluate the reconstruction
        return self.reconstruct(data)

    def reconstruct(self, y):
        """The model's reconstructions of the measurements ``y``: of the
        reconstructions ``recon_fn`` makes of them, as in training."""
        input_recon = self.reconstruct_measurements(y, self.op)
        output_recon = self.model(input_recon)
        return output_recon

    @staticmethod
    def cite(cite_format="bib"):
        """Where Proj2Proj comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{unal2024proj2proj,
            title={Proj2Proj: self-supervised low-dose CT reconstruction},
            author={Unal, Mehmet Ozan and Ertas, Metin and Yildirim, Isa},
            journal={PeerJ Computer Science},
            volume={10},
            pages={e1849},
            year={2024},
            doi={10.7717/peerj-cs.1849}
            }
            """,
            mla="""
            Unal, Mehmet Ozan, Metin Ertas, and Isa Yildirim. "Proj2Proj: self-supervised low-dose CT reconstruction." PeerJ Computer Science 10 (2024): e1849.
            """,
        )
