from typing import Optional
from LION.operators.ct.ct_geometry import Geometry
from LION.models.LIONmodel import LIONmodel
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from LION.optimizers.LIONsolver import LIONsolver, warn_if_not_mse
from LION.utils.parameter import LIONParameter
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
        solver_params: Optional[LIONParameter] = None,
        operator_params: Geometry = None,
        verbose: bool = True,
        device: torch.device = None,
        seed: Optional[int] = 0,
        deterministic: bool = False,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            operator_params,
            verbose,
            device,
            solver_params=solver_params,
            seed=seed,
            deterministic=deterministic,
        )
        warn_if_not_mse(loss_fn, "Proj2ProjSolver")

        self.op_params = self.operator_params
        self.projector, _ = self.op.autograd()
        self.global_step = 0

    def get_mask(self, shape, step):
        # shape: (B, C, H, W)
        mask = torch.ones(shape, device=self.device)
        grid = self.solver_params.grid_size

        for b in range(shape[0]):
            idx = (step + b) % (grid * grid)
            r = idx // grid
            c = idx % grid

            mask[b, :, r::grid, c::grid] = 0
        return mask

    def fill_mean(self, sinos, mask):
        kernel = (
            torch.tensor(
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                dtype=torch.float32,
                device=self.device,
            )
            / 4.0
        )

        kernel = kernel.view(1, 1, 3, 3)
        local_mean_sino = F.conv2d(sinos, kernel, padding=1)

        filled_sinos = (sinos * mask) + (local_mean_sino * (1 - mask))
        return filled_sinos

    @staticmethod
    def default_parameters() -> LIONParameter:
        params = LIONParameter()
        params.grid_size = 4
        # How measurements are reconstructed, recon_fn(y, op). "default": the
        # operator's default reconstruction (FDK for CT), replaced by that
        # function when the solver is made.
        params.recon_fn = "default"
        return params

    def mini_batch_step(self, sinos, targets):
        mask = self.get_mask(sinos.shape, self.global_step)
        self.global_step += sinos.shape[0]
        input_sino = self.fill_mean(sinos, mask)

        input_recon = self.reconstruct_measurements(input_sino, self.op)
        output_recon = self.model(input_recon)
        output_sino = self.projector(output_recon)

        output_sino_mask = output_sino * (1 - mask)
        target_sino = sinos * (1 - mask)

        batch_loss = ((output_sino_mask - target_sino) ** 2).mean()
        return batch_loss

    def _forward(self, data):
        # validation and testing evaluate the reconstruction
        return self.reconstruct(data)

    def reconstruct(self, sinos):
        input_recon = self.reconstruct_measurements(sinos, self.op)
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
