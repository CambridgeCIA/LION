from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn
from zsu.gradops.gradops_torch import GradOpsTorch

TvLambdaReg = float | torch.Tensor


TgvLambdaReg = dict[Literal["lambda1_v", "lambda0_w"], float | torch.Tensor]


TvPdhgStateDict = dict[Literal["x", "v", "p", "r", "x_bar"], torch.Tensor]


TgvPdhgStateDict = dict[
    Literal["x", "v", "p", "q", "r", "x_bar", "v_bar"], torch.Tensor
]


class PdhgTorch(nn.Module):
    def __init__(self, device: str | torch.device):
        super().__init__()  # Ensure proper initialisation
        self.device = device
        self.grad_ops = GradOpsTorch()

    def run_pdhg(
        self,
        iterator,  # determines how long to run algorithm for
        A: Callable,
        AH: Callable,
        lambda_reg: TvLambdaReg | TgvLambdaReg,
        z: torch.Tensor,  # measured data
        state: torch.Tensor | TvPdhgStateDict | TgvPdhgStateDict,
        sigma: float | torch.Tensor,
        tau: float | torch.Tensor,
        theta: float | torch.Tensor,
        return_dict: bool = False,
        device: str | torch.device | None = None,
    ) -> torch.Tensor | TgvPdhgStateDict:
        raise NotImplementedError()

    def forward(
        self,
        num_iters: int,  # number of iterations
        A: Callable,
        AH: Callable,
        lambda_reg: TvLambdaReg | TgvLambdaReg,
        z: torch.Tensor,  # measured data
        state: torch.Tensor | TvPdhgStateDict | TgvPdhgStateDict,
        sigma: float | torch.Tensor,
        tau: float | torch.Tensor,
        theta: float | torch.Tensor,
        return_dict: bool = False,
        tqdm_progress_bar: Callable | None = None,
        device: str | torch.device | None = None,
    ) -> torch.Tensor | TvPdhgStateDict | TgvPdhgStateDict:

        if tqdm_progress_bar is None:
            iterator = range(num_iters)
        else:
            iterator = tqdm_progress_bar(range(num_iters))

        state = self.run_pdhg(
            iterator=iterator,
            A=A,
            AH=AH,
            lambda_reg=lambda_reg,
            z=z,
            state=state,
            sigma=sigma,
            tau=tau,
            theta=theta,
            return_dict=return_dict,
            device=device,
        )

        return state

    def prepare_lambda(
        self, lambda_reg: float | torch.Tensor, domain: Literal["V", "W"]
    ) -> torch.Tensor:
        """Prepare the regularization parameter lambda for the PDHG algorithm.

        Args:
            lambda_reg : If tensor then assume lambda_reg in U (same shape as image)
            domain : Either "V" or "W" to indicate which domain the
                    regularisation should be applied to
                    (V for gradient, W for symmetrised gradient)

        >>> tgv_pdhg_solver = TgvPdhgTorch(sigma=8.7, tau=0.01, device="cpu")
        >>> x = torch.zeros(10, 10, 2).cpu()
        >>> lambda_reg = 0.0  # Should handle zero divided by zero
        >>> proj = tgv_pdhg_solver.P_alpha(x, lambda_reg, domain="V")
        >>> proj.shape
        torch.Size([10, 10, 2])
        >>> torch.isnan(proj).any().cpu().item()
        >>> # Should handle pixel value zero divided by lambda_reg value zero.
        >>> # Zero lambda_reg means no regularisation,
        >>> # so pixel zero remains the same.
        False
        >>> type(lambda_reg)
        <class 'float'>
        >>> lambda_reg = torch.rand(10, 10).cpu()
        >>> proj = tgv_pdhg_solver.P_alpha(x, lambda_reg, domain="V")
        >>> proj.shape
        torch.Size([10, 10, 2])
        >>> torch.isnan(proj).any().cpu().item()
        False
        >>> lambda_reg.shape # lambda_reg should be unchanged
        torch.Size([10, 10])
        """
        if not isinstance(lambda_reg, torch.Tensor):
            assert isinstance(
                lambda_reg, float
            ), f"Expected float, got {type(lambda_reg)}"
            # Make a 2D tensor
            lambda_reg = torch.full(
                size=(1, 1, 1), fill_value=lambda_reg, device=self.device
            )
        assert lambda_reg.ndim == 3, (
            "Expected 3 dimensions (batch_size, Nx, Ny). "
            + f"Got {lambda_reg.ndim}: {lambda_reg.shape}"
        )
        lambda_reg = lambda_reg.to(self.device)
        if domain == "V":
            lambda_reg = lambda_reg.unsqueeze(-1)
        elif domain == "W":
            lambda_reg = lambda_reg.unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"Unknown domain. Expected 'V' or 'W', got {domain}")
        # Add another dimension for when viewing as real
        lambda_reg = lambda_reg.unsqueeze(-1)
        return lambda_reg

    def clip_lambda(self, x: torch.Tensor, lambda_reg: torch.Tensor) -> torch.Tensor:
        if x.is_complex():
            x = torch.view_as_real(x)
            clipped_x = torch.clamp(x, -lambda_reg, lambda_reg)
            return torch.view_as_complex(clipped_x)
        return torch.clamp(x, -lambda_reg, lambda_reg)

    def prepare_variable(
        self,
        x: torch.Tensor | None,
        shape: torch.Size | tuple[int, ...],
        default_value: float | torch.Tensor = 0.0,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        if x is None:
            x = torch.zeros(shape, dtype=dtype).to(self.device) + default_value
        if device is None:
            device = self.device
        return x.to(device)

    def get_variables(
        self, state: torch.Tensor | TvPdhgStateDict | TgvPdhgStateDict, var_names: tuple
    ) -> list[torch.Tensor | None]:
        variables = []
        for var_name in var_names:
            if isinstance(state, dict):
                variables.append(state.get(var_name, None))
            else:
                variables.append(None)
        return variables


class TvPdhgTorch(PdhgTorch):
    def __init__(self, device: str | torch.device):
        super().__init__(device=device)

    def run_pdhg(
        self,
        iterator,  # determines how long to run algorithm for
        A: Callable,
        AH: Callable,
        lambda_reg: TvLambdaReg,
        z: torch.Tensor,  # measured data
        state: torch.Tensor | TvPdhgStateDict,
        sigma: float | torch.Tensor,
        tau: float | torch.Tensor,
        theta: float | torch.Tensor,
        return_dict: bool = False,
        device: str | torch.device | None = None,
    ):
        """Adapted Algorithm 2 in 'Learning Regularization Parameter-Maps for Variational Image Reconstruction
        using Deep Neural Networks and Algorithm Unrolling', 2023, by Andreas Kofler et al.
        """
        if isinstance(state, dict):
            x = state["x"]  # initial image estimate
        elif isinstance(state, torch.Tensor):
            x = state  # initial image estimate
        else:
            raise ValueError(
                "Unknown input type. "
                + f"Expected dict or torch.Tensor, got {type(state)}"
            )

        if device is None:
            device = self.device
        assert device is not None, "Device must be specified"
        x = x.to(device)
        z = z.to(device)
        v, p, r, x_bar = self.get_variables(state, ("v", "p", "r", "x_bar"))
        # matches the shape of the gradient [n, n, 2]
        v = self.prepare_variable(
            v, shape=(*x.shape, 2), dtype=x.dtype, device=device
        )  # pyright: ignore[reportArgumentType]
        # match the shape of the symmetrised gradient [n, n, 2, 2]
        p = self.prepare_variable(p, shape=(*x.shape, 2), dtype=x.dtype, device=device)
        # match the shape of the measured data
        r = self.prepare_variable(r, shape=z.shape, dtype=x.dtype, device=device)
        x_bar = self.prepare_variable(
            x_bar, shape=x.shape, default_value=x, dtype=x.dtype, device=device
        )

        lambda_reg = self.prepare_lambda(lambda_reg, domain="V")
        assert (
            torch.view_as_real(p).ndim == lambda_reg.ndim
        ), f"Expected {torch.view_as_real(p).ndim}. Got {lambda_reg.ndim}"

        g = self.grad_ops

        for _ in iterator:
            p = p + sigma * g.nabla_h(x_bar)
            p = self.clip_lambda(p, lambda_reg)  # projection operator
            r = r + sigma * (A(x_bar) - z)
            r = r / (1.0 + sigma)  # proximal operator

            # NOTE: The plus and minus signs are opposite to the paper.
            # If we switch any of the signs, the algorithm will not work.
            x_next = x + tau * (g.div_h_v(p) - AH(r))
            x_bar = x_next + theta * (x_next - x)
            x = x_next

            with torch.no_grad():
                torch.cuda.empty_cache()

        if return_dict:
            return {
                "x": x,
                "z": z,
                "v": v,
                "p": p,
                "r": r,
                "x_bar": x_bar,
            }  # pyright: ignore[reportReturnType]
        return x


class TgvPdhgTorch(PdhgTorch):
    def __init__(self, device: str | torch.device):
        super().__init__(device=device)

    def run_pdhg(
        self,
        iterator,  # determines how long to run algorithm for
        A: Callable,
        AH: Callable,
        lambda_reg: TgvLambdaReg,
        z: torch.Tensor,  # measured data
        state: torch.Tensor | TgvPdhgStateDict,
        sigma: float | torch.Tensor,
        tau: float | torch.Tensor,
        theta: float | torch.Tensor,
        return_dict: bool = False,
        device: str | torch.device | None = None,
    ) -> torch.Tensor | TgvPdhgStateDict:
        """Adapted Algorithm 2 in "Second Order Total Generalized Variation (TGV) for MRI"
        by Florian Knoll, Kristian Bredies, Thomas Pock, and Rudolf Stollberger.
        """
        if isinstance(state, dict):
            x = state["x"]  # initial image estimate
        elif isinstance(state, torch.Tensor):
            x = state  # initial image estimate
        else:
            raise ValueError(
                "Unknown input type. "
                + f"Expected dict or torch.Tensor, got {type(state)}"
            )

        if device is None:
            device = self.device
        assert device is not None, "Device must be specified"
        x = x.to(device)
        z = z.to(device)
        v, p, q, r, x_bar, v_bar = self.get_variables(
            state, ("v", "p", "q", "r", "x_bar", "v_bar")
        )
        # matches the shape of the gradient [n, n, 2]
        v = self.prepare_variable(v, shape=(*x.shape, 2), dtype=x.dtype, device=device)
        p = self.prepare_variable(p, shape=(*x.shape, 2), dtype=x.dtype, device=device)
        # match the shape of the symmetrised gradient [n, n, 2, 2]
        q = self.prepare_variable(q, shape=(*p.shape, 2), dtype=x.dtype, device=device)
        # match the shape of the measured data
        r = self.prepare_variable(r, shape=z.shape, dtype=x.dtype, device=device)
        x_bar = self.prepare_variable(
            x_bar, shape=x.shape, default_value=x, dtype=x.dtype, device=device
        )
        v_bar = self.prepare_variable(
            v_bar, shape=v.shape, default_value=v, dtype=x.dtype, device=device
        )

        lambda1_v = self.prepare_lambda(lambda_reg["lambda1_v"], domain="V")
        lambda0_w = self.prepare_lambda(lambda_reg["lambda0_w"], domain="W")
        assert (
            torch.view_as_real(p).ndim == lambda1_v.ndim
        ), f"Expected {torch.view_as_real(p).ndim}. Got {lambda1_v.ndim}"
        assert (
            torch.view_as_real(q).ndim == lambda0_w.ndim
        ), f"Expected {torch.view_as_real(q).ndim}. Got {lambda0_w.ndim}"

        g = self.grad_ops

        for _ in iterator:
            p = p + sigma * (g.nabla_h(x_bar) - v_bar)
            p = self.clip_lambda(p, lambda1_v)  # projection operator

            q = q + sigma * g.e_h(v_bar)
            q = self.clip_lambda(q, lambda0_w)  # projection operator

            r = r + sigma * (A(x_bar) - z)
            r = r / (1.0 + sigma)  # proximal operator ?

            x_next = x + tau * (g.div_h_v(p) - AH(r))
            x_bar = x_next + theta * (x_next - x)
            x = x_next

            v_next = v + tau * (g.div_h_w(q) + p)
            v_bar = v_next + theta * (v_next - v)
            v = v_next

            # with torch.no_grad():
            #     torch.cuda.empty_cache()

        if return_dict:
            return {
                "x": x,
                "z": z,
                "v": v,
                "p": p,
                "q": q,
                "r": r,
                "x_bar": x_bar,
                "v_bar": v_bar,
            }
        return x
