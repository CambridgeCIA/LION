"""Direct PyTorch PDHG implementations for 2-D TV and TGV reconstruction."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from LION.operators.Operator import Operator

TVLambda = float | torch.Tensor
TGVLambda = torch.Tensor | tuple[float | torch.Tensor, float | torch.Tensor]
State = tuple[torch.Tensor, ...]


def _dx_forward(u: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(u)
    out[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
    return out


def _dy_forward(u: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(u)
    out[..., :, :-1] = u[..., :, 1:] - u[..., :, :-1]
    return out


def _dx_backward(u: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(u)
    out[..., 0, :] = u[..., 0, :]
    out[..., 1:-1, :] = u[..., 1:-1, :] - u[..., :-2, :]
    out[..., -1, :] = -u[..., -2, :]
    return out


def _dy_backward(u: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(u)
    out[..., :, 0] = u[..., :, 0]
    out[..., :, 1:] = u[..., :, 1:] - u[..., :, :-1]
    out[..., :, -1] = -u[..., :, -2]
    return out


def gradient(u: torch.Tensor) -> torch.Tensor:
    """Return the forward 2-D gradient with zero derivative at the outer boundary."""
    return torch.stack((_dx_forward(u), _dy_forward(u)), dim=-1)


def divergence(v: torch.Tensor) -> torch.Tensor:
    """Return the discrete divergence used by the original direct PDHG implementation."""
    return _dx_backward(v[..., 0]) + _dy_backward(v[..., 1])


def sym_gradient(v: torch.Tensor) -> torch.Tensor:
    """Return the 2-D discrete symmetrized gradient."""
    dx_1 = _dx_backward(v[..., 0])
    dy_1 = _dy_backward(v[..., 0])
    dx_2 = _dx_backward(v[..., 1])
    dy_2 = _dy_backward(v[..., 1])

    off_diagonal = 0.5 * (dy_1 + dx_2)
    return torch.stack(
        (
            torch.stack((dx_1, off_diagonal), dim=-1),
            torch.stack((off_diagonal, dy_2), dim=-1),
        ),
        dim=-2,
    )


def sym_divergence(w: torch.Tensor) -> torch.Tensor:
    """Return the divergence paired with :func:`sym_gradient` in the direct TGV scheme."""
    first = _dx_forward(w[..., 0, 0]) + _dy_forward(w[..., 0, 1])
    second = _dx_forward(w[..., 0, 1]) + _dy_forward(w[..., 1, 1])
    return torch.stack((first, second), dim=-1)


def _prepare_bound(
    value: float | torch.Tensor,
    *,
    reference: torch.Tensor,
    trailing_dims: int,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        bound = value.to(device=reference.device, dtype=reference.real.dtype)
    else:
        bound = torch.tensor(value, device=reference.device, dtype=reference.real.dtype)

    if bound.ndim > 0:
        if bound.shape != reference.shape:
            raise ValueError(
                f"Expected lambda map shape {tuple(reference.shape)} or a scalar, got {tuple(bound.shape)}."
            )
        for _ in range(trailing_dims):
            bound = bound.unsqueeze(-1)
    return bound


def _run_iterations(
    step: Callable[..., State],
    state: State,
    constants: tuple[torch.Tensor, ...],
    *,
    max_iterations: int,
    use_checkpoint: bool,
    checkpoint_chunk_size: int | None,
    checkpoint_use_reentrant: bool,
    use_tqdm: bool,
    iteration_callback: Callable[[int, torch.Tensor], None] | None = None,
    callback_every: int = 1,
) -> State:
    if callback_every < 1:
        raise ValueError("callback_every must be positive.")
    if iteration_callback is not None and use_checkpoint and torch.is_grad_enabled():
        raise ValueError(
            "Iteration callbacks are not supported with gradient checkpointing."
        )
    if not use_checkpoint or not torch.is_grad_enabled():
        iterator = (
            tqdm(range(max_iterations), desc="PDHG iterations", dynamic_ncols=True)
            if use_tqdm
            else range(max_iterations)
        )
        for iteration_index in iterator:
            state = step(*state, *constants)
            iteration = iteration_index + 1
            if iteration_callback is not None and (
                iteration % callback_every == 0 or iteration == max_iterations
            ):
                iteration_callback(iteration, state[0])
        return state

    if checkpoint_chunk_size is None:
        checkpoint_chunk_size = max_iterations

    n_state = len(state)
    remaining = max_iterations
    n_chunks = (max_iterations + checkpoint_chunk_size - 1) // checkpoint_chunk_size
    iterator = (
        tqdm(range(n_chunks), desc="PDHG chunks", dynamic_ncols=True)
        if use_tqdm
        else range(n_chunks)
    )

    for _ in iterator:
        this_chunk = min(checkpoint_chunk_size, remaining)
        remaining -= this_chunk

        def run_chunk(*args: torch.Tensor, n_iterations: int = this_chunk) -> State:
            chunk_state = tuple(args[:n_state])
            chunk_constants = tuple(args[n_state:])
            for _ in range(n_iterations):
                chunk_state = step(*chunk_state, *chunk_constants)
            return chunk_state

        state = checkpoint(
            run_chunk,
            *state,
            *constants,
            use_reentrant=checkpoint_use_reentrant,
        )

    return state


def tv_pdhg(
    *,
    measurement: torch.Tensor,
    initial_image: torch.Tensor,
    forward_op: Operator,
    lambda_reg: TVLambda,
    max_iterations: int,
    primal_stepsize: float | torch.Tensor,
    dual_stepsize: float | torch.Tensor,
    theta: float | torch.Tensor = 1.0,
    use_checkpoint: bool = False,
    checkpoint_chunk_size: int | None = None,
    checkpoint_use_reentrant: bool = False,
    use_tqdm: bool = False,
    iteration_callback: Callable[[int, torch.Tensor], None] | None = None,
    callback_every: int = 1,
) -> torch.Tensor:
    """Run direct anisotropic 2-D TV-PDHG, optionally reporting accepted iterates.

    The callback is called after iterations divisible by ``callback_every`` and
    after the final iteration, without restarting the primal or dual state.
    """
    x = initial_image
    x_bar = initial_image
    p = torch.zeros((*x.shape, 2), dtype=x.dtype, device=x.device)
    r = torch.zeros_like(measurement)
    lambda_bound = _prepare_bound(lambda_reg, reference=x, trailing_dims=1)

    def step(
        x: torch.Tensor,
        x_bar: torch.Tensor,
        p: torch.Tensor,
        r: torch.Tensor,
        bound: torch.Tensor,
    ) -> State:
        p = torch.clamp(p + dual_stepsize * gradient(x_bar), -bound, bound)
        r = (r + dual_stepsize * (forward_op(x_bar) - measurement)) / (
            1.0 + dual_stepsize
        )

        x_next = x + primal_stepsize * (divergence(p) - forward_op.adjoint(r))
        x_bar = x_next + theta * (x_next - x)
        return x_next, x_bar, p, r

    x, _, _, _ = _run_iterations(
        step,
        (x, x_bar, p, r),
        (lambda_bound,),
        max_iterations=max_iterations,
        use_checkpoint=use_checkpoint,
        checkpoint_chunk_size=checkpoint_chunk_size,
        checkpoint_use_reentrant=checkpoint_use_reentrant,
        use_tqdm=use_tqdm,
        iteration_callback=iteration_callback,
        callback_every=callback_every,
    )
    return x


def tgv_pdhg(
    *,
    measurement: torch.Tensor,
    initial_image: torch.Tensor,
    forward_op: Operator,
    lambda_reg: TGVLambda,
    max_iterations: int,
    primal_stepsize: float | torch.Tensor,
    dual_stepsize: float | torch.Tensor,
    theta: float | torch.Tensor = 1.0,
    use_checkpoint: bool = False,
    checkpoint_chunk_size: int | None = None,
    checkpoint_use_reentrant: bool = False,
    use_tqdm: bool = False,
) -> torch.Tensor:
    """Run the direct 2-D TGV-PDHG scheme used by the original U-TGV implementation."""
    x = initial_image
    x_bar = initial_image

    v = torch.zeros((*x.shape, 2), dtype=x.dtype, device=x.device)
    v_bar = v
    p = torch.zeros_like(v)
    q = torch.zeros((*x.shape, 2, 2), dtype=x.dtype, device=x.device)
    r = torch.zeros_like(measurement)

    if isinstance(lambda_reg, tuple):
        lambda_1_v, lambda_0_w = lambda_reg
    else:
        if lambda_reg.ndim != 4 or lambda_reg.shape[1] != 2:
            raise ValueError(
                "TGV lambda maps must have shape (batch, 2, height, width), "
                f"got {tuple(lambda_reg.shape)}."
            )
        lambda_1_v = lambda_reg[:, 0:1, ...]
        lambda_0_w = lambda_reg[:, 1:2, ...]

    lambda_1_bound = _prepare_bound(lambda_1_v, reference=x, trailing_dims=1)
    lambda_0_bound = _prepare_bound(lambda_0_w, reference=x, trailing_dims=2)

    def step(
        x: torch.Tensor,
        x_bar: torch.Tensor,
        v: torch.Tensor,
        v_bar: torch.Tensor,
        p: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        lambda_1: torch.Tensor,
        lambda_0: torch.Tensor,
    ) -> State:
        p = torch.clamp(
            p + dual_stepsize * (gradient(x_bar) - v_bar), -lambda_1, lambda_1
        )
        q = torch.clamp(q + dual_stepsize * sym_gradient(v_bar), -lambda_0, lambda_0)
        r = (r + dual_stepsize * (forward_op(x_bar) - measurement)) / (
            1.0 + dual_stepsize
        )

        x_next = x + primal_stepsize * (divergence(p) - forward_op.adjoint(r))
        x_bar = x_next + theta * (x_next - x)

        v_next = v + primal_stepsize * (sym_divergence(q) + p)
        v_bar = v_next + theta * (v_next - v)

        return x_next, x_bar, v_next, v_bar, p, q, r

    x, _, _, _, _, _, _ = _run_iterations(
        step,
        (x, x_bar, v, v_bar, p, q, r),
        (lambda_1_bound, lambda_0_bound),
        max_iterations=max_iterations,
        use_checkpoint=use_checkpoint,
        checkpoint_chunk_size=checkpoint_chunk_size,
        checkpoint_use_reentrant=checkpoint_use_reentrant,
        use_tqdm=use_tqdm,
    )
    return x
