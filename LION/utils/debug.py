from __future__ import annotations

from collections import defaultdict

import torch


def summarize_gradients(model: torch.nn.Module) -> None:
    groups = defaultdict(
        lambda: {
            "n_tensors": 0,
            "n_with_grad": 0,
            "n_nan": 0,
            "total_norm_sq": 0.0,
            "max_norm": 0.0,
        }
    )

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Group by broad component.
        if "lambda_map_network.scale_factors_raw" in name:
            group = "scale_factors_raw"
        elif "lambda_map_network.cnn_block" in name:
            group = "lambda_cnn"
        elif name.startswith("cnn."):
            group = "old_cnn"
        elif name in {"t_1_v", "t_0_w"}:
            group = "old_scale_factors"
        else:
            group = name.split(".")[0]

        groups[group]["n_tensors"] += 1

        if p.grad is None:
            continue

        grad_norm = p.grad.detach().norm().item()
        groups[group]["n_with_grad"] += 1
        groups[group]["total_norm_sq"] += grad_norm**2
        groups[group]["max_norm"] = max(groups[group]["max_norm"], grad_norm)

        if not torch.isfinite(p.grad).all():
            groups[group]["n_nan"] += 1

    print("\nGradient summary")
    print("-" * 80)
    print(
        f'{"group":30s} {"with_grad/tensors":>18s} {"total_norm":>14s} {"max_norm":>14s} {"nan_tensors":>12s}'
    )

    for group, s in sorted(groups.items()):
        total_norm = s["total_norm_sq"] ** 0.5
        print(
            f"{group:30s} "
            f'{s["n_with_grad"]:>6d}/{s["n_tensors"]:<6d} '
            f"{total_norm:>14.6e} "
            f'{s["max_norm"]:>14.6e} '
            f'{s["n_nan"]:>12d}'
        )


def debug_finite_tensor(name: str, x: torch.Tensor) -> None:
    return
    x = x.detach()
    y = x.real if torch.is_complex(x) else x
    print(
        f"[finite] {name}: "
        f"finite={torch.isfinite(y).all().item()} "
        f"shape={tuple(x.shape)} "
        f"min={y.min().item():.3e} "
        f"mean={y.mean().item():.3e} "
        f"max={y.max().item():.3e}"
    )


def debug_finite_model(model: torch.nn.Module, tag: str) -> None:
    return
    bad_params = []
    bad_grads = []

    for name, p in model.named_parameters():
        if not torch.isfinite(p.detach()).all():
            bad_params.append(name)
        if p.grad is not None and not torch.isfinite(p.grad.detach()).all():
            bad_grads.append(name)

    print(f"[finite] {tag}: bad_params={bad_params[:10]} count={len(bad_params)}")
    print(f"[finite] {tag}: bad_grads={bad_grads[:10]} count={len(bad_grads)}")


def tensor_stats(name: str, x: torch.Tensor) -> None:
    """Prints statistics of a tensor for debugging purposes."""
    x_detached = x.detach()
    print(
        f"{name}: "
        f"min={x_detached.min().item():.3e}, "
        f"max={x_detached.max().item():.3e}, "
        f"mean={x_detached.mean().item():.3e}, "
        f"norm={torch.linalg.norm(x_detached).item():.3e}, "
        f"finite={torch.isfinite(x_detached).all().item()}"
    )
