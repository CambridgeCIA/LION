import torch


def count_parameters(model: torch.nn.Module) -> tuple[int, int, float]:
    """Count model parameters.

    Args:
    model: Model whose parameters are counted.

    Returns:
        total_params: Number of all parameters.
        trainable_params: Number of trainable parameters.
        size_mib: Parameter storage size in MiB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_mib = size_bytes / 1024**2

    return total_params, trainable_params, size_mib
