"""Reproducability utilities."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def seed_from_parts(*parts: str) -> int:
    """Generate a reproducible seed from a list of string parts.

    Args:
        *parts: List of strings to generate the seed from.

    Returns:
        A reproducible seed.
    """
    payload = "|".join(parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    seed_u64 = int.from_bytes(digest, byteorder="little", signed=False)
    seed_u32 = seed_u64 % (2**32)
    return int(seed_u32)


def random_float(
    low: float, high: float, *, rng: torch.Generator | None = None
) -> float:
    return float((high - low) * torch.rand(1, generator=rng).item() + low)


def random_int(
    low: int = 0, high: int = 2**31 - 1, *, rng: torch.Generator | None = None
) -> int:
    return int(torch.randint(low, high, (1,), generator=rng).item())


def make_noisy(clean: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=clean.shape).astype(np.float32)
    noisy = clean.astype(np.float32) + noise
    return noisy


def random_permutation(num_indices: int, seed: int) -> np.ndarray:
    """Generate a reproducible random permutation of indices.

    Args:
        num_indices: The number of indices to permute.
        seed: The seed to use for the random number generator.

    Returns:
        A reproducible random permutation of indices.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_indices)
    return permutation


@dataclass(kw_only=True)
class RandomStates:
    """Random states restored from a checkpoint."""

    rng_state: torch.Tensor | None = None
    torch_random_state: torch.Tensor | None = None
    np_random_state: Any = None
    random_random_state: tuple[Any, ...] | None = None


def make_reproducible_base(
    config, random_states: RandomStates | None = None
) -> torch.Generator:
    """Initialize the random-number generators used by training."""
    if random_states is None:
        random_states = RandomStates()

    rng = torch.Generator(device="cpu")
    if random_states.rng_state is None:
        rng.manual_seed(config.rng_seed)
    else:
        rng.set_state(random_states.rng_state)

    if random_states.torch_random_state is None:
        torch.manual_seed(config.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.torch_seed)
    else:
        torch.set_rng_state(random_states.torch_random_state)

    if random_states.np_random_state is None:
        np.random.seed(config.numpy_seed)
    else:
        np.random.set_state(random_states.np_random_state)

    if random_states.random_random_state is None:
        random.seed(config.random_seed)
    else:
        random.setstate(random_states.random_random_state)

    return rng
