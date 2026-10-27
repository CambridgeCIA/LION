"""Random numbers that can be reproduced.

- ``seed_from_parts(*parts)``: a seed from strings (a stable hash).
- ``seeded(seed, *parts, device=...)``: run code with torch's random numbers
  seeded from ``seed`` and ``parts``, e.g. the noise of one data sample, so that
  sample is the same in every epoch, run, order and DataLoader worker. The
  global random state is restored afterwards.
- ``seed_all(seed)``: seed torch, numpy and Python's ``random``, as the solvers
  do when training starts; ``random_state()`` / ``set_random_state()`` keep
  and restore their state (checkpoints).

One seed gives the same numbers on the same device type and torch version.
torch's CPU and CUDA generators are different algorithms, so CPU and GPU give
different (statistically equivalent) numbers.
"""

from __future__ import annotations

import contextlib
import hashlib
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def seed_from_parts(*parts) -> int:
    """Generate a reproducible seed from a list of string parts.

    Args:
        *parts: List of strings to generate the seed from.

    Returns:
        A reproducible seed.
    """
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    seed_u64 = int.from_bytes(digest, byteorder="little", signed=False)
    seed_u32 = seed_u64 % (2**32)
    return int(seed_u32)


@contextlib.contextmanager
def seeded(seed, *parts, device=None):
    """Run the code in the ``with`` block with torch's random numbers seeded
    from ``seed`` and ``parts`` (``seed_from_parts``), on the CPU and, if
    ``device`` is a CUDA device, on it. The random state of those devices is
    restored afterwards, so the rest of the program draws the numbers it would
    have drawn anyway.

    ``seed=None`` does nothing: the code draws from the global random state.

        with seeded(0, "noise", patient_id, slice_index, device=sinogram.device):
            noisy = sinogram_add_noise(sinogram)
    """
    if seed is None:
        yield
        return
    device = torch.device(device if device is not None else "cpu")
    cuda = [device] if device.type == "cuda" else []
    value = seed_from_parts(seed, *parts)
    with torch.random.fork_rng(devices=cuda):
        torch.default_generator.manual_seed(value)
        if cuda:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(value)
        yield


def seed_all(seed: int) -> None:
    """Seed every random generator training draws from: torch's (CPU and all
    CUDA devices), numpy's and Python's ``random``. The solvers call it when
    training starts (their ``seed``); call it at the top of a script to also
    make the model's initial weights repeatable."""
    torch.manual_seed(seed)  # also every CUDA device
    np.random.seed(seed % 2**32)
    random.seed(seed)


def random_state() -> dict:
    """The state of every generator ``seed_all`` seeds, as tensors, lists and
    numbers (so a checkpoint holding it loads with ``weights_only=True``)."""
    name, keys, position, has_gauss, cached_gauss = np.random.get_state()
    version, internal, gauss_next = random.getstate()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": [name, keys.tolist(), int(position), int(has_gauss), cached_gauss],
        "random": [version, list(internal), gauss_next],
    }


def set_random_state(state: dict) -> None:
    """Restore the generators from ``random_state()``. CUDA states are only
    restored on a machine with as many CUDA devices. The states may have been
    loaded onto a GPU (``torch.load(map_location=...)``): torch takes them as
    CPU tensors."""
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    name, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state(
        (name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gauss)
    )
    version, internal, gauss_next = state["random"]
    random.setstate((version, tuple(internal), gauss_next))


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
