from __future__ import annotations

import hashlib

import numpy as np
import torch


def seed_from_parts(*parts: str) -> int:
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
