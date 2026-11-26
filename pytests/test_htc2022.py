# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
import numpy as np
import pandas as pd
import pytest
import torch

from LION.data_loaders.HTC2022 import HTC2022, geometry, segment


def test_htc2022_reads_the_prepared_data(tmp_path, monkeypatch):
    """The loader gives each split's phantoms, their sinograms cut to the
    geometry's angles, and the chosen target; it refuses other geometries and
    a validation split. The segmentation keeps the acrylic, not the holes."""
    monkeypatch.setenv("LION_DATA_PATH", str(tmp_path))
    rows = []
    for name, split, level in (
        ("ta", "train", 0),
        ("1a", "test", 1),
        ("7c", "test", 7),
    ):
        folder = tmp_path / name
        folder.mkdir()
        np.save(folder / "sinogram.npy", np.random.rand(720, 560).astype(np.float32))
        np.save(
            folder / "reconstruction.npy", np.random.rand(512, 512).astype(np.float32)
        )
        np.save(folder / "segmentation.npy", np.ones((512, 512), np.uint8))
        rows.append(dict(name=name, split=split, level=level))
    pd.DataFrame(rows).to_csv(tmp_path / "samples.csv", index=False)

    parameters = HTC2022.default_parameters()
    parameters.path_to_dataset = tmp_path
    test = HTC2022("test", geometry(181), parameters)
    assert test.sample_ids() == ["1a", "7c"]
    sinogram, target = test[1]
    assert sinogram.shape == (1, 181, 560) and target.shape == (1, 512, 512)

    parameters = HTC2022.default_parameters()
    parameters.path_to_dataset = tmp_path
    parameters.target = "segmentation"
    train = HTC2022("train", parameters=parameters)
    assert len(train) == 1 and train[0][0].shape == (1, 720, 560)
    assert torch.all(train[0][1] == 1)

    with pytest.raises(ValueError):
        HTC2022("validation", parameters=parameters)
    with pytest.raises(ValueError):
        HTC2022("test", geometry(90, start=10), parameters)

    y, x = np.mgrid[:128, :128]
    disc = ((x - 64) ** 2 + (y - 64) ** 2 < 50**2).astype(float)
    disc[(x - 64) ** 2 + (y - 50) ** 2 < 10**2] = 0  # a hole
    noisy = disc + 0.1 * np.random.default_rng(0).standard_normal(disc.shape)
    segmentation = segment(noisy)
    assert segmentation[64, 64] == 1 and segmentation[50, 64] == 0
    assert np.mean(segmentation == disc) > 0.98
