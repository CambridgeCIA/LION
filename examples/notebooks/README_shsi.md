# sHSI example: data and the forward model

Open `shsi_data_exploration.py` with VS Code or another editor supporting
Jupyter-style `# %%` Python cells. It can also run as a normal Python script.

This example uses the sHSI loader and masked observation operator in LION.
It only explores the measurement cube. It does **not** train a network,
run low-rank ALS, reconstruct missing measurements, or start Gibbs sampling.

## Run on the Linux lion_2 LION workspace

From the LION repository root:

```bash
cd /mnt/store_ext4/GIT/LION
micromamba activate lion_2
export LION_SHSI_DATA_DIR=/mnt/store_ext4/store/LION/ttv22/shsi/data
PYTHONPATH=. python examples/notebooks/shsi_data_exploration.py
```

The script displays three figures:

1. Ground truth, sampling mask, noiseless forward projection, and observed data
   at one spectral channel, on a common colour scale.
2. Ground-truth spectrum with measured points, sampling coverage, and the
   difference between measured data and the forward projection.
3. Ground-truth spatial images at three different spectral channels.

The mask is shown explicitly. Missing measurements are grey in the
sparse-value plots, rather than being confused with true zeros.

## Save figures without opening windows

```bash
SHSI_SAVE_FIGURES=1 SHSI_SHOW_FIGURES=0 MPLBACKEND=Agg \
  PYTHONPATH=. python examples/notebooks/shsi_data_exploration.py
```

The figures are saved under:

```text
$LION_SHSI_DATA_DIR/visualizations/test_input_0/
```

Change `EXAMPLE_NAME` in the configuration cell, or set
`LION_SHSI_EXAMPLE=test_input_1.npz` to explore 10% sampling.
The default is the 5% example.

`SPECTRAL_INDEX` and `PIXEL` can be changed in the same cell. With their
default `None` values, the script chooses a signal-rich slice and pixel.

## Forward operator

For a cube `x` of shape `(H, W, D)` and a binary observation mask `M`,
the forward model in this example is `A(x) = M * x`, with elementwise
multiplication. In general, the observed data can differ from
`A(ground_truth)` because of noise. In the two supplied example NPZ files,
however, every observed value equals its ground-truth value exactly.
The files contain a nonzero `sigma` field, but no noise was added to
the stored observed values. The residual plot therefore reports zero residuals.

The model is an entry-selection operator, not a Hadamard transform. It is
not the same operator used for PCM.

## References

- Source implementation and sample data: https://github.com/trung-vt/UE4sHSI
- A3SmartML project: https://a3smartml.ptb.de
