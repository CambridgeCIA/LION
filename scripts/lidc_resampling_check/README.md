# Checking #264 on the real LIDC-IDRI data

#264 resamples every processed LIDC-IDRI slice to 0.977 mm (500 mm over
512x512) and gives the LIDC-IDRI experiments a 500 mm geometry. These checks
are meant to be run before merging it. They let you look at the resampling and
the experiments' doses on the real data, and try the in-place conversion on a
copy first. They are removed once #264 is merged.

`check_lidc_resampling.py` has four steps. The first two only read the data.
None of them changes your processed dataset: the trial works on a copy.

| step | needs | time | output |
|---|---|---|---|
| `resampling` | CPU, LION env | < 1 min | `resampling.png`, `resampling.json` |
| `doses` | GPU, LION env | a few min | `doses.png` (`doses_sweep.png` with `--sweep`) |
| `trial` | CPU, LION env | a few min, ~4 GB of disk per 30 patients | a copy of the first patients, and the slices the conversion should give |
| `loader` | GPU, LION env, after converting the copy | a few min | `loader.png`, and a list of problems (if any) |

## Commands (on holly)

Use the LION environment (e.g. `lion6`), with `LION_DATA_PATH` set as usual
and the #264 branch checked out. `pylidc` must be installed: the pixel spacings
are read from its database, without importing it.

```
git fetch origin && git checkout feat/lidc-uniform-spacing
cd /path/to/LION
mamba activate lion6

# 1. The resampling, for 6 patients from the smallest to the largest spacing
python scripts/lidc_resampling_check/check_lidc_resampling.py resampling --out lidc_check

# 2. The doses, before and after (add --sweep 3e4,2e4,1e4,5e3,2e3,1e3 to see other I0)
python scripts/lidc_resampling_check/check_lidc_resampling.py doses --out lidc_check

# 3. The conversion, on a copy of the first 30 patients (choose a folder with space)
python scripts/lidc_resampling_check/check_lidc_resampling.py trial --trial-dir /scratch/lidc_trial --out lidc_check
LION_DATA_PATH=/scratch/lidc_trial python LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py --resample-only
python scripts/lidc_resampling_check/check_lidc_resampling.py loader --trial-dir /scratch/lidc_trial --out lidc_check
```

The `trial` step prints the two commands that follow it, with your paths.

`pre_process_lidc_idri.py` imports pylidc, which needs `pkg_resources`, gone
from setuptools 81. If it fails with `No module named 'pkg_resources'`, run
`pip install "setuptools<81"` in the LION environment, or use the
pre-processing's own `lidc_idri` environment. It worked here with numpy 2 and
`setuptools<81`.

If all looks right, convert the real dataset the same way, with your usual
`LION_DATA_PATH`:
`python LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py --resample-only`.
It replaces the slices: the processed data from before can then only be made
again from the raw DICOM files, so keep a copy if you want it. If it is
interrupted, run it again; finished patients are skipped.

## What to look at

**`resampling.png`**: one row per patient, from the smallest pixel spacing
(about 0.46 mm, a 236 mm field of view) to the largest (0.977 mm, 500 mm).
- **Columns 1–2**: the slice before and after, on the same mm axes (HU window
  −1000 to 400). The anatomy should have the same size and position. After,
  the image covers 500 mm, so scans with a smaller field of view are
  surrounded by air. The red contour is the nodule mask, if the slice has
  one; it should sit on the nodule in both.
- **Column 3**: before minus after, with after interpolated back onto the
  original grid (±200 HU). Only edges should show: the resampling smooths
  them (coarser pixels). Large areas of colour would mean a shift or a wrong
  scale.
- **Column 4**: a row through the centre, before and after, in mm. The two
  curves should lie on each other, the after one a bit smoother.
- **The printed table** (also in `resampling.json`): body area (HU > −500)
  before and after should agree within about 1 %, and the body centre should
  not move (a fraction of a mm). Nodule areas should agree roughly; small
  nodules change more, because they are only a few pixels.

**`doses.png`**: a small, a typical and a large patient (5th, 50th and 95th
percentile of the thickest water path, after resampling).
- **Columns**: the ground truth, then for each experiment (clinical, low dose,
  extremely low dose) the FDK before #264 (each scan's own slices taken as
  300 mm, the images you chose the doses on) and after (the real size,
  500 mm geometry).
- **What to expect**: patients are now their real size, so most of them are
  thicker than before, and the same I0 gives more noise. With `--sweep`, extra
  columns show the clinical experiment after #264 at other I0, to choose new
  values by eye as last time.
- **Each title** has the PSNR against the ground truth.

**`loader`**: checks what `--resample-only` wrote on the copy:
- the `pre_processing.json` record and each patient's `pixel_spacing.json`;
- no temporary folders left;
- the slices and masks equal to the ones `trial` computed.

It then loads test slices of the three experiments through LION's data
loader, and checks:
- that the loader read the spacing, without warnings about the pixel spacing;
- the shapes: a 512x512 image and 1024 detector pixels.

`loader.png` shows a target and the FDK of the experiment's noisy sinogram for
three test slices per experiment. It ends with "All checks passed." or a list
of problems.
