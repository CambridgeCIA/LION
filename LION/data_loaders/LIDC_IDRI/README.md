# LIDC-IDRI Download

Use `download_LIDC_IDRI.sh` to download the LIDC-IDRI TCIA manifest with the NBIA Data Retriever CLI. The script installs the retriever under the dataset folder, so it does not need system-wide installation.

## Basic Usage

By default, the script reads `LION_DATA_PATH` from `LION/utils/paths.py` and uses its `raw` subfolder.

```bash
./download_LIDC_IDRI.sh
```

To override the raw parent folder for a single run, set `RAW_PATH`:

```bash
export RAW_PATH=/path/to/raw
./download_LIDC_IDRI.sh
```

The script creates any missing directories, including:

```text
$RAW_PATH/tools/nbia-data-retriever
$RAW_PATH/LIDC-IDRI
```

The downloaded dataset contents are written under `$RAW_PATH/LIDC-IDRI` by NBIA Data Retriever.

The script also downloads the LIDC diagnosis file, leaving:

```text
$RAW_PATH/LIDC-IDRI/tcia-diagnosis-data-2012-04-20.xls
```

## Resume Behavior

The default behavior is resume-friendly. If a previous download was interrupted, rerun the same command; the script answers NBIA's resume prompt with `M`, meaning download missing series only.

MD5 verification is enabled by default for NBIA downloads.

To change this behavior:

```bash
# Default: resume by downloading missing series only
export NBIA_RESUME_CHOICE=M

# Redownload all series if NBIA asks
export NBIA_RESUME_CHOICE=A

# Let NBIA prompt interactively
export NBIA_RESUME_CHOICE=prompt
```

## Optional Settings

If the TCIA manifest is somewhere else:

```bash
export TCIA_MANIFEST_PATH=/path/to/LIDC-IDRI.tcia
```

The script uses host Java 17+ when available. Otherwise it tries Apptainer, Singularity, then Docker with `eclipse-temurin:17-jre`. To force container use:

```bash
export NBIA_FORCE_CONTAINER=1
```

To use a different Java container image:

```bash
export JAVA_CONTAINER_IMAGE=eclipse-temurin:17-jre
```

## After Downloading

After the raw dataset is downloaded, run the preprocessing script using the dedicated preprocessing environment:

```bash
conda env create -f pre_process_lidc_idri_environment.yml
conda activate lidc_idri
python pre_process_lidc_idri.py
```

The preprocessing run finishes with a completeness check. On the current
downloaded dataset, the complete processed output contains at least `282,776`
regular files and `1,010` processed patient directories. If the count is short,
rerun the downloader with `NBIA_RESUME_CHOICE=M ./download_LIDC_IDRI.sh` to fetch
missing series, or use `NBIA_RESUME_CHOICE=A ./download_LIDC_IDRI.sh` if the
NBIA retriever needs to redownload all series.

The preprocessing script automatically writes the `pylidc` DICOM location to `~/.pylidcrc` using `LIDC_IDRI_PATH` from `LION/utils/paths.py`. With the default download layout, this points `pylidc` at:

```text
LION_DATA_PATH/raw/LIDC-IDRI/LIDC-IDRI
```

If disk space is limited, the preprocessing script can remove each per-patient
raw DICOM folder after the matching processed folder has been written:

```bash
python pre_process_lidc_idri.py --delete-raw-after-processing
```

To reduce deletion frequency, remove raw folders in completed blocks:

```bash
python pre_process_lidc_idri.py --delete-raw-after-processing --raw-delete-block-size 10
```

Use a dry run first to inspect the patient folders that would be deleted:

```bash
python pre_process_lidc_idri.py --dry-run-raw-delete --raw-delete-block-size 10
```

## Optional PaDIS Cache Preparation

The PaDIS paper scripts can use reusable tensor caches derived from the processed LIDC-IDRI slices. This avoids repeatedly walking many small `slice_*.npy` files when launching PaDIS training jobs. Run this only after `pre_process_lidc_idri.py` has produced `LION_DATA_PATH/processed/LIDC-IDRI`.

On the Cambridge Slurm cluster, submit the cache preparation job from the LION repository root:

```bash
cd /home/tjh200/DiS/Project/LION
scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/submit_PaDIS_A100_prepare_full_cache.sh
```

By default this submits a CPU job on the `icelake` partition with 8 CPU cores and 128G memory. It builds zstd-compressed archives for:

```text
256-default
256-full
512-default
```

The full 512x512 LIDC cache is intentionally not built by this helper because it would be much larger. The PaDIS Slurm training, pilot, and profiling scripts stage these prepared archives into `/ramdisks/$USER` by default and fail fast if the matching archive has not been prepared.

Default archive locations are:

```text
LION_DATA_PATH/processed/LIDC-IDRI-cache/padis_256/archives
LION_DATA_PATH/processed/LIDC-IDRI-cache/padis_512/archives
```
