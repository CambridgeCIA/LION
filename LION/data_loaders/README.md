# Supported DataSets

## 2DeteCT

A dataset of uCT scans with different settings for sinograms. The data is made of composition of different materials. [Read more in the Nature paper](https://www.nature.com/articles/s41597-023-02484-6).
To install 2DeteCT, run `python ./LION/data_loaders/2deteCT/download.py` and once finished (probably a few hours later) run `python ./LION/data_loaders/2deteCT/pre_process_2deteCT.py`. Once finished, you should have `LION_DATA_PATH/processed/2detect` folder in your system. You can now delete `LION_DATA_PATH/raw/2detect` to save space if you want. 

The data loader is `LION.data_loaders.deteCT`; its class docstring describes the modes, the two detectors (samples 1-63 and 64-109 were measured with different detectors), the train/validation/test splits and the noise options.

## LIDC-IDRI

A dataset of 1100 lung 3D CT scans with segmentations and diagnostics of lung nodules. These are available from the TCIA website [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254). However, a convenient download functionality is provided at [LION/LION/data_loaders/LIDC_IDRI/download_LIDC_IDRI.sh](LIDC_IDRI/download_LIDC_IDRI.sh). Running this should download the full dataset to your `LION_DATA_PATH/raw/LIDC-IDRI`, and will attempt to use Java, Apptainer, Singularity then Docker to run the NBIA downloader tool.

This should give a folder structure as:

```
LION_DATA_PATH/raw/LIDC-IDRI/
├── LIDC-IDRI
├── metadata.csv
└── tcia-diagnosis-data-2012-04-20.xls
```

Then, to process the dataset, create a new conda enviroment with `conda env create -f ./LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri_environment.yml`, activate it as `conda activate lidc_idri` and run the python file `python ./LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py`. 

The data loader is `LION.data_loaders.LIDC_IDRI`. Its images are densities (g/cm³); the noise of the CT experiments (`LION.experiments.ct_experiments`, see its module docstring) is added to their sinograms.

You can now delete `LION_DATA_PATH/raw/LIDC-IDRI` to save space if you want. Note that the download size is around 133GB, so both download and processing can take hours.

More details on the downloader tool are available [here](LIDC_IDRI/README.md)

**Pixel spacing.** The pre-processing resamples every slice to 500 mm over 512x512 pixels (0.977 mm, the largest pixel spacing in LIDC-IDRI, so no scan is cropped; scans with a smaller field of view are surrounded by air), and writes `pre_processing.json` in the processed folder. Datasets processed before October 2026 kept each scan's own spacing (0.46 to 0.98 mm). To resample one in place, without the raw DICOM files, run `python ./LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py --resample-only` (with `--folder <processed LIDC-IDRI folder>` if it is not `LION_DATA_PATH/processed/LIDC-IDRI`), in the `lidc_idri` environment or in LION's (pylidc needs `pkg_resources`, so `pip install "setuptools<81"` there). Patients with damaged files are left as they are and listed in `pre_processing.json`; the data loader leaves them out with a warning. `LIDC_IDRI.default_geometry()` is the CT geometry that matches the resampled slices.

## Walnuts
A dataset of uCT scans of walnuts. [Read more in the Nature paper](https://www.nature.com/articles/s41597-019-0235-y).
You can download it using `python LION/data_loaders/walnuts/download.py`
pre-processing and Data Loading WIP
