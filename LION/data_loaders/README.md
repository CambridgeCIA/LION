# Supported DataSets

## 2DeteCT

A dataset of uCT scans with different settings for sinograms. The data is made of composition of different materials. [Read more in the Nature paper](https://www.nature.com/articles/s41597-023-02484-6).
To install 2DeteCT, run `python ./LION/data_loaders/2DeteCT/dowload.py` and once finished (probably a few hours later) run `python ./LION/data_loaders/2DeteCT/pre_processs_2deteCT.py`. Once finished, you should have `LION_DATA_PATH/processed/2detect` folder in your system. You can now delete `LION_DATA_PATH/raw/2detect` to save space if you want. 

## LIDC-IDRI

A dataset of 1100 lung 3D CT scans with segmentations and diagnostics of lung nodules. To dowload the dataset, [follow the instructions in the dataset webpage](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254) (requires a third party software). Put this dataset in `LION_DATA_PATH/raw/LIDC-IDRI` such that it contains a folder structure as:

```
LION_DATA_PATH/raw/LIDC-IDRI/
├── LIDC-IDRI
├── LIDC-IDRI_MetaData.csv
├── metadata.csv
└── tcia-diagnosis-data-2012-04-20.xls
```

Then, create a new conda enviroment with `conda env create -f ./LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri_environment.yml`, activate it as `conda activate lidc_idri` and run the python file `python ./LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py`. 

You can now delete `LION_DATA_PATH/raw/LIDC-IDRI` to save space if you want. 

## Walnuts
A dataset of uCT scans of walnuts. [Read more in the Nature paper](https://www.nature.com/articles/s41597-023-02484-6).
You can dowload it using `python LION/data_loaders/walnuts/dowload_walnuts.py`
pre-processing and Data Loading WIP
