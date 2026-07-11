This folder contains some helper functions to deal with the LUNA16 dataset

Not necesarily a template to follow, as its a bit messy as is, but this is what the files are for:

download.py -> dowloads the luna dataset. 

pre_processing.py -> contains classes and objects to deal with the raw extracted LUNA16 data. These are useful to crate instances of your own data. AItomotools/scripts/data_generation_scripts uses this to create an easy-to-handle version of the datset, to be used at training/testing

luna16_dataset.py -> Contains the real code that is used by pytorch to load and use dataset during training. 