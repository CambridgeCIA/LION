# README for developers

WIP

## Familiarize yourself with the structure of the tool

<pre>
./AItomotools     : The toolbox
├── CTtools                 -> Tools for Computed Tomography specifically
├── data_loaders            -> Tools for 1) preprocessing data 2) loading data for training
├── metrics                 -> Tools to evaluate performance
├── models                  -> Machine Learning reconstruction models
└── utils                   -> Other type of generic utils

./demos           : Folder with demos to learn about the toolbox. Please add demos about your new functionality

./scripts         : Scripts that fully reproduce papers/data
├── data_generation_scripts -> Scripts that produce the datasets.
├── paper_scripts           -> Code that reproduces published papers

./utils           : Other utilities. e.g. pre-commit scripts. 
</pre>

## Aitomotools base class

Go to `./AItomotools/models/AItomomodel.py` to see the base class. 
All models should inherit from this. It provides useful tools, like `save()` , `load()`, `save_checkpoint()`, `default_parameters()` etc. 
You must make your models use these functions because they save more than the models, also parameters and commit hashes, so the code is more reproducible. 