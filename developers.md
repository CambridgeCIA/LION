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
</pre>

## Template of a model

Go to `./AItomotools/models/template.py` to see the template you should use for a model. 