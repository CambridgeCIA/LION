# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Emilien Valat
# =============================================================================


from typing import List
from pathlib import Path
import math
import ast

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

PATH_TO_2DeteCT = Path(f'/store/DAMTP/ab2860/AItomotools/data/AItomotools/raw/2DeteCT')

MIXES = [
    'Mix 1',
    'Mix 2',
    'Mix 3',
    'Fig',
    'Almond',
    'Banana',
    'Raisin',
    'Walnut',
    'Coffee beans',
    'Lava Stone',
    'Mix 3 (OOD Noise)',
    'Titanium',
    'Peanut',
    'Pistachio',
    'Hazelnut',
    'Grape',
    'Fresh Fig'
]

def format_index(index: int) -> str:
    str_index = str(index)
    while len(str_index) < 5:
        str_index = "0" + str_index
    return str_index

class ScanDataset(Dataset):
    def __init__(
        self,
        query:str,
        path_to_dataset = PATH_TO_2DeteCT,
        transform = None
        ):
        ### Defining the path to data
        self.path_to_dataset = path_to_dataset
        """
        The path_to_dataset attribute (pathlib.Path) points towards the folder
        where the data is stored
        """
        self.path_to_data_record = path_do_dataset.joinpath('data_records.csv')
        if not self.path_to_data_record.is_file():

        self.data_record = pd.read_csv(path_to_data_record)
