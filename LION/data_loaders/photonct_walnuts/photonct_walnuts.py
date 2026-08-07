import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from LION.utils.paths import PHOTONCT_WALNUTS_DATASET_PATH

class PhotonCTWalnutDataset(Dataset):
    """
    PhotonCT Walnuts Dataset Loader.
    """
    def __init__(
        self,
        walnut_index: int,
        energy_bin: str = 'Total',
        transform=None,
    ):
        """
        Args:
            walnut_index: 1 to 15
            energy_bin: 'High', 'Low', or 'Total'
            transform: Optional transform to be applied on a sample.
        """
        assert 1 <= walnut_index <= 15
        assert energy_bin in ['High', 'Low', 'Total']
        
        self.walnut_index = walnut_index
        self.energy_bin = energy_bin
        self.transform = transform
        self.n_channel = 2063
        self.n_slice = 505
        
        self.walnut_path = PHOTONCT_WALNUTS_DATASET_PATH / f"Walnut_{walnut_index}"
        self.calibration_path = PHOTONCT_WALNUTS_DATASET_PATH / "CalibrationTable"
        
        if self.energy_bin == 'Low':
            self.file_paths_total = sorted(glob.glob(os.path.join(self.walnut_path, "Total", "*.raw")))
            self.file_paths_high = sorted(glob.glob(os.path.join(self.walnut_path, "High", "*.raw")))
            assert len(self.file_paths_total) > 0 and len(self.file_paths_total) == len(self.file_paths_high), "Missing or mismatched raw files for Total/High."
            self.length = len(self.file_paths_total)
        else:
            self.file_paths = sorted(glob.glob(os.path.join(self.walnut_path, self.energy_bin, "*.raw")))
            assert len(self.file_paths) > 0, f"No raw files found for {self.energy_bin}"
            self.length = len(self.file_paths)
            
        air_table_path = self.calibration_path / f"air_table_{self.energy_bin.lower()}.raw"
        assert air_table_path.exists(), f"Air table not found at {air_table_path}"
        self.air_table = np.fromfile(air_table_path, dtype=np.uint16)
        self.air_table = self.air_table.reshape((self.n_channel, self.n_slice), order='F').astype(np.float32)

    def __len__(self):
        return self.length

    def _load_raw(self, path):
        proj = np.fromfile(path, dtype=np.uint16)
        proj = proj.reshape((self.n_channel, self.n_slice), order='F').astype(np.float32)
        return proj

    def __getitem__(self, idx):
        if self.energy_bin == 'Low':
            proj_total = self._load_raw(self.file_paths_total[idx])
            proj_high = self._load_raw(self.file_paths_high[idx])
            proj = proj_total - proj_high
        else:
            proj = self._load_raw(self.file_paths[idx])
            
        proj = -np.log(np.maximum(proj, 1e-6)) + np.log(np.maximum(self.air_table, 1e-6))
        
        proj_tensor = torch.from_numpy(proj).unsqueeze(0)
        
        if self.transform:
            proj_tensor = self.transform(proj_tensor)
            
        return proj_tensor
