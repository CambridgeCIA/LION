## Normalization in CT is non trivial. This class contains info about the type of normalization applied,
# and is useful inside models, so they can communicate how they need their inputs normalized

import numpy as np
import torch


class Normalisation:
    def __init__(self, normalisation_type="none", dataset=None, data_range=None):
        """
        Normalization in CT is non trivial. This class contains info about the type of normalization applied,
        and is useful inside models, so they can communicate how they need their inputs normalized.

        normalisation_type: The type of normalization to apply (e.g. "sample", "dataset").
            "none": skip
            "sample": will normalize inputs one by one
            "dataset": will normalize inputs across the entire dataset.
            "custom": will normalize inputs using a custom data_range. Requires data_range input.
        """
        # There are several ways to normalize CT data.

        assert normalisation_type in [
            "none",
            "sample",
            "dataset",
            "custom",
        ], "Invalid normalization type"
        self.type = normalisation_type

        if self.type == "dataset":
            assert (
                dataset is not None
            ), "Dataset must be provided for dataset normalization"
            xmax = -np.inf
            xmin = np.inf
            for _, gt in dataset:
                xmax = max(gt.max(), xmax)
                xmin = min(gt.min(), xmin)
            self.xmin = xmin
            self.xmax = xmax
        if self.type == "custom":
            assert (
                data_range is not None
            ), "Range must be provided for custom normalization"
            assert len(data_range) == 2, "Range must be a tuple of (min, max)"
            self.xmin = data_range[0]
            self.xmax = data_range[1]

        self.last_max = np.inf
        self.last_min = -np.inf

    def normalise(self, x):
        # assume x is a torch tensor
        if self.type == "sample":
            # normalize each sample in batch separately
            self.last_max = x.max(dim=(2, 3), keepdim=True)[0]
            self.last_min = x.min(dim=(2, 3), keepdim=True)[0]
            return (x - self.last_min) / (self.last_max - self.last_min)
        if self.type == "dataset" or self.type == "custom":
            return (x - self.xmin) / (self.xmax - self.xmin)
        if self.type == "none":
            return x  # return input if no normalization is set

    def unnormalise(self, x, target=None):
        if self.type == "dataset" or self.type == "custom":
            return (x * (self.xmax - self.xmin)) + self.xmin
        if self.type == "sample":
            if target is None:
                return (x * (self.last_max - self.last_min)) + self.last_min
            else:
                return (
                    x * (target.max() - target.min(dim=(2, 3), keepdim=True)[0])
                ) + target.min(dim=(2, 3), keepdim=True)[0]
        if self.type == "none":
            return x
