## Normalization in CT is non trivial. This class contains info about the type of normalization applied,
# and is useful inside models, so they can communicate how they need their inputs normalized

import numpy as np
import torch


def _nonzero(data_range):
    """The range to divide by: 1 where it is 0 (a constant image maps to 0)."""
    if isinstance(data_range, torch.Tensor):
        return torch.where(data_range == 0, torch.ones_like(data_range), data_range)
    return data_range if data_range != 0 else 1


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

        if self.type == "dataset" and dataset is None and data_range is not None:
            # the dataset's range, computed before (e.g. read from a model file)
            self.xmin, self.xmax = float(data_range[0]), float(data_range[1])
        elif self.type == "dataset":
            assert (
                dataset is not None
            ), "Dataset must be provided for dataset normalization"
            xmax = -np.inf
            xmin = np.inf
            # by index: iterating a Dataset relies on __getitem__ raising
            # IndexError at the end, which not every dataset does (LIDC-IDRI)
            for i in range(len(dataset)):
                gt = dataset[i][1]
                xmax = max(float(gt.max()), xmax)
                xmin = min(float(gt.min()), xmin)
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

    def record(self) -> dict:
        """What makes this normalisation again (``from_record``): its type,
        and its range for "dataset" and "custom"."""
        if self.type in ("dataset", "custom"):
            return {"type": self.type, "range": [float(self.xmin), float(self.xmax)]}
        return {"type": self.type, "range": None}

    @staticmethod
    def from_record(record) -> "Normalisation":
        """The normalisation ``record()`` describes (without its dataset); a
        dict or a ``LIONParameter``."""
        if not isinstance(record, dict):
            record = vars(record)
        return Normalisation(record["type"], data_range=record.get("range"))

    def normalise(self, x):
        # assume x is a torch tensor
        if self.type == "sample":
            # normalize each sample in batch separately
            self.last_max = x.amax(dim=(2, 3), keepdim=True)
            self.last_min = x.amin(dim=(2, 3), keepdim=True)
            return (x - self.last_min) / _nonzero(self.last_max - self.last_min)
        if self.type == "dataset" or self.type == "custom":
            return (x - self.xmin) / _nonzero(self.xmax - self.xmin)
        if self.type == "none":
            return x  # return input if no normalization is set

    def normalise_measurements(self, y, op=None):
        """The measurements of the normalised images, by linearity: if images
        are mapped to (x - min) / s, measurements y = A x are mapped to
        (y - A(min)) / s, A(min) being the measurements of an image of value
        min everywhere (computed with ``op``; none needed when min is 0). For
        models that take measurements; the operator is unchanged. Not defined
        for "sample" (an image's range isn't known before reconstructing)."""
        if self.type == "none":
            return y
        if self.type == "sample":
            raise ValueError(
                "a 'sample' normalisation can't be applied to measurements: an "
                "image's range isn't known before reconstructing it"
            )
        if self.xmin != 0:
            if op is None:
                raise ValueError(
                    "normalising measurements with a range that doesn't start "
                    "at 0 needs the operator"
                )
            constant = torch.full(
                tuple(op.domain_shape), float(self.xmin), dtype=y.dtype, device=y.device
            )
            y = y - op(constant)
        return y / _nonzero(self.xmax - self.xmin)

    def unnormalise(self, x, reference=None):
        """Undo ``normalise``. ``reference``: the input that was normalised
        (a "sample" normalisation maps ``x`` back to its range; without it,
        to the range of the last ``normalise`` call)."""
        if self.type == "dataset" or self.type == "custom":
            return (x * (self.xmax - self.xmin)) + self.xmin
        if self.type == "sample":
            if reference is None:
                return (x * (self.last_max - self.last_min)) + self.last_min
            low = reference.amin(dim=(2, 3), keepdim=True)
            high = reference.amax(dim=(2, 3), keepdim=True)
            return x * (high - low) + low
        if self.type == "none":
            return x
