# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
Data and experiments
====================

An **experiment** fixes a problem: the geometry, the noise, the dataset and
its splits. Every method trained or tested on the same experiment sees the
same data, so methods can be compared. Experiments give **datasets**, torch
datasets of (measurements, target) pairs, and a **record** of each: what the
data are, written so that they can be made again.

Previous: :doc:`d02_parameters_and_specs`. Next: models (d04).

Needs a GPU and the processed LIDC-IDRI data in ``LION_DATA_PATH`` (see
``LION/data_loaders/LIDC_IDRI/``).
"""

# %%
# The experiments
# ---------------
# ``LION.experiments.ct_experiments`` has LION's simulated CT problems on
# LIDC-IDRI: clinical, low and extremely low dose, each with full, sparse-view
# and limited-angle sampling. Their doses are photons per detector pixel:

import matplotlib.pyplot as plt
import torch

from LION.experiments import ct_experiments

print(
    ct_experiments.CLINICAL_I0,
    ct_experiments.LOW_DOSE_I0,
    ct_experiments.EXTREMELY_LOW_DOSE_I0,
)

experiment = ct_experiments.LowDoseCTRecon()
print(experiment.experiment_params)  # name, geometry, noise, data loader

# %%
# Its geometry is the operator's parameters (:doc:`d01_operators`), the one the
# sinograms are made with and that models take:

print(experiment.operator_params)

# %%
# Datasets
# --------
# ``get_dataset("train")``, ``"validation"`` or ``"test"``: the splits are by
# patient, so no patient is in two. A sample is (noisy sinogram, image), on
# LION's device.

training = experiment.get_dataset("train")
validation = experiment.get_dataset("validation")
test = experiment.get_dataset("test")
print(len(training), len(validation), len(test))

sinogram, image = training[0]
print(sinogram.shape, image.shape, image.device)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(sinogram[0].cpu(), cmap="gray", aspect="auto")
axes[0].set_title("noisy sinogram")
axes[1].imshow(image[0].cpu(), cmap="gray")
axes[1].set_title("image (density, g/cm³)")
for ax in axes:
    ax.axis("off")

# %%
# The noise is drawn from a seed and the slice: a slice has one noisy
# measurement, the same each epoch and in every run, as a scan is measured
# once.

print(torch.equal(training[0][0], training[0][0]))

# %%
# A dataset is a torch dataset: use it with a ``DataLoader``.

from torch.utils.data import DataLoader

loader = DataLoader(training, batch_size=4, shuffle=True)
sinograms, images = next(iter(loader))
print(sinograms.shape, images.shape)

# %%
# Records
# -------
# A dataset's record says what it is: the dataset's spec, the noise, the
# experiment and split, and which samples (by patient and slice), so that the
# same data can be made again and evaluations can check that no test patient
# was trained on.

record = experiment.data_record("train")
print(record.split, record.samples.count, record.samples.ids[:3])
print(record.samples.groups[:5])  # the patients

# %%
# A ``torch.utils.data.Subset`` of a LION dataset is recorded too, by its
# indices: training on part of the data is still reproducible.

from torch.utils.data import Subset

from LION.data_loaders.data_record import data_record

few = Subset(training, range(10))
print(data_record(few).subset.indices)

# %%
# Changing an experiment
# ----------------------
# An experiment's parameters can be changed like any others, e.g. another dose:

parameters = ct_experiments.LowDoseCTRecon.default_parameters()
parameters.noise_params = ct_experiments.Experiment.noise_parameters(I0=5000)
experiment_5000 = ct_experiments.LowDoseCTRecon(parameters)
print(experiment_5000.experiment_params.noise_params.I0)

# %%
# The other experiments are made the same way:

for name in ["clinicalCTRecon", "SparseAngleLowDoseCTRecon", "LimitedAngleCTRecon"]:
    e = getattr(ct_experiments, name)()
    print(name, len(e.operator_params.angles), "angles")

# %%
# Measured data: 2DeteCT
# ----------------------
# The same experiments run on 2DeteCT (``dataset="2DeteCT"``), real scans of
# 109 objects, where the measured sinograms take the place of simulated ones;
# ``LION.experiments.deteCT_benchmark`` has the 2DeteCT paper's own. They need
# the 2DeteCT data:
#
# .. code-block:: python
#
#    measured = ct_experiments.LowDoseCTRecon(dataset="2DeteCT")
#
# The loaders
# -----------
# Under the experiments are the datasets themselves (``LION.data_loaders``):
# ``LIDC_IDRI`` and ``deteCT``, with their own parameters, e.g. other tasks
# than reconstruction (segmentation, diagnosis):

from LION.data_loaders.LIDC_IDRI import LIDC_IDRI

print(LIDC_IDRI.default_parameters())
print(LIDC_IDRI.cite("MLA"))
