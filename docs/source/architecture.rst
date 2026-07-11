Architecture
============

Package layers
--------------

``LION.CTtools``
   Geometry definitions, CT operator construction, noise models, and unit
   conversions.

``LION.data_loaders``
   Dataset adapters and preprocessing utilities.  Dataset parameters carry the
   geometry and task definition into training or reconstruction.

``LION.experiments``
   Reproducible bundles of geometry, noise, and split-specific dataset
   configuration.

``LION.models`` and ``LION.losses``
   Learned priors and reconstruction networks together with their training
   objectives.  The PaDIS implementation uses a position-aware NCSN++ denoiser
   and an EDM-style patch loss.

``LION.optimizers``
   Training orchestration, validation, checkpointing, and resumption.  Despite
   the historical package name, these classes coordinate complete model
   training rather than exposing only numerical optimisers.

``LION.reconstructors``
   Classical and learned inverse solvers.  Reconstructors consume a geometry or
   operator, measurements, model state, and algorithm parameters.

PaDIS data flow
---------------

For PaDIS training, an :class:`~LION.experiments.ct_experiments.Experiment`
constructs an LIDC-IDRI image-prior dataset.  A
:class:`~LION.optimizers.PaDISSolver.PaDISSolver` samples patches and noise
levels, evaluates :class:`~LION.losses.PaDIS.PaDISDenoisingLoss`, updates the
NCSN++ model, and maintains exponential-moving-average checkpoints.

At inference time :class:`~LION.reconstructors.PaDIS.PaDIS` assembles a
whole-image score from patches and combines it with CT data consistency.  The
reproduction scripts select the sampler convention, tuned hyperparameters,
checkpoint, and experiment matrix without duplicating the underlying solver.

Design boundaries
-----------------

The implementation keeps three concerns separate:

- physical acquisition geometry belongs to LION operators and experiments;
- prior parameterisation belongs to model and checkpoint metadata;
- study-specific scheduling and output layout belong to reproduction scripts.

This separation is important when comparing the public PaDIS implementation,
the equations described by Hu et al., and LION-native physics scaling.
