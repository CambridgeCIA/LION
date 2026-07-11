PaDIS Concepts
==============

Patch diffusion prior
---------------------

PaDIS trains a denoiser over randomly located image patches.  Optional
position channels encode each patch location in the full image.  At inference,
shifted patch partitions produce a whole-image score without requiring a
whole-image diffusion network.

Noise schedule
--------------

Most reported reconstruction experiments use 100 geometrically spaced noise
levels from ``sigma_max=10`` to ``sigma_min=0.002`` and 10 inner updates per
level.  Method-specific exceptions are recorded in the checked-in
hyperparameter registry and the PaDIS-Reproduction README.

Implementation conventions
--------------------------

``paper``
   Follows the CT schedule and equations described by Hu et al.

``public_repo``
   Reproduces the released repository mechanics where these differ from the
   publication.

``lion_physics``
   Uses LION CT operators and operator-normalised data consistency without
   fitted public-repository scaling constants.

Corrected identifiers
---------------------

The Chambolle--Pock TV implementation is stored as ``cp_tv`` and displayed as
**CP** with prior **TV**.  The limited-angle acquisition is stored as
``ct_20_limited_angle_120``.  Legacy ``admm_tv`` and ``ct_fanbeam_180`` values
remain accepted when reading commands and historical results.
