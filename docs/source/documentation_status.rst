Documentation Status
====================

.. warning::

   This is the first structured LION API documentation release.  The PaDIS,
   LIDC-IDRI, and CT paths used by the reproduction study are actively
   documented.  Other long-standing modules are indexed provisionally and may
   expose incomplete docstrings or undocumented behaviour.

Documented in this release
--------------------------

- PaDIS loss, NCSN++ model integration, training solver, and reconstructor
- LIDC-IDRI loading, preprocessing, normalised-intensity/HU conversion, and CT
  experiment extensions used by PaDIS
- reconstruction, generation, tuning, reporting, GCP, and Slurm workflows
- hardware-aware local and CI testing

Provisional areas
-----------------

The following areas are useful but have not yet received the same narrative
and API audit:

- older learned reconstruction model families
- non-LIDC datasets beyond the interfaces required by current tests
- legacy notebooks and paper-specific scripts outside PaDIS-Reproduction
- several general-purpose optimisers and reconstructors inherited from the
  upstream project

Pages for these areas are deliberately labelled as stubs rather than implying
complete documentation.  Contributions should replace warnings with tested
examples and precise NumPy-style docstrings as each subsystem is audited.
