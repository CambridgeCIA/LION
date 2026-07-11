Reconstruction Matrix
=====================

The matrix generator combines method, implementation, model, geometry,
experiment, and trained-ablation records.  The desired full configuration
currently contains 109 jobs.

List jobs without running them::

   python scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_run_reconstruction_matrix.py \
       --training-root /path/to/training \
       --output-root /path/to/reconstruction \
       --methods all --models method_default --experiments paper_matrix \
       --ablations all --implementations method_default --geometries lion \
       --list

Hyperparameters
---------------

Final settings live in
``config/reconstruction_hparam_defaults.json``.  Selection keys include model
identity so full-dataset settings cannot leak into default-data or ablation
rows.  High-view experiments may inherit explicitly declared lower-view
settings where validation showed that inheritance was appropriate.

Verification
------------

The verifier discovers ``metrics.json`` records, canonicalises legacy names,
checks expected sampler and method settings, calculates bootstrap uncertainty,
and writes both JSON and CSV summaries.  Supplying the exact jobs manifest is
the strongest completeness check.
