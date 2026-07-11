# Reconstruction tuning reproduction

This is the minimal, non-run-specific record of the validation tuning used by
PaDIS-Reproduction. The selected settings are stored in
[`../config/reconstruction_hparam_defaults.json`](../config/reconstruction_hparam_defaults.json).
No test-set result is used here.

## Protocol

Every row used the corrected validation split, seed 33, LION geometry, one
image per experiment and validation index 4. Unless the experiment column says
otherwise, candidates were compared jointly on `ct_20` and `ct_8`. Diffusion
searches fixed the trained noise range and schedule while varying only the
listed sampler controls. Candidate tuples in braces form a Cartesian product;
explicit parenthesised tuples do not.

For diffusion models, the reproduction launcher uses `min_intense_val`, the
checkpoint family used for final inference. This intentionally merges the
near-identical searches previously made against an intensive-validation
checkpoint and a final six-hours-before-completion checkpoint. PnP uses its
actual minimum-validation checkpoint. Thus checkpoint timing is not another
tuned variable.

## Attempted values

| Implementation | Method | Model/prior | Experiment | Attempted hyperparameter values |
|---|---|---|---|---|
| LION | CP (inference identifier `cp_tv`) | patch | `ct_20,ct_8` | `lambda={0.0005,0.001,0.002}`, 1000 iterations |
| LION | PnP | patch | `ct_20,ct_8` | `(eta,iterations)={(5e-6,100),(1e-5,40),(1e-5,60),(1e-5,100),(2e-5,40),(2e-5,60),(2e-5,100),(3e-5,40),(3e-5,60),(3e-5,100),(5e-5,60),(5e-5,100)}`; then noise `{0.01,0.03,0.05}` at `eta=3e-5`, 60 iterations |
| LION | PaDIS-DPS | patch/default | `ct_20,ct_8` | `zeta={3.5,3.75,4,4.25}`, `epsilon={0.3,0.5}`; initialization/clipping follow-up selected Gaussian initialization with no initial/output clipping |
| Public-compatible | PaDIS-DPS | patch/default | `ct_20,ct_8` | `zeta={0.15,0.2}`, `epsilon={0.5,0.75}` |
| Paper | PaDIS-DPS | patch/default | `ct_20,ct_8` | `zeta={0.0075,0.01,0.015}`, `epsilon={0.5,1}` |
| LION | Langevin | patch/default | `ct_20,ct_8` | `zeta={3.5,4,4.5}`, `epsilon={0.5,0.75}` |
| Public-compatible | Langevin | patch/default | `ct_20,ct_8` | `zeta={0.2,0.3}`, `epsilon={0.5,0.75}` |
| Paper | Langevin | patch/default | `ct_20,ct_8` | `zeta={0.01,0.03}`, `epsilon={0.5,0.75}` |
| LION | PC | patch/default | `ct_20,ct_8` | `zeta={3.75,4,4.25,4.5,4.75}`, `r={0.01,0.015}` |
| Public-compatible | PC | patch/default | `ct_20,ct_8` | `zeta=0.5`, `r={0.08,0.16}` |
| Paper | PC | patch/default | `ct_20,ct_8` | `zeta={0.01,0.02,0.03}`, `r={0.04,0.08,0.16}` |
| LION | VE-DDNM | patch/default | `ct_20,ct_8` | sampling `epsilon={0.05,0.1,0.2}`; noise scale `{0,0.5,1.5}` at `epsilon=0.1`; corrected-state clipping diagnostics |
| Public-compatible | VE-DDNM | patch/default | `ct_20,ct_8` | sampling `epsilon={0.05,0.1,0.2}` |
| Paper | VE-DDNM | patch/default | `ct_20,ct_8` | sampling `epsilon={0.05,0.1,0.2}`; corrected-state clipping diagnostics |
| LION | Patch average | patch/default | `ct_20,ct_8` | `zeta={3.5,4}`, `epsilon=0.5` |
| Paper | DPS | whole/default | `ct_20,ct_8` | `zeta={0.0075,0.01,0.015}`, `epsilon=0.5` |
| LION | Langevin | whole/default | `ct_20` | `zeta={3.5,4,4.5}`, `epsilon={0.5,0.75}` |
| LION | PC | whole/default | `ct_20` | `zeta={3.75,4,4.25,4.5,4.75}`, `r={0.01,0.015}` |
| LION | VE-DDNM | whole/default | `ct_20` | sampling `epsilon={0.05,0.1,0.2}` |
| Public-compatible | PaDIS-DPS | patch/native-512 | `ct_512_60` | `(zeta,epsilon)={(0.4,0.5),(0.5,0.5),(0.8,0.5),(1.2,0.5),(1.6,0.5),(1.6,0.75)}` |
| LION | PaDIS-DPS | patch/native-512 | `ct_512_60` | `zeta={0.8,1.2,1.6,2}`, `epsilon=0.5` |
| LION | CP (inference identifier `cp_tv`) | patch/native-512 | `ct_512_60` | `lambda={0.001,0.002}`, 1000 iterations |
| LION | PaDIS-DPS | patch/full | `ct_20` | `zeta={2,4.5,4.75,5}`, `epsilon=0.5`, Gaussian initialization, no clipping |
| LION | DPS | whole/full | `ct_20` | `zeta={4,5}`, `epsilon=0.5` |

### Inheritance

Combinations not independently tuned inherit as follows:

| Target | Inherits from |
|---|---|
| Higher-view 256 experiments | Same implementation, method, model/prior and dataset-size `ct_20` setting |
| Full-data experiments other than the two explicit full-data DPS rows | Matching default-data implementation/method/prior setting |
| Patch stitch/fixed-overlap | Matching patch PaDIS-DPS sampler setting; overlap mechanics are not sampler hyperparameters |
| Whole-image public-compatible rows | Matching patch public-compatible setting when no whole-image row exists |
| Native-512 methods other than the explicit PaDIS-DPS and CP rows | Matching default-resolution method/implementation setting |
| Different diffusion checkpoint timestamps for one trained model | The row above evaluated with `min_intense_val`; timestamp is not treated as a new combination |

## Run the complete scheme

Point `--training-root` at a completed GCP or Slurm training root in the layout
accepted by the reconstruction matrix:

```bash
conda run -n lion-dev python -u \
  scripts/paper_scripts/PaDIS-Reproduction/tuning/PaDIS_run_reproduction_tuning.py \
  --training-root /path/to/completed/training-root \
  --output-root /path/to/tuning-output
```

The launcher is resumable: existing `metrics.json` files are reused. Add
`--dry-run` to inspect every concrete command, `--only full_patch,full_whole`
to select rows, or `--rerun-existing` to deliberately replace completed
results. Each sweep retains the tuner's `manifest.json`, `runs.jsonl`,
`summary.json`, `summary.csv`, and per-command log, so the exact executed grid
and ranking inputs remain auditable.

The launcher deliberately forces `--checkpoint-policy min_intense_val` and
does not stage historical externally named checkpoints. This prevents original
run names or machine-specific paths from becoming part of the reproduction
procedure.
