# Example scripts

Short, complete programs, one feature of LION each. For a step-by-step
introduction to the library, see the demos (`demos/`).

Run them from the repository root, in order (some use the models of earlier
ones), e.g. `python scripts/example_scripts/01_supervised_training.py`. Each
one has a `# %%` cell per step, so an editor that knows cells (VS Code,
Spyder, Jupyter) runs them a step at a time.

They need:

- a GPU: LION's CT operators project with tomosipo and ASTRA's CUDA code;
- the processed LIDC-IDRI data in `LION_DATA_PATH` (see
  [Datasets](https://cambridgecia.github.io/LION/datasets.html));
- `LION_RESULTS_PATH`: they write to `$LION_RESULTS_PATH/examples/`.

By default they train a few epochs on a few hundred slices, to show they work.
Change the settings at the top for real training (more epochs, all the data).

| Script | What it shows |
|---|---|
| `01_supervised_training.py` | Supervised training of FBPConvNet: an experiment, a model, an optimizer and a learning-rate scheduler, validation (keeping the best model), checkpoints, saving. Any supervised model is one line away. |
| `02_learned_primal_dual.py` | An unrolled method, LPD: a model that takes the measurements and trains through the CT operator; changing a model's parameters. |
| `03_self_supervised_noise2inverse.py` | Self-supervised training with Noise2Inverse (a U-Net), without clean images; the other self-supervised solvers are used the same way. |
| `04_pnp_denoiser.py` | Plug-and-Play: a DRUNet trained as a Gaussian denoiser, then the prior of a PnP-ADMM reconstruction, compared with FDK. |
| `05_evaluate_and_compare.py` | Evaluating methods on an experiment's test data, and a comparison (tables, scores, figures) of the models of 01, 02 and 04 with SIRT, against FDK. |
| `06_resume_and_fine_tune.py` | A stopped training resumed from its checkpoint; a trained model fine-tuned on another problem, its record naming its original. |
| `07_publish_and_reproduce.py` | Publishing a trained model with its records, then making it again, and training it again with a change, from the package alone. |
| `08_your_own_pieces.py` | Your own torch model (`TorchModel`), forward operator (a blur) and dataset, trained with LION's solver. |
| `photocurrent_mapping.py` | Photocurrent mapping, a modality other than CT: compressed-sensing baselines (FISTA, SPGL1) and PnP-ADMM on subsampled measurements. Needs the PCM data in `data_dir`, at the top of the file. |
