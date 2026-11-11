# Scripts

Complete programs that use LION, outside the library itself:

- **`example_scripts/`**: short scripts, one feature of LION each (supervised
  and self-supervised training, unrolled methods, PnP, evaluation, resuming,
  publishing, your own pieces, photocurrent mapping). Their README says what
  each shows and what it needs. For a step-by-step introduction, see the
  tutorials (`demos/`, on the [site](https://cambridgecia.github.io/LION/)).
- **`paper_scripts/`**: scripts that reproduce published papers with LION,
  one folder per paper, each with a README citing it. A script that trains a
  paper's model in another way than the paper's is an example script instead.

## Writing a script

As the example scripts do:

- the licence header, then a docstring: what the script does, why, and what
  it needs (a GPU, data, `LION_RESULTS_PATH`);
- `# %%` cells, one per step, each with a comment saying what it does;
- a settings block at the top (epochs, data size, GPU, `verbose`), so a short
  run and the real one differ only there;
- results under `results_path(...)` (`LION.utils.paths`), never a fixed path;
- LION's experiments and data loaders for the data, so the models' records say
  what they were trained on;
- commented-out alternatives where they help the reader (another model,
  another algorithm).
