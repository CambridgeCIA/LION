# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Maximilian B. Kiss, Ander Biguri
# =============================================================================
"""Tests the noise paper's models: its Table 2 (and the noise of Table 1).

Every trained model (``train.py``; the ones with the lowest validation loss,
as the paper) on both test sets, simulated and experimental noise, with SSIM
and PSNR on each slice's own range (mean ± standard deviation):

- **sinogram domain**: the sinogram denoisers' outputs against the mode 2
  sinograms, with the noisy sinograms themselves for reference;
- **reconstruction domain**: the FDK of the denoised sinograms, and the
  end-to-end models' outputs, against the mode 2 reconstructions, with the
  FDK of the noisy sinograms for reference.

    python evaluate.py
    python evaluate.py --test experimental --domain sinogram

Writes ``$LION_RESULTS_PATH/paper_scripts/noise_paper/comparisons/<domain>_
<test noise>``: the table (markdown, LaTeX, CSV), every slice's scores, box
plots and images, and the test records; and ``tables.md``, all of them
together. Models not trained yet are left out, with a message. ``--samples``
tests on fewer slices, evenly spaced; ``--final`` uses the models after the
last epoch instead.

Needs a GPU, the processed 2DeteCT data in ``LION_DATA_PATH`` and
``LION_RESULTS_PATH``.
"""

# %% Imports
import argparse

import torch

from LION.evaluation import Comparison, Evaluation
from LION.metrics.psnr import PSNR
from LION.metrics.ssim import SSIM

import settings

# %% Settings
DOMAINS = ("sinogram", "reconstruction")
arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
arguments.add_argument("--test", choices=[*settings.NOISE, "all"], default="all")
arguments.add_argument("--domain", choices=[*DOMAINS, "all"], default="all")
arguments.add_argument(
    "--samples", type=int, default=None, help="test slices (default: all)"
)
arguments.add_argument("--final", action="store_true", help="the final models")
arguments.add_argument("--gpu", type=int, default=0)
arguments = arguments.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(arguments.gpu)
verbose = 1  # 0: nothing; 1: a progress bar per method
suffix = "" if arguments.final else "_best"
folder = settings.folder()


def trained(model_name, noise):
    """The file of a trained model, or None (with a message) if there is none."""
    name = settings.name(model_name, noise)
    file = folder / f"{name}{suffix}"
    if file.with_suffix(".json").exists():
        return file
    print(f"{name}{suffix}: not trained yet, left out")
    return None


# %% One domain, one test set: the methods, the comparison
def evaluate(domain, test_noise):
    task = "sino2sino" if domain == "sinogram" else "sino2recon"
    evaluation = Evaluation(
        dataset=settings.dataset(test_noise, task, "test"),
        operator=settings.operator_params().make_operator(),
        # SSIM and PSNR, each on its target's own range (as the paper's)
        metrics={"SSIM": SSIM(), "PSNR": PSNR()},
        normalisation="sample",
        samples=arguments.samples,
        save_folder=folder / "evaluations",
    )
    if domain == "sinogram":
        methods = {"noisy": settings.noisy}
        for model_name in ("UNet", "MSDNet"):
            for noise in settings.NOISE:
                file = trained(model_name, noise)
                if file is not None:
                    methods[f"{model_name}, trained on {noise}"] = file
    else:
        methods = {"FBP": settings.fbp}
        for model_name in ("UNet", "MSDNet"):
            for noise in settings.NOISE:
                file = trained(model_name, noise)
                if file is not None:
                    methods[
                        f"FBP of {model_name}, trained on {noise}"
                    ] = settings.denoised_then_fbp(file)
        for model_name in ("FBP+UNet", "FBP+MSDNet"):
            for noise in settings.NOISE:
                file = trained(model_name, noise)
                if file is not None:
                    methods[f"{model_name}, trained on {noise}"] = file
    # every method in the table; none is the reference
    comparison = Comparison(evaluation, methods, baseline=False, allow=settings.ALLOW)
    comparison.run(verbose=verbose)
    comparison.save(folder / "comparisons" / f"{domain}_{test_noise}")
    return comparison.table()


# %% Evaluate what was asked for
domains = list(DOMAINS) if arguments.domain == "all" else [arguments.domain]
tests = list(settings.NOISE) if arguments.test == "all" else [arguments.test]
tables = []
for domain in domains:
    for test_noise in tests:
        table = evaluate(domain, test_noise)
        title = f"{domain} domain, {test_noise} test set"
        print(f"\n{title}\n{table}")
        tables.append(f"## {title}\n\n{table}\n")
summary = folder / "tables.md"
summary.write_text("# Noise paper, Table 2\n\n" + "\n".join(tables))
print(f"All: {summary}")
