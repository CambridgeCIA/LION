# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Maximilian B. Kiss, Ander Biguri
# =============================================================================
"""The noise paper's Table 1: how the simulated noise compares with the
experimental one, for I0 = 200, 250, 300 and 350.

The noisy test sinograms against the mode 2 ones (sinogram domain), and their
FDK against the mode 2 reconstructions (reconstruction domain), with SSIM and
PSNR on each slice's own range: the experimental noise (mode 1), and the
simulated noise at each I0, by both noise models (``settings.Noisy2DeteCT``).
The model whose numbers match the paper's Table 1 is the one that produced
its data; set it as ``settings.NOISE_MODEL``.

    python noise_check.py
    python noise_check.py --samples 50

Writes ``$LION_RESULTS_PATH/paper_scripts/noise_paper/table1.md``, with the
paper's numbers for comparison. Needs a GPU and the processed 2DeteCT data in
``LION_DATA_PATH``; no trained model.
"""

# %% Imports
import argparse

import torch

from LION.evaluation import Evaluation, table
from LION.metrics.psnr import PSNR
from LION.metrics.ssim import SSIM

import settings

# %% Settings
arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
arguments.add_argument(
    "--samples", type=int, default=None, help="test slices (default: all)"
)
arguments.add_argument("--gpu", type=int, default=0)
arguments = arguments.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(arguments.gpu)
verbose = 1

PAPER = """| Noise | Sinogram SSIM | Sinogram PSNR | Reconstruction SSIM | Reconstruction PSNR |
|---|---|---|---|---|
| experimental | 0.2658 ± 0.0963 | 19.81 ± 4.66 | 0.1899 ± 0.0987 | 21.80 ± 3.60 |
| simulated, I0 = 200 | 0.2965 ± 0.0409 | 25.72 ± 0.86 | 0.1364 ± 0.0307 | 21.45 ± 1.83 |
| simulated, I0 = 250 | 0.3448 ± 0.0435 | 26.66 ± 0.86 | 0.1627 ± 0.0356 | 22.40 ± 1.83 |
| simulated, I0 = 300 | 0.3854 ± 0.0451 | 27.42 ± 0.87 | 0.1865 ± 0.0397 | 23.17 ± 1.83 |
| simulated, I0 = 350 | 0.4201 ± 0.0459 | 28.05 ± 0.87 | 0.2083 ± 0.0432 | 23.81 ± 1.83 |
"""


# %% The noisy data of one setting, in one domain
def scored(domain, name, noise, **noise_settings):
    task = "sino2sino" if domain == "sinogram" else "sino2recon"
    evaluation = Evaluation(
        dataset=settings.dataset(noise, task, "test", **noise_settings),
        operator=settings.operator_params().make_operator(),
        metrics={"SSIM": SSIM(), "PSNR": PSNR()},
        normalisation="sample",
        samples=arguments.samples,
        save_folder=settings.folder() / "evaluations",
    )
    reconstruct = settings.noisy if domain == "sinogram" else settings.fbp
    record = evaluation.run(reconstruct, name, allow=settings.ALLOW, verbose=verbose)
    record.name = name  # a record saved before (e.g. by evaluate.py) has its own
    return record


# %% Every setting, both domains
settings_to_check = [("experimental", "experimental", {})]
for noise_model in ("relative", "paper"):
    for I0 in settings.TABLE1_I0:
        settings_to_check.append(
            (
                f"simulated, I0 = {I0}, {noise_model} model",
                "simulated",
                {"I0": I0, "noise_model": noise_model},
            )
        )

tables = []
for domain in ("sinogram", "reconstruction"):
    records = [
        scored(domain, name, noise, **noise_settings)
        for name, noise, noise_settings in settings_to_check
    ]
    tables.append(f"## {domain} domain\n\n{table(records)}\n")
    print(f"\n{domain} domain\n{table(records)}")

summary = settings.folder() / "table1.md"
summary.parent.mkdir(parents=True, exist_ok=True)
summary.write_text(
    "# Noise paper, Table 1\n\n" + "\n".join(tables) + "\n## The paper's\n\n" + PAPER
)
print(f"\nThe paper's:\n{PAPER}\nAll: {summary}")
