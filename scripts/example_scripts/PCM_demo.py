# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: lion_proposed_no_pytorch_channel
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
print("Importing torch...")
import torch

device = "cuda:0"
torch.set_default_device(device)
print(f"Using device: {device}")

print("Setting up dataset...")
from LION.experiments import ct_experiments

experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")
lidc_dataset = experiment.get_training_dataset()

from scripts.example_scripts.PCM_demo_main import run_demo

# %%
print(f"Running PCM demo with 25% sampling rate and 1/4 in-order measurements...")
run_demo(dataset=lidc_dataset, subtract_from_J=1, delta_divided_by=4)
print(f"Running PCM demo with 25% sampling rate and 1/16 in-order measurements...")
run_demo(dataset=lidc_dataset, subtract_from_J=2, delta_divided_by=4)

# %%
print(f"Running PCM demo with 12.5% sampling rate and 1/8 in-order measurements...")
run_demo(dataset=lidc_dataset, subtract_from_J=1, delta_divided_by=8)
print(f"Running PCM demo with 12.5% sampling rate and 1/32 in-order measurements...")
run_demo(dataset=lidc_dataset, subtract_from_J=2, delta_divided_by=8)

# %%
print(f"Running PCM demo with 6.25% sampling rate and 1/16 in-order measurements...")
run_demo(dataset=lidc_dataset, subtract_from_J=2, delta_divided_by=16)
print(f"Running PCM demo with 6.25% sampling rate and 1/64 in-order measurements...")
run_demo(dataset=lidc_dataset, subtract_from_J=3, delta_divided_by=16)
