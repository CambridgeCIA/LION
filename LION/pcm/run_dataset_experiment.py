"""Run a PCM experiment for a dataset."""

import dataclasses
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import yaml
from tqdm import tqdm as std_tqdm

from LION.data_loaders.pcm.data_loader import PCMSyntheticPNGTorchDataset
from LION.data_loaders.pcm.data_primitives import PCMSingleSampleParameter
from LION.experiments.pcm_experiments import PCMExperimentParameter
from LION.pcm.run_pcm_single_sample_experiment import run_single_sample_experiment
from LION.utils.log import CompactListDumper, to_yaml_friendly


def run_dataset_experiment(config: PCMExperimentParameter) -> Path:
    """Run the full PCM experiment.

    Args:
        config: Experiment configuration.
    """
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.runtime.root_output_dir = (
        config.runtime.root_output_dir / f"{current_datetime_str}_{config.id}"
    )
    config.runtime.root_output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.runtime.root_output_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            to_yaml_friendly(dataclasses.asdict(config)),
            f,
            Dumper=CompactListDumper,
            sort_keys=False,
        )

    dataset = PCMSyntheticPNGTorchDataset(config.data)
    tqdm = partial(std_tqdm, dynamic_ncols=True)
    sample_ids_iterator = tqdm(dataset.sample_ids, desc="Running samples")
    for category in tqdm(dataset.categories, desc="Running categories"):
        for sample_id in sample_ids_iterator:
            filename = f"{category}/{dataset.filename(sample_id)}"
            sample_ids_iterator.set_postfix_str(f"{filename}")
            sample_config = deepcopy(config)
            sample_config.this_sample = PCMSingleSampleParameter(
                data_filename=filename, ground_truth_filename=filename
            )
            run_single_sample_experiment(
                sample_config,
                sample_name=f"artifacts/{category}/sample_{sample_id:04d}",
                category=category,
                sample_id=sample_id,
            )
    return config.runtime.root_output_dir
