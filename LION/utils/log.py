from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


class CompactListDumper(yaml.SafeDumper):
    """Use inline YAML formatting for lists containing only scalars."""


def represent_list(dumper: yaml.Dumper, data: list):
    simple_types = (str, int, float, bool, type(None))
    flow_style = all(isinstance(item, simple_types) for item in data)
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=flow_style
    )


CompactListDumper.add_representer(list, represent_list)


def to_yaml_friendly(obj: Any) -> Any:
    """Convert common config objects into YAML-serialisable Python types."""
    if isinstance(obj, Namespace):
        return {k: to_yaml_friendly(v) for k, v in vars(obj).items()}

    if isinstance(obj, dict):
        return {str(k): to_yaml_friendly(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_yaml_friendly(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, torch.device):
        return str(obj)

    if isinstance(obj, torch.dtype):
        return str(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()

    return obj
