import re
from typing import Any, Dict, List, Optional, OrderedDict, Union

import torch


def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    """
    :param state_dict: pytorch deep model state dict
    :param rename_in_layers: where to update in tensor names
    :return: updated state dict
    """
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result


def state_dict_from_disk(
        file_path: str,
        rename_in_layers: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], OrderedDict]:
    """
    Loads PyTorch checkpoint from disk, optionally renaming layer names.
    ex:
    {
        "model.0.": "",
        "model.": ""
    }
    :param file_path: path to the torch checkpoint.
    :param rename_in_layers: {from_name: to_name}
    :return state dictionary of the new model
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:
        state_dict = rename_layers(state_dict, rename_in_layers)

    return state_dict


def find_average(outputs: List, name: str) -> torch.Tensor:
    """
    :param outputs: output metrics in a list form
    :param name: target metric name
    :return: average value of the given metric
    """
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()
