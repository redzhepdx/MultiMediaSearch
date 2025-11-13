import json
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def read_annotations(annotations: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Reads annotations from json file (Current standard)
    :param annotations: full path of the annotations file
    :return: Annotations as a list or dictionary
    """
    with open(annotations, "rb") as fp:
        annotations = json.load(fp)
    fp.close()
    return annotations


def read_config(config_path: Union[Path, str]) -> Dict:
    """
    Parses a yaml file and stores in a dictionary
    :param config_path: full path of a config file
    :return: configuration file in dictionary
    """
    with open(config_path) as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    fp.close()
    return config


def write_annotations(config_path: Union[Path, str], config: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
    """
    :param config_path: target path of the configuration file [JSON]
    :param config: content of the configuration
    """
    with open(config_path, "w+") as fp:
        json.dump(config, fp)
    fp.close()


def write_config(config_path: Union[Path, str], config: Dict[str, Any]) -> None:
    """
    :param config_path: target path of the configuration file [YAML]
    :param config: content of the configuration
    """
    with open(config_path, "w+") as fp:
        yaml.safe_dump(config, fp, sort_keys=True, indent=4)
    fp.close()
