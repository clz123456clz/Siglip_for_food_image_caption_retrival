import random
from typing import Any, Optional
import torch
import numpy as np
import json, os


__all__ = [
    "set_random_seed",
    "config",
]
SEED = 42


def set_random_seed() -> None:
    """Set the random seed for reproducibility and enforces deterministic algorithms.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config(attr: str, config_dir: Optional[str] = None) -> Any:
    """
    Retrieve a configuration value from a JSON config file.

    Supports specifying a custom config directory; defaults to the current working directory.
    Uses caching to avoid reloading the config multiple times.

    Args:
        attr (str): The attribute to retrieve (supports dot notation, e.g., "training.lr").
        config_dir (Optional[str]): Directory containing `config.json`.
            If None, uses the current working directory.

    Returns:
        Any: The value of the requested attribute.

    Raises:
        FileNotFoundError: If `config.json` does not exist.
        KeyError: If the requested key path is not found in the config.
        ValueError: If the JSON file is malformed.
    """
    config_dir = config_dir or os.getcwd()
    config_path = os.path.join(config_dir, "config.json")

    if not hasattr(config, "_cache") or getattr(config, "_cache_path", None) != config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config._cache = json.load(f)
                config._cache_path = config_path
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in {config_path}: {e}")

    node = config._cache
    for part in attr.split("."):
        if part not in node:
            raise KeyError(f"Key '{part}' not found in config.json ({config_path})")
        node = node[part]

    return node