"""Shared config loading and path validation for script entrypoints."""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import yaml

DEFAULT_PATHS: Dict[str, str] = {
    "raw_data_root": "data/raw",
    "processed_data_root": "data/processed",
    "label_root": "data/labels",
    "split_files_root": "data/splits",
    "model_cache_dir": "models/cache",
    "weights_output_dir": "models",
    "evaluation_output_dir": "outputs/eval",
}


@dataclass
class RuntimeConfig:
    """Container for merged config dictionaries and resolved paths."""

    raw: Dict[str, Any]
    paths: Dict[str, pathlib.Path]


def load_runtime_config(
    config_dir: pathlib.Path | None = None,
) -> RuntimeConfig:
    """Load and merge dataset/train/eval config files."""
    root = pathlib.Path.cwd() if config_dir is None else config_dir
    configs = {
        "dataset": _load_yaml(root / "configs" / "dataset.yaml"),
        "train": _load_yaml(root / "configs" / "train_cfg.yaml"),
        "eval": _load_yaml(root / "configs" / "eval_cfg.yaml"),
    }
    merged: Dict[str, Any] = {}
    for section in ("dataset", "train", "eval"):
        _deep_update(merged, configs[section])

    config_paths = _collect_path_overrides(merged)
    final_paths = {
        key: pathlib.Path(config_paths.get(key, fallback))
        for key, fallback in DEFAULT_PATHS.items()
    }
    return RuntimeConfig(raw=merged, paths=final_paths)


def add_path_override_args(
    parser: argparse.ArgumentParser,
    defaults: Mapping[str, pathlib.Path],
) -> None:
    """Add CLI overrides for all standard path settings."""
    parser.add_argument(
        "--raw-data-root",
        type=pathlib.Path,
        default=defaults["raw_data_root"],
    )
    parser.add_argument(
        "--processed-data-root",
        type=pathlib.Path,
        default=defaults["processed_data_root"],
    )
    parser.add_argument(
        "--label-root",
        type=pathlib.Path,
        default=defaults["label_root"],
    )
    parser.add_argument(
        "--split-files-root",
        type=pathlib.Path,
        default=defaults["split_files_root"],
    )
    parser.add_argument(
        "--model-cache-dir",
        type=pathlib.Path,
        default=defaults["model_cache_dir"],
    )
    parser.add_argument(
        "--weights-output-dir",
        type=pathlib.Path,
        default=defaults["weights_output_dir"],
    )
    parser.add_argument(
        "--evaluation-output-dir",
        type=pathlib.Path,
        default=defaults["evaluation_output_dir"],
    )


def validate_paths(
    *,
    required_existing: Iterable[pathlib.Path],
    create_if_missing: Iterable[pathlib.Path],
) -> None:
    """Fail fast for required inputs and create safe output directories."""
    for path in required_existing:
        if not path.exists():
            msg = f"Required path does not exist: {path}"
            raise FileNotFoundError(msg)
    for path in create_if_missing:
        path.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    """Load YAML content from path."""
    if not path.exists():
        msg = f"Missing required config file: {path}"
        raise FileNotFoundError(msg)
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"Config file must be a mapping: {path}"
        raise ValueError(msg)
    return data


def _deep_update(target: Dict[str, Any], incoming: Mapping[str, Any]) -> None:
    """Recursively merge incoming mapping into target mapping."""
    for key, value in incoming.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _collect_path_overrides(config: Mapping[str, Any]) -> Dict[str, str]:
    """Collect path overrides from canonical and legacy config fields."""
    result: Dict[str, str] = {}
    paths_section = config.get("paths", {})
    if isinstance(paths_section, Mapping):
        for key in DEFAULT_PATHS:
            value = paths_section.get(key)
            if value:
                result[key] = str(value)

    legacy = {
        "processed_data_root": config.get("root"),
        "split_files_root": pathlib.Path(config["train_split"]).parent
        if config.get("train_split")
        else None,
        "evaluation_output_dir": pathlib.Path(config["summary_path"]).parent
        if config.get("summary_path")
        else None,
    }
    for key, value in legacy.items():
        if value:
            result[key] = str(value)
    return result
