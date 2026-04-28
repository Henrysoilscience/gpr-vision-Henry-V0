"""Preprocess GPR .DZT files into PNG and NPY artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Iterable, List

import numpy as np
from PIL import Image

from config_layer import add_path_override_args, load_runtime_config
from config_layer import validate_paths

class PrepScriptError(Exception):
    """Raised for invalid preprocessing inputs."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DZT radargrams into normalized PNG and NPY pairs.")
    parser.add_argument("input_root", type=pathlib.Path)
    parser.add_argument("output_root", type=pathlib.Path)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--manifest", type=pathlib.Path, default=None)
    return parser.parse_args()


def validate_extension(path: pathlib.Path, allowed: List[str], context: str) -> None:
    if path.suffix.lower() not in allowed:
        raise PrepScriptError(f"Invalid file extension for {context}: '{path}'. Expected one of {allowed}.")


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_root.is_dir():
        raise PrepScriptError(f"input_root does not exist or is not a directory: '{args.input_root}'.")
    if args.gain <= 0:
        raise PrepScriptError("Invalid --gain value. Use a float greater than 0.")
    if args.manifest is not None:
        validate_extension(args.manifest, [".json"], "manifest output")


def read_dzt(path: pathlib.Path) -> np.ndarray:
    validate_extension(path, [".dzt"], "source radargram")
    raw_bytes = path.read_bytes()
    data = np.frombuffer(raw_bytes, dtype=np.float32)
    shape_factor = int(np.sqrt(data.size))
    radargram = data[: shape_factor * shape_factor]
    return radargram.reshape(shape_factor, shape_factor)


def scale_radargram(radargram: np.ndarray, gain: float, normalize: bool) -> np.ndarray:
    scaled = radargram * gain
    if normalize:
        min_val = float(scaled.min())
        max_val = float(scaled.max())
        scaled = (scaled - min_val) / max(max_val - min_val, 1e-8)
    return scaled


def save_outputs(scaled: np.ndarray, source_path: pathlib.Path, output_root: pathlib.Path) -> pathlib.Path:
    target_dir = output_root / source_path.with_suffix("").name
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "radargram.npy", scaled)
    Image.fromarray((scaled * 255).astype(np.uint8)).save(target_dir / "radargram.png")
    return target_dir


def iter_sources(root: pathlib.Path) -> Iterable[pathlib.Path]:
    return root.rglob("*.DZT")


def build_manifest_entry(source: pathlib.Path, target_dir: pathlib.Path, gain: float, normalize: bool) -> dict:
    return {"source": str(source), "output_dir": str(target_dir), "gain": gain, "normalized": normalize}


def write_manifest(entries: List[dict], manifest_path: pathlib.Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, indent=2))


def config_hash(args: argparse.Namespace) -> str:
    payload = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "gain": args.gain,
        "normalize": args.normalize,
        "manifest": str(args.manifest) if args.manifest else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_run_metadata(args: argparse.Namespace, processed: int) -> None:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"input_root": str(args.input_root)},
        "config_hash": config_hash(args),
        "processed_files": processed,
    }
    (args.output_root / "prep_run_metadata.json").write_text(json.dumps(metadata, indent=2))


def run() -> None:
    args = parse_args()
    validate_args(args)
    entries: List[dict] = []
    processed = 0
    for source in iter_sources(args.input_root):
        radargram = read_dzt(source)
        scaled = scale_radargram(radargram, args.gain, args.normalize)
        target_dir = save_outputs(scaled, source, args.output_root)
        processed += 1
        if args.manifest is not None:
            entries.append(build_manifest_entry(source, target_dir, args.gain, args.normalize))
    if processed == 0:
        raise PrepScriptError("No .DZT files found under input_root.")
    if args.manifest is not None:
        write_manifest(entries, args.manifest)
    write_run_metadata(args, processed)


def main() -> None:
    try:
        run()
    except PrepScriptError as exc:
        print(f"[prep_readgssi] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover
        print(f"[prep_readgssi] Unexpected failure: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
