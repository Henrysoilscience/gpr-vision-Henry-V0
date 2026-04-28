"""Generate synthetic GPR scenes for model experimentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np


class SynthScriptError(Exception):
    """Raised for synthetic data generation failures."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic radar scenes using random material mixes.")
    parser.add_argument("output_root", type=pathlib.Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--anomalies", type=int, default=3)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise SynthScriptError("Invalid --count value. Use an integer greater than 0.")
    if args.width <= 0 or args.height <= 0:
        raise SynthScriptError("Invalid --width/--height values. Use integers greater than 0.")
    if args.anomalies < 0:
        raise SynthScriptError("Invalid --anomalies value. Use an integer greater than or equal to 0.")


def random_materials(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "permittivity": rng.uniform(4.0, 9.0),
        "conductivity": rng.uniform(0.01, 0.05),
        "moisture": rng.uniform(0.05, 0.25),
    }


def synthesize_scene(width: int, height: int, anomalies: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    background = rng.normal(loc=0.0, scale=1.0, size=(height, width))
    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(anomalies):
        center_x = rng.integers(0, width)
        center_y = rng.integers(0, height)
        radius = rng.integers(5, 15)
        y, x = np.ogrid[:height, :width]
        region = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) <= radius
        mask[region] = 1
        background[region] += rng.uniform(2.0, 4.0)
    signal = background + rng.normal(loc=0.0, scale=0.3, size=(height, width))
    return {"signal": signal.astype(np.float32), "mask": mask}


def save_scene(scene: Dict[str, np.ndarray], meta: Dict[str, float], output_dir: pathlib.Path, index: int) -> Dict[str, str]:
    sample_dir = output_dir / f"scene_{index:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    signal_path = sample_dir / "signal.npy"
    mask_path = sample_dir / "mask.npy"
    meta_path = sample_dir / "meta.json"
    np.save(signal_path, scene["signal"])
    np.save(mask_path, scene["mask"])
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"signal": str(signal_path), "mask": str(mask_path), "meta": str(meta_path)}


def config_hash(args: argparse.Namespace) -> str:
    payload = {
        "count": args.count,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "anomalies": args.anomalies,
        "output_root": str(args.output_root),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_run_metadata(args: argparse.Namespace, manifest_path: pathlib.Path) -> None:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"output_root": str(args.output_root)},
        "config_hash": config_hash(args),
        "manifest": str(manifest_path),
    }
    (args.output_root / "synth_run_metadata.json").write_text(json.dumps(metadata, indent=2))


def run() -> None:
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    manifests: List[Dict[str, str]] = []
    for index in range(args.count):
        materials = random_materials(rng)
        scene = synthesize_scene(args.width, args.height, args.anomalies, rng)
        manifest_entry = save_scene(scene, materials, args.output_root, index)
        manifest_entry.update(materials)
        manifests.append(manifest_entry)
    manifest_path = args.output_root / "synth_manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2))
    write_run_metadata(args, manifest_path)


def main() -> None:
    try:
        run()
    except SynthScriptError as exc:
        print(f"[synth_gprmax] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover
        print(f"[synth_gprmax] Unexpected failure: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
