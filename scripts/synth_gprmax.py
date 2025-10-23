"""Generate synthetic GPR scenes for model experimentation."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    """Build CLI arguments for synthetic scene generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Create synthetic radar scenes using random material mixes."
        ),
    )
    parser.add_argument(
        "output_root",
        type=pathlib.Path,
        help=(
            "Directory receiving synthetic scenes; reducing it shortens "
            "stored history depth."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help=(
            "Number of scenes to generate; larger values broaden class "
            "coverage yet extend runtime."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help=(
            "Random seed controlling reproducibility; changing it "
            "shifts scene layouts."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help=(
            "Grid width in samples; larger widths increase file size "
            "and resolution."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help=(
            "Grid height in samples; taller grids improve depth "
            "coverage yet require more memory."
        ),
    )
    parser.add_argument(
        "--anomalies",
        type=int,
        default=3,
        help=(
            "Count of buried targets per scene; higher values raise "
            "class frequency yet risk overlaps."
        ),
    )
    return parser.parse_args()


def random_materials(rng: np.random.Generator) -> Dict[str, float]:
    """Sample material properties from broad priors."""
    permittivity = rng.uniform(4.0, 9.0)
    conductivity = rng.uniform(0.01, 0.05)
    # Larger moisture raises permittivity and slows propagation.
    moisture = rng.uniform(0.05, 0.25)
    return {
        "permittivity": permittivity,
        "conductivity": conductivity,
        "moisture": moisture,
    }


def synthesize_scene(
    width: int,
    height: int,
    anomalies: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Create synthetic amplitude grids and label masks."""
    background = rng.normal(loc=0.0, scale=1.0, size=(height, width))
    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(anomalies):
        center_x = rng.integers(0, width)
        center_y = rng.integers(0, height)
        radius = rng.integers(5, 15)
        # Larger radius increases visibility yet reduces sample diversity.
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        region = distance <= radius
        mask[region] = 1
        amplitude = rng.uniform(2.0, 4.0)
        background[region] += amplitude
    noise = rng.normal(loc=0.0, scale=0.3, size=(height, width))
    # Higher noise simulates interference yet lowers training SNR.
    signal = background + noise
    return {
        "signal": signal.astype(np.float32),
        "mask": mask,
    }


def save_scene(
    scene: Dict[str, np.ndarray],
    meta: Dict[str, float],
    output_dir: pathlib.Path,
    index: int,
) -> Dict[str, str]:
    """Store scene arrays and metadata to disk."""
    sample_dir = output_dir / f"scene_{index:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    signal_path = sample_dir / "signal.npy"
    mask_path = sample_dir / "mask.npy"
    meta_path = sample_dir / "meta.json"
    np.save(signal_path, scene["signal"])
    np.save(mask_path, scene["mask"])
    meta_path.write_text(json.dumps(meta, indent=2))
    return {
        "signal": str(signal_path),
        "mask": str(mask_path),
        "meta": str(meta_path),
    }


def main() -> None:
    """Entry point for generating synthetic data."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    manifests: List[Dict[str, str]] = []
    for index in range(args.count):
        materials = random_materials(rng)
        scene = synthesize_scene(
            width=args.width,
            height=args.height,
            anomalies=args.anomalies,
            rng=rng,
        )
        manifest_entry = save_scene(
            scene=scene,
            meta=materials,
            output_dir=args.output_root,
            index=index,
        )
        manifest_entry.update(materials)
        manifests.append(manifest_entry)
    manifest_path = args.output_root / "synth_manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
