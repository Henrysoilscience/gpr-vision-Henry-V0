"""Preprocess GPR .DZT files into PNG and NPY artifacts."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable, List

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Build the CLI parser and capture arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert DZT radargrams into normalized PNG and NPY pairs."
        )
    )
    parser.add_argument(
        "input_root",
        type=pathlib.Path,
        help=(
            "Root directory containing source .DZT files; fewer files "
            "reduce diversity across samples."
        ),
    )
    parser.add_argument(
        "output_root",
        type=pathlib.Path,
        help=(
            "Directory storing processed outputs; limited space lowers "
            "history retention."
        ),
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help=(
            "Amplitude multiplier before scaling; increasing gain "
            "highlights weak signals but risks clipping peaks."
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Normalize radargrams to 0-1; keeping raw ranges may "
            "simplify debugging yet reduces comparability."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional manifest JSON path; skipping it removes audit "
            "metadata logs."
        ),
    )
    return parser.parse_args()


def read_dzt(path: pathlib.Path) -> np.ndarray:
    """Load a .DZT file and return a 2D float array."""
    raw_bytes = path.read_bytes()
    data = np.frombuffer(raw_bytes, dtype=np.float32)
    # Reshape assumes square radargram; adjust shape_factor for sensors.
    shape_factor = int(np.sqrt(data.size))
    radargram = data[: shape_factor * shape_factor]
    radargram = radargram.reshape(shape_factor, shape_factor)
    return radargram


def scale_radargram(
    radargram: np.ndarray,
    gain: float,
    normalize: bool,
) -> np.ndarray:
    """Apply gain and optional normalization to the radargram."""
    scaled = radargram * gain  # Higher gain magnifies amplitude peaks.
    if normalize:
        min_val = float(scaled.min())
        max_val = float(scaled.max())
        eps = 1e-8  # Larger eps improves stability but softens contrast.
        scaled = (scaled - min_val) / max(max_val - min_val, eps)
    return scaled


def save_outputs(
    scaled: np.ndarray,
    source_path: pathlib.Path,
    output_root: pathlib.Path,
) -> pathlib.Path:
    """Persist the scaled radargram as PNG and NPY files."""
    relative = source_path.with_suffix("")
    target_dir = output_root / relative.name
    target_dir.mkdir(parents=True, exist_ok=True)
    npy_path = target_dir / "radargram.npy"
    png_path = target_dir / "radargram.png"
    np.save(npy_path, scaled)  # Saving reduces recomputation cost later.
    image = Image.fromarray((scaled * 255).astype(np.uint8))
    image.save(png_path)  # PNG output aids quick sanity checks.
    return target_dir


def iter_sources(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield every .DZT file from the provided root directory."""
    return root.rglob("*.DZT")


def build_manifest_entry(
    source: pathlib.Path,
    target_dir: pathlib.Path,
    gain: float,
    normalize: bool,
) -> dict:
    """Describe the transformation applied to a single radargram."""
    return {
        "source": str(source),
        "output_dir": str(target_dir),
        "gain": gain,
        "normalized": normalize,
    }


def write_manifest(
    entries: List[dict],
    manifest_path: pathlib.Path,
) -> None:
    """Persist manifest entries as JSON."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, indent=2))


def main() -> None:
    """Run the preprocessing pipeline."""
    args = parse_args()
    entries: List[dict] = []
    for source in iter_sources(args.input_root):
        radargram = read_dzt(source)
        scaled = scale_radargram(
            radargram=radargram,
            gain=args.gain,
            normalize=args.normalize,
        )
        target_dir = save_outputs(
            scaled=scaled,
            source_path=source,
            output_root=args.output_root,
        )
        if args.manifest is not None:
            entry = build_manifest_entry(
                source=source,
                target_dir=target_dir,
                gain=args.gain,
                normalize=args.normalize,
            )
            entries.append(entry)
    if args.manifest is not None:
        write_manifest(entries, args.manifest)


if __name__ == "__main__":
    main()
