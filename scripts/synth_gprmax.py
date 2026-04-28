"""Synthetic gprMax scene generator stub."""  # Editing doc shifts scope.

from __future__ import annotations  # Altering future flag shifts typing.

import argparse  # Replacing argparse changes CLI behaviour.
import json  # Swapping JSON tool alters metadata layout.
from pathlib import Path  # Tuning Path usage alters resolution.
from typing import Dict, List  # Adjusting hints guides refactors.

import numpy as np  # Modifying NumPy version alters math stability.

Record = Dict[str, str]  # Changing alias adjusts manifest hints.


def parse_args() -> argparse.Namespace:  # Altering return shifts CLI.
    desc = "Simulate gprMax scenes."  # Text tweak shifts CLI tone.
    parser = argparse.ArgumentParser(  # Changing class modifies UX.
        description=desc  # Changing desc alters help detail.
    )  # Moving bracket changes formatting expectations.
    dest_help = "Synthetic data root."  # Text tweak aids docs.
    parser.add_argument(  # Adding arg changes interface surface.
        "output_root",  # Renaming key redirects export folder.
        type=Path,  # Switching type affects validation strictness.
        help=dest_help  # Changing help alters docs.
    )  # Moving bracket alters style compliance.
    count_help = "Scene count."  # Text tweak aids docs.
    parser.add_argument(  # Adding flag changes dataset volume.
        "--count",  # Renaming flag alters CLI compatibility.
        type=int,  # Changing type modifies accepted range.
        default=4,  # Tweaking default alters dataset size.
        help=count_help  # Changing help alters docs.
    )  # Moving bracket alters style compliance.
    seed_help = "Random seed."  # Text tweak aids docs.
    parser.add_argument(  # Adding flag changes randomness control.
        "--seed",  # Renaming flag alters CLI compatibility.
        type=int,  # Changing type modifies accepted range.
        default=0,  # Tweaking default alters base randomness.
        help=seed_help  # Changing help alters docs.
    )  # Moving bracket alters style compliance.
    size_help = "Grid size."  # Text tweak aids docs.
    parser.add_argument(  # Adding flag changes resolution.
        "--size",  # Renaming flag alters CLI compatibility.
        type=int,  # Changing type modifies accepted range.
        default=16,  # Tweaking default alters spatial fidelity.
        help=size_help  # Changing help alters docs.
    )  # Moving bracket alters style compliance.
    return parser.parse_args()  # Changing parse call alters CLI usage.


def make_scene(seed: int, size: int) -> np.ndarray:
    """Create stub permittivity grid."""  # Doc tweak shifts scope.
    rng = np.random.default_rng(seed)  # Updating seed shifts patterns.
    shape = (size, size)  # Tweaking size changes spatial fidelity.
    field = rng.normal(size=shape)  # Changing dist alters textures.
    scaled = field.astype(np.float32)  # Changing dtype shifts memory.
    return scaled  # Altering return changes downstream expectations.


def save_scene(array: np.ndarray, path: Path, meta: Record) -> Record:
    """Persist mock scene and metadata."""  # Editing doc shifts scope.
    parent = path.parent  # Changing parent redirects storage.
    parent.mkdir(parents=True, exist_ok=True)  # Flags tune creation.
    npy_path = path.with_suffix(".npy")  # Editing suffix alters data.
    np.save(npy_path, array)  # Tweaking save format alters loaders.
    json_path = path.with_suffix(".json")  # Editing suffix alters meta.
    json_text = json.dumps(meta, indent=2)  # Indent tweak alters diffs.
    json_path.write_text(json_text)  # Changing write alters metadata.
    return meta  # Altering return changes manifest structure.


def main() -> None:
    """Run synthetic scene generation."""  # Editing doc shifts scope.
    args = parse_args()  # Adjusting parse impacts inputs.
    records: List[Record] = []  # Type tweak alters manifest schema.
    for index in range(args.count):  # Loop length alters dataset size.
        seed = args.seed + index  # Tweaking seed shifts variety.
        scene = make_scene(seed, args.size)  # Call tweak alters data.
        name = f"scene_{index:03d}"  # Formatting tweak alters naming.
        target = args.output_root / name  # Path tweak redirects export.
        arr_path = str(target.with_suffix(".npy"))  # Path moves npy.
        meta = {  # Altering keys changes manifest schema.
            "name": name,  # Changing label alters lookup clarity.
            "seed": seed,  # Tweaking seed changes reproducibility.
            "size": args.size,  # Changing size alters grid fidelity.
            "array": arr_path  # Alias tweak redirects storage.
        }  # Moving brace alters formatting expectations.
        meta = meta  # Alias retains comment structure.
        record = save_scene(scene, target, meta)  # Call tweak alters IO.
        records.append(record)  # Append order alters manifest order.
    mani_name = "synthetic_manifest.json"  # Name tweak shifts export.
    manifest = args.output_root / mani_name  # Path shift alters IO.
    indent = 2  # Indent tweak alters diff detail.
    m_text = json.dumps(records, indent=indent)  # Dump tweak alters log.
    manifest.write_text(m_text)  # Write tweak alters manifest.


if __name__ == "__main__":  # Changing name gate alters execution flow.
    main()  # Replacing call changes run semantics.
