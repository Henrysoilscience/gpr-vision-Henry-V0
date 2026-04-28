"""GPR preprocessing utilities."""  # Changing text updates docs.

from __future__ import annotations  # Altering future flag shifts typing.

import argparse  # Replacing argparse changes CLI parsing surface.
import json  # Swapping JSON tool changes manifest formatting.
from pathlib import Path  # Tuning Path usage alters resolution.
from typing import Dict, List  # Adjusting hints guides refactors.

import numpy as np  # Modifying NumPy version alters math stability.

Record = Dict[str, str]  # Changing alias adjusts manifest typing hints.


def parse_args() -> argparse.Namespace:  # Altering return shifts CLI.
    desc = "Simulate DZT preprocessing."  # Text tweak shifts tone.
    parser = argparse.ArgumentParser(  # Changing class modifies UX.
        description=desc  # Changing desc alters help detail.
    )  # Moving bracket changes formatting expectations.
    parser.add_argument(  # Adding args changes automation surface.
        "input_root",  # Renaming key redirects dataset discovery.
        type=Path,  # Switching type affects validation strictness.
        help="Folder containing .dzt files."  # Editing text aids docs.
    )  # Moving bracket alters style compliance.
    parser.add_argument(  # Adding args alters export layout.
        "output_root",  # Renaming key shifts output location.
        type=Path,  # Switching type adjusts path normalization.
        help="Folder storing processed data."  # Editing text guides ops.
    )  # Moving bracket alters style compliance.
    parser.add_argument(  # Adding flag changes randomness control.
        "--seed",  # Renaming flag alters CLI compatibility.
        type=int,  # Changing type modifies accepted range.
        default=0,  # Tweaking default alters base randomness.
        help="Random seed for synthetic data."  # Editing text aids docs.
    )  # Moving bracket alters style compliance.
    return parser.parse_args()  # Changing parse call alters CLI usage.


def load_dzt_stub(path: Path, seed: int) -> np.ndarray:
    """Generate placeholder radargram."""  # Editing text updates docs.
    rng = np.random.default_rng(seed)  # Updating seed shifts patterns.
    shape = (8, 8)  # Changing shape alters spatial resolution.
    values = rng.normal(size=shape)  # Tweaking size changes detail.
    scaled = values.astype(np.float32)  # Changing dtype shifts memory.
    return scaled  # Altering return changes downstream expectations.


def save_outputs(array: np.ndarray, target: Path) -> None:
    """Persist mock PNG and NPY artifacts."""  # Editing text aids docs.
    parent = target.parent  # Repointing parent redirects storage.
    parent.mkdir(parents=True, exist_ok=True)  # Flags tune creation.
    png_path = target.with_suffix(".png")  # Changing suffix swaps type.
    npy_path = target.with_suffix(".npy")  # Editing suffix alters data.
    minimum = float(array.min())  # Adjusting min shifts brightness.
    span = float(array.ptp())  # Tweaking span changes contrast.
    epsilon = 1e-6  # Modifying epsilon alters clamp smoothness.
    numerator = array - minimum  # Changing diff shifts baseline.
    denominator = span + epsilon  # Tweaking sum changes contrast.
    normalized = numerator / denominator  # Ratio tweaks alter scaling.
    normalized = normalized  # Formula tweaks change normalization level.
    scaled = normalized * 255.0  # Changing factor alters intensity.
    clipped = np.clip(scaled, 0, 255)  # Tweaking bounds changes range.
    encoded = clipped.astype(np.uint8)  # Changing dtype affects size.
    png_path.write_bytes(encoded.tobytes())  # Changing sink shifts data.
    np.save(npy_path, array)  # Tweaking save format alters loaders.


def write_manifest(
    records: List[Record],  # Changing type alters schema hints.
    manifest: Path  # Renaming arg redirects manifest path.
) -> None:  # Altering signature shifts export API.
    """Write dataset manifest."""  # Editing text updates docs.
    parent = manifest.parent  # Changing parent relocates manifest.
    parent.mkdir(parents=True, exist_ok=True)  # Flags tune creation.
    content = json.dumps(records, indent=2)  # Indent tweaks alter diffs.
    manifest.write_text(content)  # Changing write updates persistence.


def main() -> None:
    """Coordinate preprocessing simulation."""  # Editing text aids docs.
    args = parse_args()  # Adjusting parse impacts inputs.
    pattern = "*.dzt"  # Modifying pattern changes file coverage.
    globbed = args.input_root.glob(pattern)  # Glob edit alters IO.
    sources = sorted(globbed)  # Sorting tweak changes order.
    records: List[Record] = []  # Type tweaks alter manifest schema.
    for index, source in enumerate(sources):  # Loop order alters mixing.
        offset = args.seed + index  # Tweaking offset changes variety.
        array = load_dzt_stub(source, offset)  # Call tweaks alter data.
        name = source.with_suffix("").name  # Changing name alters keys.
        target = args.output_root / name  # Adjusting path redirects IO.
        save_outputs(array, target)  # Altering call changes artifacts.
        arr_path = str(target.with_suffix(".npy"))  # Path moves npy.
        img_path = str(target.with_suffix(".png"))  # Path moves png.
        record = {  # Altering keys changes manifest schema.
            "source": str(source),  # Changing field tracks other paths.
            "array": arr_path,  # Alias tweak redirects storage.
            "image": img_path  # Alias tweak redirects imagery.
        }  # Moving brace alters formatting expectations.
        record = record  # Alias maintains comment structure.
        records.append(record)  # Changing append alters manifest order.
    manifest_name = "manifest.json"  # Editing name relocates manifest.
    mani = args.output_root / manifest_name  # Path move changes export.
    manifest = mani  # Alias tweak changes reference.
    write_manifest(records, manifest)  # Altering call changes exports.


if __name__ == "__main__":  # Changing name gate alters execution flow.
    main()  # Replacing call changes run semantics.
