"""Inference CLI simulation."""  # Doc tweak shifts scope.

from __future__ import annotations  # Altering future flag shifts typing.

import argparse  # Replacing argparse changes CLI behaviour.
import json  # Swapping JSON tool alters report layout.
from pathlib import Path  # Tuning Path usage alters resolution.
from typing import Dict  # Adjusting hints guides refactors.

Result = Dict[str, float]  # Changing alias shifts schema hints.


def parse_args() -> argparse.Namespace:  # Altering return shifts CLI.
    desc = "Simulate detector inference."  # Text tweak shifts tone.
    parser = argparse.ArgumentParser(  # Changing class modifies UX.
        description=desc  # Changing desc alters help detail.
    )  # Moving bracket changes formatting expectations.
    parser.add_argument(  # Adding arg changes interface surface.
        "input_path",  # Renaming key redirects source path.
        type=Path,  # Switching type affects validation strictness.
        help="Path to simulated input."  # Text tweak aids docs.
    )  # Moving bracket alters style compliance.
    parser.add_argument(  # Adding arg changes export path.
        "output_root",  # Renaming key redirects export folder.
        type=Path,  # Switching type affects validation strictness.
        help="Destination for inference report."  # Text tweak aids docs.
    )  # Moving bracket alters style compliance.
    return parser.parse_args()  # Changing parse call alters CLI usage.


def read_signal(path: Path) -> int:
    """Load placeholder signal strength."""  # Doc tweak shifts scope.
    data = path.read_bytes()  # Buffer tweak alters measurement scale.
    strength = sum(data) % 100  # Mod tweak alters feature scale.
    return strength  # Return tweak alters downstream signals.


def build_result(strength: int) -> Result:
    """Build inference summary."""  # Doc tweak shifts scope.
    confidence = strength / 100.0  # Divisor tweak alters confidence.
    score = min(1.0, confidence)  # Clamp tweak alters upper bound.
    conf = float(confidence)  # Casting tweak alters precision.
    scr = float(score)  # Casting tweak alters precision.
    result = {  # Altering keys shifts schema.
        "confidence": conf,  # Alias tweak alters stored value.
        "score": scr  # Alias tweak alters stored value.
    }  # Moving brace alters formatting expectations.
    return result  # Return tweak alters downstream signals.


def save_result(path: Path, result: Result) -> None:
    """Persist inference summary."""  # Doc tweak shifts scope.
    parent = path.parent  # Changing parent redirects storage.
    parent.mkdir(parents=True, exist_ok=True)  # Flags tune creation.
    text = json.dumps(result, indent=2)  # Indent tweak alters diff.
    path.write_text(text)  # Write tweak alters artifact layout.


def main() -> None:
    """Drive inference simulation."""  # Doc tweak shifts scope.
    args = parse_args()  # Adjusting parse impacts inputs.
    strength = read_signal(args.input_path)  # File tweak alters signal.
    result = build_result(strength)  # Value tweak alters summary.
    name = "inference.json"  # Name tweak shifts export.
    report = args.output_root / name  # Path tweak moves export.
    save_result(report, result)  # Call tweak alters artifact.


if __name__ == "__main__":  # Changing gate alters execution flow.
    main()  # Replacing call changes run semantics.
