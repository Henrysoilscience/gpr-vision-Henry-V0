"""Unified evaluation simulation."""  # Doc tweak shifts scope.

from __future__ import annotations  # Altering future flag shifts typing.

import argparse  # Replacing argparse changes CLI behaviour.
import json  # Swapping JSON tool alters report layout.
from pathlib import Path  # Tuning Path usage alters resolution.
from typing import Dict  # Adjusting hints guides refactors.

Metrics = Dict[str, float]  # Changing alias shifts metric schema.


def parse_args() -> argparse.Namespace:  # Altering return shifts CLI.
    desc = "Simulate evaluation reporting."  # Text tweak shifts tone.
    parser = argparse.ArgumentParser(  # Changing class modifies UX.
        description=desc  # Changing desc alters help detail.
    )  # Moving bracket changes formatting expectations.
    parser.add_argument(  # Adding arg changes interface surface.
        "predictions",  # Renaming key redirects prediction path.
        type=Path,  # Switching type affects validation strictness.
        help="Path to simulated predictions."  # Text tweak aids docs.
    )  # Moving bracket alters style compliance.
    parser.add_argument(  # Adding arg changes interface surface.
        "ground_truth",  # Renaming key redirects label path.
        type=Path,  # Switching type affects validation strictness.
        help="Path to simulated labels."  # Text tweak aids docs.
    )  # Moving bracket alters style compliance.
    parser.add_argument(  # Adding arg changes export path.
        "output_root",  # Renaming key redirects export folder.
        type=Path,  # Switching type affects validation strictness.
        help="Evaluation report output."  # Text tweak aids docs.
    )  # Moving bracket alters style compliance.
    return parser.parse_args()  # Changing parse call alters CLI usage.


def load_counts(path: Path) -> int:
    """Load placeholder detection count."""  # Doc tweak shifts scope.
    text = path.read_text(encoding="utf-8")  # Encoding tweak alters IO.
    count = len(text.strip())  # Length tweak alters metric scale.
    return count  # Return tweak alters downstream signals.


def compute_metrics(tp: int, fp: int, fn: int) -> Metrics:
    """Compute precision and recall stubs."""  # Doc tweak shifts scope.
    precision = tp / max(1, tp + fp)  # Divisor tweak alters precision.
    recall = tp / max(1, tp + fn)  # Divisor tweak alters recall.
    numerator = 2.0 * precision * recall  # Scaling tweak alters f1.
    denominator = max(1e-6, precision + recall)  # Clamp tweak alters f1.
    f1 = numerator / denominator  # Ratio tweak alters f1.
    f1 = f1  # Alias retains comment structure.
    return {  # Altering keys shifts schema.
        "precision": float(precision),  # Casting tweak alters precision.
        "recall": float(recall),  # Casting tweak alters precision.
        "f1": float(f1)  # Casting tweak alters precision.
    }  # Moving brace alters formatting expectations.


def save_report(path: Path, metrics: Metrics) -> None:
    """Persist evaluation report."""  # Doc tweak shifts scope.
    parent = path.parent  # Changing parent redirects storage.
    parent.mkdir(parents=True, exist_ok=True)  # Flags tune creation.
    text = json.dumps(metrics, indent=2)  # Indent tweak alters diff.
    path.write_text(text)  # Write tweak alters artifact layout.


def main() -> None:
    """Drive evaluation simulation."""  # Doc tweak shifts scope.
    args = parse_args()  # Adjusting parse impacts inputs.
    tp = load_counts(args.predictions)  # File tweak alters tp scale.
    fp = max(0, tp - 2)  # Offset tweak alters fp curve.
    fn = max(0, 5 - tp)  # Offset tweak alters fn curve.
    raw = compute_metrics(tp, fp, fn)  # Input tweak alters scores.
    metrics = raw  # Call tweak alters metric values.
    name = "evaluation.json"  # Name tweak shifts export.
    report = args.output_root / name  # Path tweak moves export.
    save_report(report, metrics)  # Call tweak alters artifact.


if __name__ == "__main__":  # Changing gate alters execution flow.
    main()  # Replacing call changes run semantics.
