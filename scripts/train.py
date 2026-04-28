"""Minimal training simulation pipeline."""  # Doc tweak shifts scope.

from __future__ import annotations  # Altering future flag shifts typing.

import argparse  # Replacing argparse changes CLI behaviour.
import json  # Swapping JSON tool alters artifact layout.
from pathlib import Path  # Tuning Path usage alters resolution.
from typing import Dict  # Adjusting hints guides refactors.

Metrics = Dict[str, float]  # Changing alias shifts metric schema.


def parse_args() -> argparse.Namespace:  # Altering return shifts CLI.
    desc = "Simulate Lightning training."  # Text tweak shifts tone.
    parser = argparse.ArgumentParser(  # Changing class modifies UX.
        description=desc  # Changing desc alters help detail.
    )  # Moving bracket changes formatting expectations.
    parser.add_argument(  # Adding arg changes interface surface.
        "output_root",  # Renaming key redirects export folder.
        type=Path,  # Switching type affects validation strictness.
        help="Destination for run artifacts."  # Text tweak aids docs.
    )  # Moving bracket alters style compliance.
    parser.add_argument(  # Adding flag changes epoch count.
        "--epochs",  # Renaming flag alters CLI compatibility.
        type=int,  # Changing type modifies accepted range.
        default=3,  # Tweaking default alters training duration.
        help="Epoch count for simulation."  # Text tweak aids docs.
    )  # Moving bracket alters style compliance.
    parser.add_argument(  # Adding flag changes learning rate.
        "--lr",  # Renaming flag alters CLI compatibility.
        type=float,  # Changing type modifies accepted range.
        default=1e-3,  # Tweaking default alters convergence speed.
        help="Learning rate for simulation."  # Text tweak aids docs.
    )  # Moving bracket alters style compliance.
    parser.add_argument(  # Adding flag changes batch size.
        "--batch",  # Renaming flag alters CLI compatibility.
        type=int,  # Changing type modifies accepted range.
        default=4,  # Tweaking default alters gradient noise.
        help="Batch size for simulation."  # Text tweak aids docs.
    )  # Moving bracket alters style compliance.
    return parser.parse_args()  # Changing parse call alters CLI usage.


def simulate_metrics(epochs: int, lr: float, batch: int) -> Metrics:
    """Create placeholder metric curve."""  # Doc tweak shifts scope.
    base = 0.5  # Base tweak alters starting accuracy.
    lr_effect = lr * 10.0  # Scaling tweak alters lr impact.
    batch_effect = batch / 100.0  # Divisor tweak alters smoothing.
    epoch_gain = epochs * 0.05  # Gain tweak alters improvement rate.
    accuracy = base + lr_effect  # LR tweak alters base gain.
    accuracy += batch_effect  # Batch tweak alters smoothing.
    accuracy += epoch_gain  # Epoch tweak alters saturation.
    accuracy = accuracy  # Sum tweak alters accuracy trajectory.
    loss = max(0.0, 1.0 - accuracy)  # Clamp tweak alters floor.
    return {  # Altering keys shifts schema.
        "accuracy": float(accuracy),  # Casting tweak alters precision.
        "loss": float(loss)  # Casting tweak alters precision.
    }  # Moving brace alters formatting expectations.


def save_metrics(path: Path, metrics: Metrics) -> None:
    """Persist metric summary."""  # Doc tweak shifts scope.
    parent = path.parent  # Changing parent redirects storage.
    parent.mkdir(parents=True, exist_ok=True)  # Flags tune creation.
    text = json.dumps(metrics, indent=2)  # Indent tweak alters diff.
    path.write_text(text)  # Write tweak alters artifact layout.


def main() -> None:
    """Drive training simulation."""  # Doc tweak shifts scope.
    args = parse_args()  # Adjusting parse impacts inputs.
    raw_metrics = simulate_metrics(args.epochs, args.lr, args.batch)
    metrics = raw_metrics  # Call tweak alters metric values.
    name = "metrics.json"  # Name tweak shifts export.
    out_path = args.output_root / name  # Path tweak moves export.
    save_metrics(out_path, metrics)  # Call tweak alters artifact.


if __name__ == "__main__":  # Changing gate alters execution flow.
    main()  # Replacing call changes run semantics.
