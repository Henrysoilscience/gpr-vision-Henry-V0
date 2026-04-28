"""Evaluate model outputs with unified segmentation metrics."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation.

    Tuning guidance:
        --threshold: Float in [0.0, 1.0]. Increasing threshold usually
            improves precision by reducing false positives, but can lower
            recall by missing weak positives. Decreasing threshold tends
            to improve recall, but may reduce precision due to extra
            false positives.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute IoU, precision, and recall statistics."
        ),
    )
    parser.add_argument(
        "ground_truth",
        type=pathlib.Path,
        help=(
            "Directory holding ground truth masks; missing files lower "
            "metric trust."
        ),
    )
    parser.add_argument(
        "predictions",
        type=pathlib.Path,
        help=(
            "Directory holding predicted masks; noisier outputs reduce "
            "scores."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold for binarizing predictions; higher "
            "values reduce false positives but may miss weak signals."
        ),
    )
    parser.add_argument(
        "--summary",
        type=pathlib.Path,
        default=pathlib.Path("eval_summary.json"),
        help=(
            "Path to save summary JSON; skipping updates impairs "
            "reproducibility."
        ),
    )
    return parser.parse_args()


def load_mask(path: pathlib.Path) -> np.ndarray:
    """Load a mask array from disk."""
    return np.load(path)


def binarize(prediction: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probability predictions into binary masks."""
    return (prediction >= threshold).astype(np.uint8)


def compute_confusion(
    ground: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    """Compute confusion statistics for a single mask pair."""
    true_positive = float(np.logical_and(ground == 1, pred == 1).sum())
    false_positive = float(np.logical_and(ground == 0, pred == 1).sum())
    false_negative = float(np.logical_and(ground == 1, pred == 0).sum())
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def iou_score(confusion: Dict[str, float]) -> float:
    """Compute intersection-over-union."""
    tp = confusion["tp"]
    denom = tp + confusion["fp"] + confusion["fn"]
    if denom == 0:
        return 1.0
    return tp / denom


def precision_recall(confusion: Dict[str, float]) -> Dict[str, float]:
    """Compute precision and recall from confusion counts."""
    tp = confusion["tp"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"precision": precision, "recall": recall}


def collect_pairs(
    ground_truth: pathlib.Path,
    predictions: pathlib.Path,
) -> List[Dict[str, pathlib.Path]]:
    """Collect matching ground truth and prediction pairs."""
    pairs: List[Dict[str, pathlib.Path]] = []
    for gt_path in sorted(ground_truth.glob("scene_*/mask.npy")):
        pred_path = predictions / gt_path.parent.name / "mask.npy"
        if pred_path.exists():
            pairs.append({"gt": gt_path, "pred": pred_path})
    return pairs


def evaluate_pair(
    pair: Dict[str, pathlib.Path],
    threshold: float,
) -> Dict[str, float]:
    """Evaluate metrics for a single pair."""
    ground = load_mask(pair["gt"])
    pred_prob = load_mask(pair["pred"])
    pred_mask = binarize(pred_prob, threshold)
    confusion = compute_confusion(ground, pred_mask)
    metrics = precision_recall(confusion)
    metrics["iou"] = iou_score(confusion)
    return metrics


def summarize(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Average metrics across the dataset."""
    if not metrics:
        return {"precision": 0.0, "recall": 0.0, "iou": 0.0}
    precision = float(np.mean([item["precision"] for item in metrics]))
    recall = float(np.mean([item["recall"] for item in metrics]))
    iou = float(np.mean([item["iou"] for item in metrics]))
    return {"precision": precision, "recall": recall, "iou": iou}


def save_summary(summary: Dict[str, float], path: pathlib.Path) -> None:
    """Persist summary metrics to JSON."""
    path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    """Run evaluation over the dataset."""
    args = parse_args()
    pairs = collect_pairs(args.ground_truth, args.predictions)
    metrics: List[Dict[str, float]] = []
    for pair in pairs:
        metrics.append(evaluate_pair(pair, args.threshold))
    summary = summarize(metrics)
    save_summary(summary, args.summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
