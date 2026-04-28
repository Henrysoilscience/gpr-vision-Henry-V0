"""Evaluate model outputs with unified segmentation metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np


class EvalScriptError(Exception):
    """Raised when evaluation inputs are invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute IoU, precision, and recall statistics.")
    parser.add_argument("ground_truth", type=pathlib.Path)
    parser.add_argument("predictions", type=pathlib.Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--summary", type=pathlib.Path, default=pathlib.Path("eval_summary.json"))
    return parser.parse_args()


def validate_extension(path: pathlib.Path, allowed: List[str], context: str) -> None:
    if path.suffix.lower() not in allowed:
        raise EvalScriptError(
            f"Invalid file extension for {context}: '{path}'. Expected one of {allowed}."
        )


def validate_args(args: argparse.Namespace) -> None:
    if not args.ground_truth.is_dir():
        raise EvalScriptError(f"ground_truth directory does not exist: '{args.ground_truth}'.")
    if not args.predictions.is_dir():
        raise EvalScriptError(f"predictions directory does not exist: '{args.predictions}'.")
    if not (0.0 <= args.threshold <= 1.0):
        raise EvalScriptError("Invalid --threshold value. Use a float in [0, 1].")
    validate_extension(args.summary, [".json"], "summary output")


def load_mask(path: pathlib.Path) -> np.ndarray:
    validate_extension(path, [".npy"], "mask input")
    return np.load(path)


def binarize(prediction: np.ndarray, threshold: float) -> np.ndarray:
    return (prediction >= threshold).astype(np.uint8)


def compute_confusion(ground: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    true_positive = float(np.logical_and(ground == 1, pred == 1).sum())
    false_positive = float(np.logical_and(ground == 0, pred == 1).sum())
    false_negative = float(np.logical_and(ground == 1, pred == 0).sum())
    return {"tp": true_positive, "fp": false_positive, "fn": false_negative}


def iou_score(confusion: Dict[str, float]) -> float:
    tp = confusion["tp"]
    denom = tp + confusion["fp"] + confusion["fn"]
    return 1.0 if denom == 0 else tp / denom


def precision_recall(confusion: Dict[str, float]) -> Dict[str, float]:
    tp = confusion["tp"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"precision": precision, "recall": recall}


def collect_pairs(ground_truth: pathlib.Path, predictions: pathlib.Path) -> List[Dict[str, pathlib.Path]]:
    pairs: List[Dict[str, pathlib.Path]] = []
    for gt_path in sorted(ground_truth.glob("scene_*/mask.npy")):
        pred_path = predictions / gt_path.parent.name / "mask.npy"
        if pred_path.exists():
            pairs.append({"gt": gt_path, "pred": pred_path})
    return pairs


def evaluate_pair(pair: Dict[str, pathlib.Path], threshold: float) -> Dict[str, float]:
    ground = load_mask(pair["gt"])
    pred_prob = load_mask(pair["pred"])
    pred_mask = binarize(pred_prob, threshold)
    confusion = compute_confusion(ground, pred_mask)
    metrics = precision_recall(confusion)
    metrics["iou"] = iou_score(confusion)
    return metrics


def summarize(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {"precision": 0.0, "recall": 0.0, "iou": 0.0}
    return {
        "precision": float(np.mean([item["precision"] for item in metrics])),
        "recall": float(np.mean([item["recall"] for item in metrics])),
        "iou": float(np.mean([item["iou"] for item in metrics])),
    }


def config_hash(args: argparse.Namespace) -> str:
    payload = {
        "ground_truth": str(args.ground_truth),
        "predictions": str(args.predictions),
        "threshold": args.threshold,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def save_summary(summary: Dict[str, float], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def write_run_metadata(args: argparse.Namespace, evaluated_count: int, path: pathlib.Path) -> None:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"ground_truth": str(args.ground_truth), "predictions": str(args.predictions)},
        "config_hash": config_hash(args),
        "evaluated_pairs": evaluated_count,
    }
    path.write_text(json.dumps(metadata, indent=2))


def run() -> None:
    args = parse_args()
    validate_args(args)
    pairs = collect_pairs(args.ground_truth, args.predictions)
    if not pairs:
        raise EvalScriptError("No matching scene_*/mask.npy pairs found between ground_truth and predictions.")
    metrics = [evaluate_pair(pair, args.threshold) for pair in pairs]
    summary = summarize(metrics)
    save_summary(summary, args.summary)
    write_run_metadata(args, evaluated_count=len(pairs), path=args.summary.parent / "eval_run_metadata.json")
    print(json.dumps(summary, indent=2))


def main() -> None:
    try:
        run()
    except EvalScriptError as exc:
        print(f"[eval] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover
        print(f"[eval] Unexpected failure: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
