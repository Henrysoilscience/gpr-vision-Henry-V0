"""Run inference on GPR samples using trained checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Iterable, List

import numpy as np
import torch

from config_layer import add_path_override_args, load_runtime_config
from config_layer import validate_paths

class InferScriptError(Exception):
    """Raised for validation and inference failures."""


class SimpleUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = 8
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(batch))


class SimpleClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden = 32
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, hidden, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(batch).mean(dim=[2, 3]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions from trained checkpoints.")
    parser.add_argument("input_path", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--model-type", choices=["segmentation", "classification"], default="segmentation")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def validate_extension(path: pathlib.Path, allowed: List[str], context: str) -> None:
    if path.suffix.lower() not in allowed:
        raise InferScriptError(f"Invalid file extension for {context}: '{path}'. Expected one of {allowed}.")


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_path.exists():
        raise InferScriptError(f"input_path does not exist: '{args.input_path}'.")
    if not args.checkpoint.exists() or not args.checkpoint.is_file():
        raise InferScriptError(f"checkpoint does not exist or is not a file: '{args.checkpoint}'.")
    validate_extension(args.checkpoint, [".ckpt", ".pt", ".pth"], "checkpoint")
    if not (0.0 <= args.threshold <= 1.0):
        raise InferScriptError("Invalid --threshold value. Use a float in [0, 1].")


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    model: torch.nn.Module = SimpleUNet() if args.model_type == "segmentation" else SimpleClassifier()
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict({k.replace("net.", "", 1): v for k, v in state["state_dict"].items() if not k.startswith("trainer")})
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def iter_inputs(path: pathlib.Path) -> Iterable[pathlib.Path]:
    if path.is_file():
        validate_extension(path, [".npy"], "input radargram")
        yield path
    else:
        for candidate in sorted(path.glob("**/*.npy")):
            yield candidate


def load_signal(path: pathlib.Path) -> torch.Tensor:
    validate_extension(path, [".npy"], "input radargram")
    array = np.load(path)
    tensor = torch.from_numpy(array).float()
    tensor = tensor - tensor.mean()
    return tensor.unsqueeze(0).unsqueeze(0)


def save_output(output_dir: pathlib.Path, input_path: pathlib.Path, logits: torch.Tensor, threshold: float, model_type: str) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / input_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / "logits.npy", logits.squeeze().numpy())
    if model_type == "segmentation":
        mask = (torch.sigmoid(logits) >= threshold).byte()
        np.save(result_dir / "mask.npy", mask.squeeze(0).squeeze(0).numpy())
    else:
        probs = torch.softmax(logits, dim=1)
        np.save(result_dir / "probs.npy", probs.squeeze(0).numpy())
    return result_dir


def config_hash(args: argparse.Namespace) -> str:
    payload = {
        "input_path": str(args.input_path),
        "checkpoint": str(args.checkpoint),
        "model_type": args.model_type,
        "threshold": args.threshold,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_run_metadata(args: argparse.Namespace, outputs: List[str]) -> None:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"input_path": str(args.input_path), "checkpoint": str(args.checkpoint)},
        "config_hash": config_hash(args),
        "outputs": outputs,
    }
    (args.output_dir / "infer_run_metadata.json").write_text(json.dumps(metadata, indent=2))


def run() -> None:
    args = parse_args()
    validate_args(args)
    model = load_model(args)
    outputs: List[str] = []
    for input_file in iter_inputs(args.input_path):
        signal = load_signal(input_file)
        with torch.no_grad():
            logits = model(signal)
        outputs.append(str(save_output(args.output_dir, input_file, logits, args.threshold, args.model_type)))
    if not outputs:
        raise InferScriptError("No .npy input files found to process.")
    (args.output_dir / "infer_summary.json").write_text(json.dumps(outputs, indent=2))
    write_run_metadata(args, outputs)


def main() -> None:
    try:
        run()
    except InferScriptError as exc:
        print(f"[infer] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover
        print(f"[infer] Unexpected failure: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
