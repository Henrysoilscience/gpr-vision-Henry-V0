"""Run inference on GPR samples using trained checkpoints."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable, List

import numpy as np
import torch


class SimpleUNet(torch.nn.Module):
    """Simplified UNet decoder for binary segmentation."""

    def __init__(self) -> None:
        super().__init__()
        channels = 8  # More channels lift accuracy yet slow inference.
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                1,
                channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded


class SimpleClassifier(torch.nn.Module):
    """Simplified classifier mirroring training architecture."""

    def __init__(self) -> None:
        super().__init__()
        hidden = 32  # Larger hidden dims lift accuracy yet add cost.
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(
                1,
                hidden,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        features = self.net(batch)
        pooled = features.mean(dim=[2, 3])
        logits = self.head(pooled)
        return logits


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for inference."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate predictions from trained checkpoints."
        ),
    )
    parser.add_argument(
        "input_path",
        type=pathlib.Path,
        help=(
            "File or directory holding .npy radargrams; fewer samples "
            "shorten processing."
        ),
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help=(
            "Directory to store predictions; limited space prunes "
            "history."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        default=pathlib.Path("models"),
        help=(
            "Directory containing pretrained and fine-tuned weights; "
            "relative checkpoints resolve from here."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("last.ckpt"),
        help=(
            "Checkpoint filename/path; relative values resolve to "
            "--model-dir/<model-type>/."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=["segmentation", "classification"],
        default="segmentation",
        help=(
            "Model family for inference; classification outputs class "
            "probabilities."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold for segmentation masks; higher "
            "values reduce false alarms."
        ),
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    """Instantiate and load model weights."""
    if args.model_type == "segmentation":
        model = SimpleUNet()
    else:
        model = SimpleClassifier()
    model_dir = args.model_dir.resolve()
    checkpoint_path = args.checkpoint
    if checkpoint_path.is_absolute():
        resolved_checkpoint = checkpoint_path.resolve()
    else:
        resolved_checkpoint = (model_dir / args.model_type / checkpoint_path).resolve()
    if not resolved_checkpoint.exists():
        expected_patterns = [
            model_dir / args.model_type / "last.ckpt",
            model_dir / args.model_type / "best.ckpt",
            model_dir / args.model_type / "*.ckpt",
        ]
        expected_text = "\n".join(
            f"  - {pattern}" for pattern in expected_patterns
        )
        message = (
            f"Checkpoint file not found: {resolved_checkpoint}\n"
            "Expected filenames/patterns include:\n"
            f"{expected_text}\n"
            "Set --model-dir and --checkpoint to valid locations."
        )
        raise FileNotFoundError(message)
    state = torch.load(resolved_checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        # Lightning checkpoints keep weights under state_dict.
        model.load_state_dict({k.replace("net.", "", 1): v
                               for k, v in state["state_dict"].items()
                               if not k.startswith("trainer")})
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def iter_inputs(path: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield input files from path."""
    if path.is_file():
        yield path
    else:
        for candidate in sorted(path.glob("**/*.npy")):
            yield candidate


def load_signal(path: pathlib.Path) -> torch.Tensor:
    """Load and normalize a radargram."""
    array = np.load(path)
    tensor = torch.from_numpy(array).float()
    tensor = tensor - tensor.mean()  # Centering stabilizes logits.
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor


def save_output(
    output_dir: pathlib.Path,
    input_path: pathlib.Path,
    logits: torch.Tensor,
    threshold: float,
    model_type: str,
) -> pathlib.Path:
    """Persist inference results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    relative = input_path.stem
    result_dir = output_dir / relative
    result_dir.mkdir(parents=True, exist_ok=True)
    logits_path = result_dir / "logits.npy"
    np.save(logits_path, logits.squeeze().numpy())
    if model_type == "segmentation":
        mask = (torch.sigmoid(logits) >= threshold).byte()
        mask_path = result_dir / "mask.npy"
        np.save(mask_path, mask.squeeze(0).squeeze(0).numpy())
    else:
        probs = torch.softmax(logits, dim=1)
        probs_path = result_dir / "probs.npy"
        np.save(probs_path, probs.squeeze(0).numpy())
    return result_dir


def main() -> None:
    """Execute inference pipeline."""
    args = parse_args()
    model = load_model(args)
    outputs: List[str] = []
    for input_file in iter_inputs(args.input_path):
        signal = load_signal(input_file)
        with torch.no_grad():
            logits = model(signal)
        result_dir = save_output(
            output_dir=args.output_dir,
            input_path=input_file,
            logits=logits,
            threshold=args.threshold,
            model_type=args.model_type,
        )
        outputs.append(str(result_dir))
    summary_path = args.output_dir / "infer_summary.json"
    summary = {
        "outputs": outputs,
        "resolved_model_dir": str(args.model_dir.resolve()),
        "resolved_checkpoint": str(
            args.checkpoint.resolve()
            if args.checkpoint.is_absolute()
            else (
                args.model_dir.resolve()
                / args.model_type
                / args.checkpoint
            ).resolve()
        ),
        "input_path": str(args.input_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "model_type": args.model_type,
        "threshold": args.threshold,
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
