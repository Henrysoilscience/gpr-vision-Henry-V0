"""Train GPR models with explicit architecture selection and real metrics."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class TrainConfig:
    """Container for training configuration."""

    data_root: pathlib.Path
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    val_ratio: float
    output_dir: pathlib.Path


class RadarDataset(Dataset):
    """Dataset reading radargram signal/mask arrays from scene folders."""

    def __init__(self, data_root: pathlib.Path) -> None:
        self.samples: List[Tuple[pathlib.Path, pathlib.Path]] = []
        for scene_dir in sorted(data_root.glob("scene_*/")):
            signal_path = scene_dir / "signal.npy"
            mask_path = scene_dir / "mask.npy"
            if signal_path.exists() and mask_path.exists():
                self.samples.append((signal_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal_path, mask_path = self.samples[index]
        signal = np.load(signal_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        signal_tensor = torch.from_numpy(signal)
        signal_tensor = signal_tensor - signal_tensor.mean()
        denom = signal_tensor.std() + 1e-6
        signal_tensor = signal_tensor / denom
        signal_tensor = signal_tensor.unsqueeze(0)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        mask_tensor = (mask_tensor > 0.5).float()
        return signal_tensor, mask_tensor


class UNetModel(torch.nn.Module):
    """Lightweight UNet-like model for segmentation."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, base_channels, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(base_channels, base_channels, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.pool = torch.nn.MaxPool2d(2)
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.up = torch.nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(base_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.enc1(x)
        low = self.enc2(self.pool(skip))
        upsampled = self.up(low)
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([upsampled, skip], dim=1)
        return self.dec(fused)


class YOLOv8LikeSegModel(torch.nn.Module):
    """Tiny YOLOv8-style dense head for binary segmentation/objectness."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            torch.nn.SiLU(),
        )
        self.head = torch.nn.Conv2d(channels * 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits_lowres = self.head(features)
        return F.interpolate(logits_lowres, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(model_name: str) -> torch.nn.Module:
    """Create model from CLI selection."""
    if model_name == "unet":
        return UNetModel()
    if model_name == "yolov8":
        return YOLOv8LikeSegModel()
    raise ValueError(f"Unsupported model: {model_name}")


def dice_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute batch Dice score."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + 1e-6) / (union + 1e-6)).mean()


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training/validation epoch and return metrics."""
    is_train = optimizer is not None
    model.train(mode=is_train)

    loss_values: List[float] = []
    dice_values: List[float] = []

    for signals, masks in dataloader:
        signals = signals.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(signals)
            loss = F.binary_cross_entropy_with_logits(logits, masks)
            dice = dice_score(logits, masks)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        loss_values.append(float(loss.detach().cpu()))
        dice_values.append(float(dice.detach().cpu()))

    if not loss_values:
        return {"loss": 0.0, "dice": 0.0}
    return {"loss": float(np.mean(loss_values)), "dice": float(np.mean(dice_values))}


def seed_everything(seed: int) -> None:
    """Set deterministic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train GPR models with real metrics.")
    parser.add_argument("data_root", type=pathlib.Path, help="Root containing scene_*/signal.npy and mask.npy.")
    parser.add_argument("--model", choices=["unet", "yolov8"], default="unet", help="Model architecture.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio in [0, 0.9].")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("models"))
    parser.add_argument("--tracking-uri", default="file:mlruns")
    parser.add_argument("--mlflow-run-name", default="gpr-training")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Build training config object from CLI args."""
    return TrainConfig(
        data_root=args.data_root,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Train model and save best checkpoint plus metrics artifacts."""
    args = parse_args()
    config = build_config(args)
    if not (0.0 <= config.val_ratio < 0.9):
        raise ValueError("--val-ratio must be in [0.0, 0.9).")

    seed_everything(config.seed)
    dataset = RadarDataset(config.data_root)
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 samples for train/validation split.")

    val_size = max(1, int(len(dataset) * config.val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("Validation split is too large for dataset size.")

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    mlflow.set_tracking_uri(args.tracking_uri)
    run_dir = config.output_dir / config.model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_ckpt = run_dir / "best.ckpt"
    last_ckpt = run_dir / "last.ckpt"

    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.log_params(
            {
                "model": config.model_name,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "val_ratio": config.val_ratio,
                "seed": config.seed,
            }
        )

        for epoch in range(1, config.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, device)
            val_metrics = run_epoch(model, val_loader, None, device)
            combined = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
            }
            history.append(combined)
            mlflow.log_metrics(combined, step=epoch)
            print(json.dumps(combined))

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {
                        "model_name": config.model_name,
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                    },
                    best_ckpt,
                )

        torch.save(
            {
                "model_name": config.model_name,
                "state_dict": model.state_dict(),
                "epoch": config.epochs,
                "val_loss": history[-1]["val_loss"] if history else None,
            },
            last_ckpt,
        )

    metrics_path = run_dir / "metrics_history.json"
    metrics_path.write_text(json.dumps(history, indent=2))
    summary = {
        "model": config.model_name,
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "best_val_loss": best_val_loss,
        "epochs": config.epochs,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
