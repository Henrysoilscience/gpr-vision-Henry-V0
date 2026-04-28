"""Train baseline GPR models using Lightning and MLflow."""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import lightning as L


@dataclass
class TrainConfig:
    """Container for training hyperparameters.

    Tunables:
        epochs: Integer in [1, 500]. Increasing epochs usually improves
            fit and recall until overfitting starts. Decreasing epochs
            shortens runtime and lowers overfitting risk, but can leave
            the model underfit.
        batch_size: Integer in [1, 256], bounded by GPU/CPU memory.
            Increasing batch size improves throughput and gradient
            smoothness, but raises memory use and can reduce
            generalization. Decreasing it lowers memory pressure and can
            improve generalization, but increases gradient noise and
            training time.
        learning_rate: Float in [1e-6, 1e-1] for Adam in this project.
            Increasing it speeds early progress, but raises divergence
            and unstable loss risk. Decreasing it improves stability and
            final convergence quality, but requires more iterations.
        seed: Non-negative integer, commonly [0, 2**31-1]. Changing the
            seed explores stochastic variation. Holding it fixed improves
            reproducibility for comparisons.
    """

    data_root: pathlib.Path
    model_type: str
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int


class RadarDataset(Dataset):
    """Simple dataset reading signal and mask arrays."""

    def __init__(self, data_root: pathlib.Path) -> None:
        self.samples: List[Tuple[pathlib.Path, pathlib.Path]] = []
        for sample_dir in sorted(data_root.glob("scene_*/")):
            signal = sample_dir / "signal.npy"
            mask = sample_dir / "mask.npy"
            if signal.exists() and mask.exists():
                self.samples.append((signal, mask))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        signal_path, mask_path = self.samples[index]
        signal = np.load(signal_path)
        mask = np.load(mask_path)
        signal_tensor = torch.from_numpy(signal).float()
        mask_tensor = torch.from_numpy(mask).float()
        # Scaling improves stability; altering factor affects gradients.
        signal_tensor = (signal_tensor - signal_tensor.mean())
        signal_tensor = signal_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        return signal_tensor, mask_tensor


class SimpleUNet(L.LightningModule):
    """Minimal UNet-style network for segmentation.

    Tunable architecture value:
        channels: Integer in [4, 128] depending on available memory.
            Increasing channels raises representational capacity and can
            improve fine-detail recall, but increases runtime, memory,
            and overfitting risk. Decreasing channels makes training and
            inference faster and lighter, but may reduce segmentation
            precision on complex patterns.
    """

    def __init__(self, learning_rate: float) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        channels = 8
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

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer


class SimpleClassifier(L.LightningModule):
    """Lightweight classifier using global pooling.

    Tunable architecture value:
        hidden: Integer in [8, 256] depending on memory budget.
            Increasing hidden width can improve class separation and
            recall, but increases memory use, latency, and overfitting
            risk. Decreasing hidden width improves speed and memory
            usage, but can reduce precision/recall on subtle targets.
    """

    def __init__(self, learning_rate: float) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        hidden = 32
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

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        mean_activation = targets.mean(dim=[1, 2, 3])
        labels = (mean_activation > 0.0).long()
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer


def make_trainer(config: TrainConfig) -> L.Trainer:
    """Configure the Lightning trainer.

    The trainer precision is selected from accelerator availability:
    mixed precision on GPU and full precision on CPU.

    Tuning guidance:
        precision (implicit): values are effectively {16-mixed, 32}.
            Using mixed precision increases throughput and lowers memory
            use on supported GPUs, with a small chance of numerical
            instability for some workloads. Using full precision is
            slower and heavier, but is generally more numerically stable.
    """
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = "16-mixed" if accelerator == "gpu" else 32
    trainer = L.Trainer(
        max_epochs=config.epochs,
        accelerator=accelerator,
        precision=precision,
        enable_checkpointing=True,
        log_every_n_steps=1,
    )
    return trainer


def load_model(config: TrainConfig) -> L.LightningModule:
    """Build the requested Lightning module."""
    if config.model_type == "segmentation":
        return SimpleUNet(learning_rate=config.learning_rate)
    if config.model_type == "classification":
        return SimpleClassifier(learning_rate=config.learning_rate)
    message = f"Unknown model_type: {config.model_type}"
    raise ValueError(message)


def seed_everything(seed: int) -> None:
    """Seed all random generators for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training.

    Tuning guidance for command-line hyperparameters:
        --epochs: Integer in [1, 500]. Increasing improves convergence
            potential but lengthens training and can overfit. Decreasing
            shortens runs and overfitting risk but may underfit.
        --batch-size: Integer in [1, 256], memory-limited. Increasing
            raises throughput and memory use while smoothing gradients.
            Decreasing lowers memory demand but slows throughput and adds
            gradient noise.
        --learning-rate: Float in [1e-6, 1e-1]. Increasing speeds early
            learning but can diverge. Decreasing improves stability and
            final minima quality but needs more epochs.
        --seed: Non-negative integer. Adjusting explores run variance;
            fixing ensures reproducibility.
    """
    parser = argparse.ArgumentParser(
        description="Run Lightning training with MLflow tracking.",
    )
    parser.add_argument(
        "data_root",
        type=pathlib.Path,
        help=(
            "Directory holding training samples; fewer samples reduce "
            "generalization."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=["segmentation", "classification"],
        default="segmentation",
        help=(
            "Model family to train; classification ignores pixel "
            "targets."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help=(
            "Number of epochs; more epochs improve convergence yet "
            "extend runtime."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Batch size; larger batches smooth gradients yet require "
            "more memory."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help=(
            "Learning rate; higher values speed progress yet risk "
            "divergence."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed; modifying it alters weight initialization and "
            "batch order."
        ),
    )
    parser.add_argument(
        "--mlflow-run-name",
        default="gpr-training",
        help=(
            "MLflow run name; changing it keeps experiments grouped "
            "across phases."
        ),
    )
    parser.add_argument(
        "--tracking-uri",
        default="file:mlruns",
        help=(
            "MLflow tracking URI; pointing to a server centralizes "
            "records."
        ),
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Convert parsed arguments into a TrainConfig."""
    return TrainConfig(
        data_root=args.data_root,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


def main() -> None:
    """Run the training workflow."""
    args = parse_args()
    config = build_config(args)
    seed_everything(config.seed)
    dataset = RadarDataset(config.data_root)
    if not dataset:
        message = f"No samples found in {config.data_root}"
        raise RuntimeError(message)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    model = load_model(config)
    trainer = make_trainer(config)
    mlflow.set_tracking_uri(args.tracking_uri)
    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.log_params(
            {
                "model_type": config.model_type,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "seed": config.seed,
            }
        )
        trainer.fit(model=model, train_dataloaders=dataloader)
        train_loss = trainer.callback_metrics.get("train_loss", 0.0)
        metrics = {"train_loss": float(train_loss)}
        mlflow.log_metrics(metrics)
        ckpt_dir = pathlib.Path("models") / config.model_type
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "last.ckpt"
        trainer.save_checkpoint(str(ckpt_path))
        summary = {
            "model_type": config.model_type,
            "checkpoint": str(ckpt_path),
            "metrics": metrics,
        }
        summary_path = ckpt_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
