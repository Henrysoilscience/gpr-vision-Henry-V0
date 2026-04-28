"""Run model-based inference for radargrams/images and emit JSON artifacts."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


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
        return self.dec(torch.cat([upsampled, skip], dim=1))


class YOLOv8LikeSegModel(torch.nn.Module):
    """Tiny YOLOv8-style dense segmentation/objectness model."""

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
        logits = self.head(self.backbone(x))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for trained GPR models.")
    parser.add_argument("inputs", nargs="+", type=pathlib.Path, help="Input files/directories (.npy/.png/.jpg/.jpeg/.tif).")
    parser.add_argument("output_dir", type=pathlib.Path, help="Directory to store JSON + optional masks/overlays.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--model", choices=["unet", "yolov8"], default="unet")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--visualization-dir", type=pathlib.Path, default=None, help="Optional directory for overlay PNG outputs.")
    return parser.parse_args()


def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "unet":
        return UNetModel()
    if model_name == "yolov8":
        return YOLOv8LikeSegModel()
    raise ValueError(f"Unsupported model: {model_name}")


def load_model(checkpoint_path: pathlib.Path, model_name: str, device: torch.device) -> torch.nn.Module:
    model = build_model(model_name)
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    cleaned = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def iter_inputs(paths: List[pathlib.Path]) -> Iterable[pathlib.Path]:
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            for p in sorted(path.glob("**/*")):
                if p.suffix.lower() in {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                    yield p


def load_array(path: pathlib.Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
    else:
        image = Image.open(path).convert("L")
        array = np.asarray(image).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D input, got shape {array.shape} for {path}")
    array = array - array.mean()
    array = array / (array.std() + 1e-6)
    return array


def connected_components(mask: np.ndarray) -> List[Dict[str, object]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    detections: List[Dict[str, object]] = []

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords: List[Tuple[int, int]] = []
            min_y, min_x, max_y, max_x = y, x, y, x
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                min_y = min(min_y, cy)
                min_x = min(min_x, cx)
                max_y = max(max_y, cy)
                max_x = max(max_x, cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx] == 1:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            detections.append(
                {
                    "bbox_xyxy": [int(min_x), int(min_y), int(max_x), int(max_y)],
                    "area": len(coords),
                }
            )
    return detections


def save_overlay(base: np.ndarray, mask: np.ndarray, output_path: pathlib.Path) -> None:
    base_norm = base - base.min()
    denom = base_norm.max() + 1e-6
    gray = np.uint8((base_norm / denom) * 255)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[mask == 1] = np.array([255, 64, 64], dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.model, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    for input_path in iter_inputs(args.inputs):
        arr = load_array(input_path)
        inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inp)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

        mask = (probs >= args.threshold).astype(np.uint8)
        detections = connected_components(mask)

        sample_name = input_path.parent.name if input_path.stem == "signal" and input_path.parent.name.startswith("scene_") else input_path.stem
        out_dir = args.output_dir / sample_name
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "probs.npy", probs)
        np.save(out_dir / "mask.npy", mask)

        overlay_path = None
        if args.visualization_dir is not None:
            overlay_path = args.visualization_dir / f"{sample_name}_overlay.png"
            save_overlay(arr, mask, overlay_path)

        confidence = float(probs[mask == 1].mean()) if np.any(mask == 1) else float(probs.mean())
        record = {
            "input": str(input_path),
            "output_dir": str(out_dir),
            "mask_path": str(out_dir / "mask.npy"),
            "probs_path": str(out_dir / "probs.npy"),
            "model": args.model,
            "threshold": args.threshold,
            "confidence": confidence,
            "detections": detections,
            "overlay_path": str(overlay_path) if overlay_path else None,
        }
        (out_dir / "result.json").write_text(json.dumps(record, indent=2))
        records.append(record)

    summary_path = args.output_dir / "infer_results.json"
    summary_path.write_text(json.dumps(records, indent=2))
    print(json.dumps({"num_inputs": len(records), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
