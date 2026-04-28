# Experiment Log

| Date | Experiment | Key Settings | Result |
| --- | --- | --- | --- |
| 2024-05-01 | Baseline UNet | lr=1e-3, batch=4, epochs=5 | mIoU 0.62 |
| 2024-05-05 | Augmented UNet | +noise 0.3, +rotate 10 deg | mIoU 0.68 |
| 2024-05-10 | Classifier | hidden=32, stride=2 | mAP 0.74 |

## Optimization

### Reproducibility Policy
- Use a fixed train/val/test split for all optimization experiments.
- Use a fixed random seed policy across data loading, augmentation, and model initialization.

### Baseline
- Define one baseline model configuration before running optimization sweeps.
- Record baseline metrics as the reference point for all subsequent comparisons.

### Controlled Experiment Grid
Run one axis of change at a time while keeping all other settings fixed.

- Preprocessing variants
  - Image normalization strategy
  - Resize/crop policy
  - Label cleanup rules
- Augmentation variants
  - Geometric transforms
  - Color/intensity transforms
  - Probability/scheduling of augmentations
- Model architecture variants
  - Backbone choice
  - Decoder/head design
  - Capacity scaling (depth/width)
- Threshold/post-processing variants
  - Decision thresholds
  - Connected-component filtering
  - Morphological cleanup rules

### Model Selection Decision Rule
- **Primary metric:** choose the model with the best validation score on the primary objective metric.
- **Secondary metrics (tie-breakers):** use robustness and efficiency metrics (e.g., precision/recall balance, latency/FPS, and stability across folds/seeds) to break ties.

### Per-Experiment Tracking Checklist
For every experiment entry, include:

- Config snapshot
  - Commit hash or config file reference
  - Hyperparameters and data/augmentation settings
- Metrics
  - Primary and secondary validation metrics
  - Test metrics for finalized candidates
- Runtime/FPS
  - Training runtime
  - Inference latency and FPS on target hardware
- Error-case notes
  - Typical failure patterns
  - Representative qualitative examples and hypotheses

## Notes
- Record MLflow run IDs for reproducibility and attach links when shared.
- Update this log immediately after each completed experiment batch.
