# Experiment Log

| Date | Experiment | Key Settings | Result |
| --- | --- | --- | --- |
| 2024-05-01 | Baseline UNet | lr=1e-3, batch=4, epochs=5 | mIoU 0.62 |
| 2024-05-05 | Augmented UNet | +noise 0.3, +rotate 10 deg | mIoU 0.68 |
| 2024-05-10 | Classifier | hidden=32, stride=2 | mAP 0.74 |

## Notes
- Record MLflow run IDs for reproducibility and attach links when shared.
- Update this log immediately after each completed experiment batch.
