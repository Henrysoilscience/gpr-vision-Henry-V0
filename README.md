# gpr-vision-Henry-V0

Ground-penetrating radar (GPR) toolkit for preprocessing, simulation,
training, evaluation, and inference workflows.

## Repository Layout
```
gpr-vision/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ labels/
│  └─ splits/
├─ scripts/
│  ├─ prep_readgssi.py
│  ├─ synth_gprmax.py
│  ├─ train.py
│  ├─ eval.py
│  └─ infer.py
├─ models/
│  ├─ yolov8_detector/
│  └─ unet_segmenter/
├─ configs/
│  ├─ dataset.yaml
│  ├─ train_cfg.yaml
│  └─ eval_cfg.yaml
├─ docs/
│  ├─ labels_manual.md
│  ├─ experiment_log.md
│  ├─ changelog.md
│  └─ report_final.pdf
├─ requirements.txt
└─ README.md
```

## Setup
1. Install Anaconda and create an environment:
   ```bash
   conda create -n gpr-vision python=3.10 -y
   conda activate gpr-vision
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Config Loading and Path Precedence
All script entrypoints load these files at startup:

- `configs/dataset.yaml`
- `configs/train_cfg.yaml`
- `configs/eval_cfg.yaml`

Standard path keys live under `paths`:

- `raw_data_root`
- `processed_data_root`
- `label_root`
- `split_files_root`
- `model_cache_dir`
- `weights_output_dir`
- `evaluation_output_dir`

Path resolution order is: **CLI overrides > config values > built-in
defaults**.

Every script validates required input paths and fails fast when missing.
Output directories are auto-created when safe.

## Data Preparation
1. Place `.DZT` files under `data/raw`.
2. Run preprocessing:
   ```bash
   python scripts/prep_readgssi.py --normalize \
       --manifest docs/prep_manifest.json
   ```
3. Review generated manifests and PNG previews for quality control.

## Synthetic Data Generation
```bash
python scripts/synth_gprmax.py --count 20 --seed 7
```
Tune the `--count`, `--width`, `--height`, and `--anomalies` arguments to
control variety, resolution, and target density.

## Training
```bash
python scripts/train.py --model-type segmentation \
    --epochs 10 --batch-size 8 --learning-rate 0.0005 \
    --mlflow-run-name phase-a-baseline
```
Results and checkpoints are stored under `models/<model_type>/` and logged to
MLflow using the configured tracking URI.

## Evaluation
```bash
python scripts/eval.py --threshold 0.5 \
    --summary docs/eval_summary.json
```
Adjust the threshold to balance precision and recall for downstream use.

## Inference
```bash
python scripts/infer.py data/processed outputs/eval \
    --checkpoint models/segmentation/last.ckpt --model-type segmentation
```
Generated logits, masks, or probabilities are stored inside the specified
output directory together with an inference summary JSON.

## Documentation
- `docs/labels_manual.md`: Annotation criteria and negative examples.
- `docs/experiment_log.md`: Key experiment metadata and results.
- `docs/changelog.md`: Revision history for labeling rules and configs.
- `docs/report_final.pdf`: Final report placeholder ready for publication.

## Contribution Guidelines
- Follow PEP8 with a maximum line length of 73 characters.
- Track experiments through MLflow and document major changes in `docs/`.
- Commit reproducible configuration files for every delivered milestone.
