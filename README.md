# gpr-vision-Henry-V0

Ground-penetrating radar (GPR) simulation toolkit with preprocessing,
synthetic data, training, evaluation, and inference stubs for rapid
experimentation.

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
1. Create and activate a Python 3.10+ environment:
   ```bash
   conda create -n gpr-vision python=3.10 -y
   conda activate gpr-vision
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Scripts
All scripts respect PEP8, enforce 73 character lines, and log how
variable adjustments influence outputs through inline comments.

### Preprocessing: `prep_readgssi.py`
```
python scripts/prep_readgssi.py data/raw data/processed --seed 7
```
* `input_root`: folder with `.dzt` files.
* `output_root`: folder for generated `.png`, `.npy`, and manifest.
* `--seed`: adjusts deterministic random content.

### Synthetic Scenes: `synth_gprmax.py`
```
python scripts/synth_gprmax.py data/processed --count 8 --seed 3 --size 24
```
* `output_root`: destination directory for synthetic arrays and manifest.
* `--count`: number of generated scenes.
* `--seed`: base seed for stochastic generation.
* `--size`: square dimension of each grid.

### Training Stub: `train.py`
```
python scripts/train.py models --epochs 5 --lr 0.002 --batch 6
```
* `output_root`: directory receiving `metrics.json`.
* `--epochs`: simulated number of epochs.
* `--lr`: simulated learning rate.
* `--batch`: simulated batch size.

### Evaluation Stub: `eval.py`
```
python scripts/eval.py predictions.txt labels.txt docs
```
* `predictions`: text file summarizing predictions.
* `ground_truth`: text file summarizing ground truth labels.
* `output_root`: directory receiving `evaluation.json`.

### Inference Stub: `infer.py`
```
python scripts/infer.py sample.bin docs
```
* `input_path`: binary or text file providing raw data.
* `output_root`: directory receiving `inference.json`.

## Data Directories
Populate `data/raw` with source radargrams. Synthetic results populate
`data/processed`. Labels and split definitions remain optional folders
for project-specific metadata.

## Documentation
* `docs/labels_manual.md`: annotation criteria and sample guidelines.
* `docs/experiment_log.md`: template for tracking hyperparameters.
* `docs/changelog.md`: change history for preprocessing and labeling.
* `docs/report_final.pdf`: placeholder for the final technical report.

## Development Notes
* Maintain line lengths ≤ 73 characters in all code files.
* Preserve inline comments that describe parameter sensitivities.
* Update manifests and logs when generating synthetic data or metrics.
* Commit reproducible configuration snapshots for milestone reviews.
