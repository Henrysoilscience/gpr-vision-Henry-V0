# gpr-vision-Henry-V0

Ground-penetrating radar (GPR) toolkit for preprocessing, simulation,
training, evaluation, and inference workflows.

## 1) Environment setup (Python/Conda, dependency install)

### Option A: Conda (recommended)
```bash
conda create -n gpr-vision python=3.10 -y
conda activate gpr-vision
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Existing Python environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Quick sanity checks
```bash
python --version
python scripts/train.py --help
python scripts/eval.py --help
python scripts/infer.py --help
```

## 2) Directory contract (raw data, labels, processed data, models, outputs)

Use the following project-relative layout so all scripts resolve paths
consistently:

```text
gpr-vision-Henry-V0/
├─ data/
│  ├─ raw/                 # Source .DZT files (input to preprocessing)
│  ├─ labels/              # Optional manual labels/annotation artifacts
│  ├─ processed/           # Training/eval-ready scene folders
│  │  └─ scene_0001/
│  │     ├─ signal.npy
│  │     └─ mask.npy
│  └─ splits/              # Optional train/val/test split manifests
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

## Data Preparation
1. Place `.DZT` files under `data/raw`.
2. Run preprocessing:
   ```bash
   python scripts/prep_readgssi.py data/raw data/processed \
       --normalize --manifest docs/prep_manifest.json
   ```
3. Review generated manifests and PNG previews for quality control.

## Synthetic Data Generation
```bash
python scripts/synth_gprmax.py data/processed --count 20 --seed 7
```

## 4) Training command examples with key parameter explanations

### A. Segmentation baseline
```bash
python scripts/train.py \
  data/processed \
  --model-type segmentation \
  --epochs 20 \
  --batch-size 8 \
  --learning-rate 0.0005 \
  --seed 42 \
  --mlflow-run-name seg-baseline-e20 \
  --tracking-uri file:mlruns
```

### B. Classification baseline
```bash
python scripts/train.py \
  data/processed \
  --model-type classification \
  --epochs 15 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --seed 42 \
  --mlflow-run-name cls-baseline-e15 \
  --tracking-uri file:mlruns
```

### Key parameters
- `data/processed` (positional): must contain `scene_*/signal.npy` and
  `scene_*/mask.npy`
- `--model-type`: `segmentation` or `classification`
- `--epochs`: training passes over dataset
- `--batch-size`: memory/throughput tradeoff
- `--learning-rate`: convergence stability vs speed
- `--seed`: reproducibility for initialization and sampling order
- `--tracking-uri`: local (`file:mlruns`) or remote MLflow server

### Expected training artifacts
- checkpoint: `models/<model-type>/last.ckpt`
- MLflow runs: `mlruns/` (for `file:mlruns`)

## 5) Evaluation command examples and expected outputs

### A. Evaluate predicted masks against ground truth
```bash
python scripts/eval.py \
  data/processed \
  outputs/inference \
  --threshold 0.5 \
  --summary outputs/evaluation/eval_summary_t050.json
```

### B. Threshold sweep examples
```bash
python scripts/eval.py data/processed outputs/inference --threshold 0.3 \
  --summary outputs/evaluation/eval_summary_t030.json

python scripts/eval.py data/processed outputs/inference --threshold 0.7 \
  --summary outputs/evaluation/eval_summary_t070.json
```

### Expected outputs
- A JSON summary file with dataset-level `precision`, `recall`, and `iou`
- The same metrics printed to stdout, for example:
```json
{
  "precision": 0.81,
  "recall": 0.76,
  "iou": 0.65
}
```

## 6) Inference command examples (single file and batch)

### A. Single-file inference (segmentation)
```bash
python scripts/infer.py \
  data/processed/scene_0001/signal.npy \
  outputs/inference_single \
  --checkpoint models/segmentation/last.ckpt \
  --model-type segmentation \
  --threshold 0.5
```

### B. Batch inference over a directory (segmentation)
```bash
python scripts/infer.py \
  data/processed \
  outputs/inference \
  --checkpoint models/segmentation/last.ckpt \
  --model-type segmentation \
  --threshold 0.5
```

### C. Batch inference for classification
```bash
python scripts/infer.py \
  data/processed \
  outputs/inference_cls \
  --checkpoint models/classification/last.ckpt \
  --model-type classification
```

### Expected inference outputs
- Per-input directory under output root:
  - `logits.npy`
  - segmentation mode: `mask.npy`
  - classification mode: `probs.npy`
- Aggregate list: `outputs/.../infer_summary.json`

## 7) Troubleshooting (missing paths, missing weights, bad formats)

### Missing input paths
- Symptom: runtime errors (no files found / missing directories)
- Checks:
```bash
test -d data/raw && echo "data/raw exists"
test -d data/processed && echo "data/processed exists"
find data/processed -maxdepth 2 -type f | head
```
- Fix: ensure directories follow the directory contract above.

### Missing model weights/checkpoint
- Symptom: inference fails when loading `--checkpoint`
- Checks:
```bash
test -f models/segmentation/last.ckpt && echo "seg checkpoint exists"
test -f models/classification/last.ckpt && echo "cls checkpoint exists"
```
- Fix: run training first, or pass the correct checkpoint path.

### Bad data formats
- Symptom: shape/type issues, invalid arrays, empty metrics
- Checks:
```bash
python - <<'PY'
import numpy as np
from pathlib import Path
p = Path('data/processed/scene_0001/signal.npy')
if p.exists():
    a = np.load(p)
    print('shape=', a.shape, 'dtype=', a.dtype, 'nan=', np.isnan(a).any())
else:
    print('missing', p)
PY
```
- Fixes:
  - ensure training scenes use `signal.npy` + `mask.npy`
  - ensure inference input is `.npy` (not `.png` or `.DZT`)
  - ensure prediction and GT scene names match for evaluation.

## 8) Reproducibility checklist (seed, splits, config capture)

For each experiment run, verify:

- [ ] **Seed fixed:** pass `--seed <value>` to `scripts/train.py`
- [ ] **Stable split definitions:** save split manifests under
      `data/splits/` (e.g., `train.txt`, `val.txt`, `test.txt`)
- [ ] **Config captured:** copy relevant config(s) from `configs/` plus full
      CLI commands into run notes
- [ ] **Data manifest saved:** use `--manifest outputs/prep_manifest.json`
      during preprocessing
- [ ] **Checkpoint archived:** retain `models/<model-type>/last.ckpt`
- [ ] **Metrics archived:** keep `outputs/evaluation/*.json`
- [ ] **Inference snapshot archived:** keep
      `outputs/inference*/infer_summary.json`

### Minimal end-to-end reproducible command log
```bash
python scripts/prep_readgssi.py data/raw data/processed --normalize \
  --manifest outputs/prep_manifest.json
python scripts/train.py data/processed --model-type segmentation --epochs 20 \
  --batch-size 8 --learning-rate 0.0005 --seed 42 \
  --mlflow-run-name seg-repro --tracking-uri file:mlruns
python scripts/infer.py data/processed outputs/inference \
  --checkpoint models/segmentation/last.ckpt --model-type segmentation
python scripts/eval.py data/processed outputs/inference --threshold 0.5 \
  --summary outputs/evaluation/eval_summary.json
```

## Documentation
- `docs/labels_manual.md`: Annotation criteria and negative examples.
- `docs/experiment_log.md`: Key experiment metadata and results.
- `docs/changelog.md`: Revision history for labeling rules and configs.
- `docs/report_final.pdf`: Final report placeholder ready for publication.

## Contribution Guidelines
- Follow PEP8 with a maximum line length of 73 characters.
- Track experiments through MLflow and document major changes in `docs/`.
- Commit reproducible configuration files for every delivered milestone.
