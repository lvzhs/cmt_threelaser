# Weld Image Unsupervised Learning

This repository contains an unsupervised learning baseline for weld images.
It extracts frames from weld videos, trains a convolutional autoencoder, and
scores frames with reconstruction error.

## Pipeline

1. Put raw weld videos in the project root.
2. Extract frames with `scripts/extract_frames.py`.
3. Create train, validation, and test splits with `scripts/make_splits.py`.
4. Train the autoencoder with `scripts/train_autoencoder.py`.
5. Evaluate reconstruction error and generate heatmaps with `scripts/evaluate_reconstruction.py`.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
$env:PYTHONPATH="src"
python scripts/extract_frames.py --config configs/default.yaml
python scripts/make_splits.py --config configs/default.yaml
python scripts/train_autoencoder.py --config configs/default.yaml
python scripts/evaluate_reconstruction.py --checkpoint runs/autoencoder_baseline/best.pt
```

Or run the PowerShell helper:

```powershell
.\scripts\run_pipeline.ps1
```

## Outputs

- `data/frames/`: extracted frame images.
- `data/splits/`: train, validation, and test split files.
- `runs/autoencoder_baseline/best.pt`: best checkpoint.
- `runs/autoencoder_baseline/history.json`: training history.
- `runs/autoencoder_baseline/eval/scores.json`: reconstruction scores.
- `runs/autoencoder_baseline/eval/visualizations/`: input, reconstruction, and error heatmap images.

## Notes

The uploaded repository excludes raw video files. Large binary artifacts such
as extracted images, checkpoints, and evaluation visualizations are tracked
with Git LFS.
