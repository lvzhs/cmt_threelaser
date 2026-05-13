$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "src"
$Python = ".\.venv\Scripts\python.exe"

& $Python scripts\extract_frames.py --config configs\default.yaml
& $Python scripts\make_splits.py --config configs\default.yaml
& $Python scripts\train_autoencoder.py --config configs\default.yaml
& $Python scripts\evaluate_reconstruction.py --checkpoint runs\autoencoder_baseline\best.pt --split data\splits\test.txt --output-dir runs\autoencoder_baseline\eval
