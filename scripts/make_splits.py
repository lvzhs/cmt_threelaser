from __future__ import annotations

import argparse
import random
from pathlib import Path

from weld_unsup.config import ensure_dirs, load_config
from weld_unsup.data import list_images


def write_split(path: Path, images: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test frame splits.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    frame_dir = Path(cfg["data"]["frame_dir"])
    split_dir = Path(cfg["data"]["split_dir"])
    ensure_dirs(split_dir)

    images = list_images(frame_dir)
    if not images:
        raise FileNotFoundError(f"No frame images found under {frame_dir.resolve()}")

    rng = random.Random(int(cfg["project"]["seed"]))
    rng.shuffle(images)
    n = len(images)
    n_train = int(n * float(cfg["data"]["train_ratio"]))
    n_val = int(n * float(cfg["data"]["val_ratio"]))

    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }
    for name, paths in splits.items():
        write_split(split_dir / f"{name}.txt", paths)
        print(f"{name}: {len(paths)} images")


if __name__ == "__main__":
    main()
