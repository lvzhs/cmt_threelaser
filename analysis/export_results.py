from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def video_name(path: str) -> str:
    match = re.search(r"data[\\/]+frames[\\/]+([^\\/]+)[\\/]+", path)
    if not match:
        raise ValueError(f"Cannot infer video group from {path}")
    return match.group(1)


history_payload = json.loads((ROOT / "runs" / "autoencoder_baseline" / "history.json").read_text(encoding="utf-8"))
with (OUT / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_l1", "val_l1"])
    writer.writeheader()
    writer.writerows(history_payload["history"])


score_payload = json.loads((ROOT / "runs" / "autoencoder_baseline" / "eval" / "scores.json").read_text(encoding="utf-8"))
with (OUT / "reconstruction_scores.csv").open("w", newline="", encoding="utf-8") as f:
    fieldnames = ["path", "video", "frame_id", "reconstruction_l1", "threshold", "is_anomaly"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for item in score_payload["scores"]:
        path = item["path"]
        frame_match = re.search(r"_(\d+)\.jpg$", path)
        score = float(item["reconstruction_l1"])
        threshold = float(score_payload["threshold"])
        writer.writerow(
            {
                "path": path,
                "video": video_name(path),
                "frame_id": int(frame_match.group(1)) if frame_match else "",
                "reconstruction_l1": score,
                "threshold": threshold,
                "is_anomaly": int(score >= threshold),
            }
        )


split_rows: list[dict[str, str | int]] = []
for split_path in sorted((ROOT / "data" / "splits").glob("*.txt")):
    split = split_path.stem
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            split_rows.append({"split": split, "video": video_name(line), "path": line})

with (OUT / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["split", "video", "path"])
    writer.writeheader()
    writer.writerows(split_rows)

print(f"Wrote analysis data to {OUT}")
