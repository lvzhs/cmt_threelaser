from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from tqdm import tqdm

from weld_unsup.config import ensure_dirs, load_config
from weld_unsup.utils import write_json


def find_videos(video_dir: Path, extensions: list[str]) -> list[Path]:
    extensions = [ext.lower() for ext in extensions]
    return sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def count_frames(output_dir: Path) -> int:
    return len(list(output_dir.glob("*.jpg")))


def extract_video(video: Path, output_dir: Path, fps: int, quality: int, overwrite: bool) -> dict[str, str | int]:
    ensure_dirs(output_dir)
    output_pattern = output_dir / f"{video.stem}_%06d.jpg"
    if overwrite:
        for image in output_dir.glob(f"{video.stem}_*.jpg"):
            image.unlink()
    elif any(output_dir.glob(f"{video.stem}_*.jpg")):
        return {"video": str(video), "output_dir": str(output_dir), "frames": count_frames(output_dir), "status": "skipped"}

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-q:v",
        str(quality),
        str(output_pattern),
    ]
    subprocess.run(cmd, check=True)
    return {"video": str(video), "output_dir": str(output_dir), "frames": count_frames(output_dir), "status": "done"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract images from weld videos.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--quality", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video_dir = Path(cfg["data"]["video_dir"])
    frame_dir = Path(cfg["data"]["frame_dir"])
    fps = args.fps or int(cfg["data"]["extract_fps"])
    videos = find_videos(video_dir, cfg["data"]["extensions"])
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir.resolve()}")

    manifest = []
    for video in tqdm(videos, desc="extracting"):
        out_dir = frame_dir / video.stem
        manifest.append(extract_video(video, out_dir, fps=fps, quality=args.quality, overwrite=args.overwrite))

    write_json(frame_dir / "manifest.json", manifest)
    print(f"Extracted/checked {sum(int(item['frames']) for item in manifest)} frames from {len(videos)} videos.")


if __name__ == "__main__":
    main()
