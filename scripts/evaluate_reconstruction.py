from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from weld_unsup.data import WeldFrameDataset
from weld_unsup.model import ConvAutoEncoder
from weld_unsup.utils import get_device, read_lines, write_json


def denorm(x: torch.Tensor) -> np.ndarray:
    x = (x.detach().cpu().squeeze(0) * 0.5 + 0.5).clamp(0, 1).numpy()
    return x


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Score frames with reconstruction error.")
    parser.add_argument("--checkpoint", default="runs/autoencoder_baseline/best.pt")
    parser.add_argument("--split", default="data/splits/test.txt")
    parser.add_argument("--output-dir", default="runs/autoencoder_baseline/eval")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    device = get_device()
    output_dir = Path(args.output_dir)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    dataset = WeldFrameDataset(read_lines(args.split), image_size=int(cfg["data"]["image_size"]), train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = ConvAutoEncoder(latent_channels=int(cfg["train"]["latent_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    scores = []
    max_visualizations = int(cfg["eval"]["max_visualizations"])
    for idx, (image, path) in enumerate(tqdm(loader, desc="scoring")):
        image = image.to(device)
        recon = model(image)
        diff = F.l1_loss(recon, image, reduction="none").mean(dim=1).squeeze(0)
        score = float(diff.mean().detach().cpu())
        scores.append({"path": path[0], "reconstruction_l1": score})

        if idx < max_visualizations:
            fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
            axes[0].imshow(denorm(image[0]), cmap="gray")
            axes[0].set_title("input")
            axes[1].imshow(denorm(recon[0]), cmap="gray")
            axes[1].set_title("reconstruction")
            axes[2].imshow(diff.detach().cpu().numpy(), cmap="inferno")
            axes[2].set_title(f"error {score:.4f}")
            for ax in axes:
                ax.axis("off")
            fig.savefig(output_dir / "visualizations" / f"{idx:04d}.png", dpi=140)
            plt.close(fig)

    values = np.array([item["reconstruction_l1"] for item in scores], dtype=np.float32)
    threshold = float(np.percentile(values, float(cfg["eval"]["threshold_percentile"]))) if len(values) else 0.0
    write_json(output_dir / "scores.json", {"threshold": threshold, "scores": scores})
    print(f"Scored {len(scores)} frames. Threshold: {threshold:.6f}. Output: {output_dir}")


if __name__ == "__main__":
    main()
