from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from weld_unsup.config import ensure_dirs, load_config
from weld_unsup.data import WeldFrameDataset
from weld_unsup.model import ConvAutoEncoder
from weld_unsup.utils import get_device, read_lines, set_seed, write_json


def make_loader(split_path: Path, image_size: int, batch_size: int, train: bool, num_workers: int) -> DataLoader:
    dataset = WeldFrameDataset(read_lines(split_path), image_size=image_size, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train and len(dataset) >= batch_size,
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        recon = model(images)
        loss = F.l1_loss(recon, images, reduction="none").mean(dim=(1, 2, 3))
        losses.extend(loss.detach().cpu().tolist())
    return float(sum(losses) / max(len(losses), 1))


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, epoch: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "config": cfg,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an unsupervised convolutional autoencoder.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    run_dir = Path(cfg["project"]["run_dir"])
    ensure_dirs(run_dir)

    device = get_device()
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    train_loader = make_loader(
        Path(data_cfg["split_dir"]) / "train.txt",
        image_size=int(data_cfg["image_size"]),
        batch_size=int(train_cfg["batch_size"]),
        train=True,
        num_workers=int(train_cfg["num_workers"]),
    )
    val_loader = make_loader(
        Path(data_cfg["split_dir"]) / "val.txt",
        image_size=int(data_cfg["image_size"]),
        batch_size=int(train_cfg["batch_size"]),
        train=False,
        num_workers=int(train_cfg["num_workers"]),
    )

    model = ConvAutoEncoder(latent_channels=int(train_cfg["latent_channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg["amp"]) and device.type == "cuda")
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"

    history = []
    best_val = float("inf")
    start = time.time()
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                recon = model(images)
                loss = F.l1_loss(recon, images)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{train_losses[-1]:.4f}", grad=f"{float(grad_norm):.2f}")

        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        val_loss = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_l1": train_loss, "val_l1": val_loss}
        history.append(row)
        print(row)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(run_dir / "best.pt", model, optimizer, scaler, epoch, cfg)
        if epoch % int(train_cfg["save_every"]) == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scaler, epoch, cfg)

    write_json(run_dir / "history.json", {"seconds": time.time() - start, "device": str(device), "history": history})
    print(f"Best val L1: {best_val:.6f}. Checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
