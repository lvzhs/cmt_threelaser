from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def make_transforms(image_size: int, train: bool) -> transforms.Compose:
    ops: list[torch.nn.Module] = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size), antialias=True),
    ]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.15, contrast=0.15)],
                    p=0.35,
                ),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return transforms.Compose(ops)


class WeldFrameDataset(Dataset):
    def __init__(self, image_paths: list[str | Path], image_size: int, train: bool) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = make_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)
