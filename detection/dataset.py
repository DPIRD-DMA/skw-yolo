"""SKW YOLO26 detection dataset loading for fastai."""

import sys
from pathlib import Path

import torch
from fastai.data.core import DataLoaders
from fastai.torch_core import TensorImage
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.data import (
    adjust_bboxes_for_letterbox,
    letterbox,
    load_labels,
    parse_splits,
)

_ALL_CLASS_NAMES = {0: "skw_0S", 1: "skw_1R"}
_SINGLE_CLASS_NAMES = {0: "skw"}


def get_class_info(single_class: bool = False):
    """Return (class_names, num_classes) based on single_class flag."""
    if single_class:
        return _SINGLE_CLASS_NAMES, 1
    return _ALL_CLASS_NAMES, 2


class SKWDetectionDataset(Dataset):
    """Dataset that returns (image_tensor, label_dict) per item."""

    def __init__(
        self,
        image_paths: list[Path],
        data_dir: Path,
        img_size: int = 640,
        single_class: bool = False,
    ):
        self.image_paths = image_paths
        self.labels_dir = data_dir / "labels"
        self.img_size = img_size
        self.single_class = single_class

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image as [3, H, W] float32 0-1
        img = Image.open(img_path).convert("RGB")
        img = TF.to_tensor(img)  # [3, H, W] float32 0-1
        _, orig_h, orig_w = img.shape

        # Letterbox resize: preserve aspect ratio, pad to target size
        img, scale, pad_w, pad_h = letterbox(img, self.img_size)

        # Load labels
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        cls, bboxes = load_labels(label_path)

        # Remap bboxes from original image space to letterboxed space
        bboxes = adjust_bboxes_for_letterbox(
            bboxes, orig_w, orig_h, self.img_size, scale, pad_w, pad_h
        )

        if self.single_class and cls.shape[0] > 0:
            cls = torch.zeros_like(cls)

        return img, {"cls": cls, "bboxes": bboxes}


def yolo_collate(batch):
    """Collate into ultralytics-compatible format.

    Returns (TensorImage[B,3,H,W], {"batch_idx": [N], "cls": [N,1], "bboxes": [N,4]})
    """
    images = []
    all_cls = []
    all_bboxes = []
    all_batch_idx = []

    for i, (img, labels) in enumerate(batch):
        images.append(img)
        n = labels["cls"].shape[0]
        if n > 0:
            all_cls.append(labels["cls"])
            all_bboxes.append(labels["bboxes"])
            all_batch_idx.append(torch.full((n,), i, dtype=torch.float32))

    images = TensorImage(torch.stack(images, 0))

    if all_cls:
        targets = {
            "batch_idx": torch.cat(all_batch_idx, 0),
            "cls": torch.cat(all_cls, 0),
            "bboxes": torch.cat(all_bboxes, 0),
        }
    else:
        targets = {
            "batch_idx": torch.zeros(0),
            "cls": torch.zeros(0, 1),
            "bboxes": torch.zeros(0, 4),
        }

    return images, targets


def build_dataloaders(
    data_dir: Path,
    img_size: int = 640,
    bs: int = 16,
    num_workers: int = 4,
    single_class: bool = False,
) -> DataLoaders:
    """Build fastai DataLoaders for SKW detection dataset."""
    split_map = parse_splits(data_dir)
    images_dir = data_dir / "images"

    all_images = sorted(
        f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    train_images = [p for p in all_images if split_map.get(p.stem) in ("train", "test")]
    val_images = [p for p in all_images if split_map.get(p.stem) == "val"]

    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")

    train_ds = SKWDetectionDataset(
        train_images, data_dir, img_size, single_class=single_class
    )
    val_ds = SKWDetectionDataset(
        val_images, data_dir, img_size, single_class=single_class
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=yolo_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=yolo_collate,
        pin_memory=True,
    )

    return DataLoaders(train_dl, val_dl, device=torch.device("cuda"))
