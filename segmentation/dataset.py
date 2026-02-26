"""SKW segmentation dataset — rasterizes YOLO bbox labels into binary masks."""

import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from fastai.data.core import DataLoaders
from fastai.torch_core import TensorImage
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.data import (
    adjust_bboxes_for_letterbox,
    letterbox,
    load_labels,
    parse_splits,
)


def rasterize_boxes(bboxes: torch.Tensor, img_size: int) -> torch.Tensor:
    """Convert normalized xywh boxes to a binary mask [H, W]."""
    mask = torch.zeros(img_size, img_size, dtype=torch.long)
    for box in bboxes:
        x1 = int(max(0, (box[0] - box[2] / 2) * img_size))
        y1 = int(max(0, (box[1] - box[3] / 2) * img_size))
        x2 = int(min(img_size, (box[0] + box[2] / 2) * img_size))
        y2 = int(min(img_size, (box[1] + box[3] / 2) * img_size))
        mask[y1:y2, x1:x2] = 1
    return mask


def center_weight_map(
    bboxes: torch.Tensor, img_size: int, max_weight: float = 5.0
) -> torch.Tensor:
    """Generate a spatial weight map with higher weight at bbox centers.

    For each bounding box, weight is max_weight at the center and linearly
    fades to 1.0 at the edges (Chebyshev distance). Where boxes overlap,
    the larger weight is kept.

    Args:
        bboxes: Normalized xywh boxes [N, 4].
        img_size: Image dimension (square).
        max_weight: Weight at the center of each box (edges are 1.0).

    Returns:
        Weight map [H, W] with values in [1.0, max_weight].
    """
    weights = torch.ones(img_size, img_size, dtype=torch.float32)

    for box in bboxes:
        cx = box[0].item() * img_size
        cy = box[1].item() * img_size
        w = box[2].item() * img_size
        h = box[3].item() * img_size
        hw, hh = w / 2, h / 2

        x1 = int(max(0, cx - hw))
        y1 = int(max(0, cy - hh))
        x2 = int(min(img_size, cx + hw))
        y2 = int(min(img_size, cy + hh))

        if x2 <= x1 or y2 <= y1:
            continue

        xs = torch.arange(x1, x2, dtype=torch.float32) + 0.5
        ys = torch.arange(y1, y2, dtype=torch.float32) + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        # Normalized distance from center: 0 at center, 1 at edge
        dx = torch.abs(xx - cx) / hw if hw > 0 else torch.zeros_like(xx)
        dy = torch.abs(yy - cy) / hh if hh > 0 else torch.zeros_like(yy)
        dist = torch.clamp(torch.maximum(dx, dy), 0.0, 1.0)

        # max_weight at center (dist=0), 1.0 at edge (dist=1)
        box_weight = max_weight - (max_weight - 1.0) * dist

        weights[y1:y2, x1:x2] = torch.maximum(weights[y1:y2, x1:x2], box_weight)

    return weights


class SKWSegDataset(Dataset):
    """Dataset returning (image, binary_mask) for segmentation."""

    def __init__(
        self,
        image_paths: list[Path],
        data_dir: Path,
        img_size: int = 800,
        max_weight: float = 5.0,
    ):
        self.image_paths = image_paths
        self.labels_dir = data_dir / "labels"
        self.img_size = img_size
        self.max_weight = max_weight

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image as [3, H, W] float32 0-1
        img = Image.open(img_path).convert("RGB")
        img = TF.to_tensor(img)
        _, orig_h, orig_w = img.shape

        # Letterbox resize
        img, scale, pad_w, pad_h = letterbox(img, self.img_size)

        # Load YOLO labels and adjust bboxes for letterbox
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        cls, bboxes = load_labels(label_path)
        bboxes = adjust_bboxes_for_letterbox(
            bboxes, orig_w, orig_h, self.img_size, scale, pad_w, pad_h
        )

        # Rasterize all boxes into binary mask (all classes -> foreground)
        mask = rasterize_boxes(bboxes, self.img_size)

        # Center-weighted loss map (high at bbox centers, 1.0 at edges/background)
        weight_map = center_weight_map(
            bboxes, self.img_size, max_weight=self.max_weight
        )

        return img, mask, weight_map


def seg_collate(batch):
    """Collate into (TensorImage[B,3,H,W], masks[B,H,W], weights[B,H,W]).

    Returns 3 flat items so fastai (n_inp=1) splits as:
        xb = (images,)
        yb = (masks, weight_maps)
    """
    images, masks, weight_maps = zip(*batch)
    images = TensorImage(torch.stack(images, 0))
    masks = torch.stack(masks, 0)
    weight_maps = torch.stack(weight_maps, 0)
    return images, masks, weight_maps


def build_seg_dataloaders(
    data_dir: Path,
    img_size: int = 800,
    bs: int = 16,
    num_workers: int = 4,
    max_weight: float = 5.0,
) -> DataLoaders:
    """Build fastai DataLoaders for SKW segmentation dataset."""
    split_map = parse_splits(data_dir)
    images_dir = data_dir / "images"

    all_images = sorted(
        f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    train_images = [p for p in all_images if split_map.get(p.stem) in ("train", "test")]
    val_images = [p for p in all_images if split_map.get(p.stem) == "val"]

    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")

    train_ds = SKWSegDataset(
        train_images, data_dir, img_size, max_weight=max_weight
    )
    val_ds = SKWSegDataset(
        val_images, data_dir, img_size, max_weight=max_weight
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=seg_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=seg_collate,
        pin_memory=True,
    )

    dls = DataLoaders(train_dl, val_dl, device=torch.device("cuda"))
    dls.n_inp = 1  # only images are model input; masks + weight_maps are targets
    return dls
