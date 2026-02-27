"""Data loading for SKW segmentation using fastai DataBlock."""

import sys
from functools import partial
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.core import DataLoaders
from fastai.data.transforms import FuncSplitter
from fastai.torch_core import TensorImage, TensorMask
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.data import load_labels, parse_splits


def rasterize_boxes(bboxes: torch.Tensor, img_size: int) -> torch.Tensor:
    """Convert normalized xywh boxes to a binary mask [H, W]."""
    if len(bboxes) == 0:
        return torch.zeros(img_size, img_size, dtype=torch.long)
    # Vectorized coordinate computation: [N, 4] -> pixel xyxy clamped to image
    x1 = ((bboxes[:, 0] - bboxes[:, 2] / 2) * img_size).clamp(0, img_size).int()
    y1 = ((bboxes[:, 1] - bboxes[:, 3] / 2) * img_size).clamp(0, img_size).int()
    x2 = ((bboxes[:, 0] + bboxes[:, 2] / 2) * img_size).clamp(0, img_size).int()
    y2 = ((bboxes[:, 1] + bboxes[:, 3] / 2) * img_size).clamp(0, img_size).int()
    # Build row/col grids and test all boxes at once
    rows = torch.arange(img_size)
    cols = torch.arange(img_size)
    # [N] coords -> [N,1] for broadcasting against [img_size] grid
    in_y = (rows.unsqueeze(0) >= y1.unsqueeze(1)) & (rows.unsqueeze(0) < y2.unsqueeze(1))  # [N, H]
    in_x = (cols.unsqueeze(0) >= x1.unsqueeze(1)) & (cols.unsqueeze(0) < x2.unsqueeze(1))  # [N, W]
    # Outer product per box then union across boxes: [N, H, W] -> [H, W]
    mask = (in_y.unsqueeze(2) & in_x.unsqueeze(1)).any(dim=0).long()
    return mask


def open_img(img_path: Path, img_size: int) -> TensorImage:
    """Open image, resize to img_size, return TensorImage [3, H, W]."""
    img = Image.open(img_path).convert("RGB")
    img = TF.to_tensor(img)  # [3, H, W] float32 [0, 1]
    img = TF.resize(img, [img_size, img_size], antialias=True)
    return TensorImage(img)


def open_mask(label_path: Path, img_size: int) -> TensorMask:
    """Load YOLO labels and rasterize to binary mask at img_size."""
    _, bboxes = load_labels(label_path)
    mask = rasterize_boxes(bboxes, img_size)
    return TensorMask(mask)


def label_func(img_path: Path, data_dir: Path) -> Path:
    """Map image path to YOLO label path."""
    return data_dir / "labels" / f"{img_path.stem}.txt"


def build_dataloaders(
    data_dir: Path,
    img_size: int = 800,
    bs: int = 6,
    batch_tfms: list | None = None,
    num_workers: int = 0,
) -> DataLoaders:
    """Build fastai DataLoaders using DataBlock for SKW segmentation."""
    split_map = parse_splits(data_dir)
    images_dir = data_dir / "images"

    # Collect all image paths (train + test go to training, val to validation)
    def get_items(source):
        return sorted(
            f
            for f in images_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def is_val(img_path):
        return split_map.get(img_path.stem) == "val"

    open_img_func = partial(open_img, img_size=img_size)
    open_mask_func = partial(open_mask, img_size=img_size)
    label_func_bound = partial(label_func, data_dir=data_dir)

    dblock = DataBlock(
        blocks=[
            TransformBlock([open_img_func]),
            TransformBlock([open_mask_func]),
        ],
        get_items=get_items,
        get_y=label_func_bound,
        splitter=FuncSplitter(is_val),
        batch_tfms=batch_tfms or [],
    )

    dls = dblock.dataloaders(
        source=data_dir,
        bs=bs,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dls
