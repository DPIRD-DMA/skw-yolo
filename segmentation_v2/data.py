"""Data loading for SKW segmentation using fastai DataBlock."""

import random
import sys
import threading
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

# Thread-local state to share random placement between image and mask loaders
_crop_state = threading.local()


def rasterize_boxes(bboxes: torch.Tensor, img_size: int, shape: str = "box") -> torch.Tensor:
    """Convert normalized xywh boxes to a binary mask [H, W].

    Args:
        bboxes: [N, 4] normalized xywh bounding boxes.
        img_size: Output mask size (square).
        shape: "box" for rectangles, "ellipse" for inscribed ellipses.
    """
    if len(bboxes) == 0:
        return torch.zeros(img_size, img_size, dtype=torch.long)

    rows = torch.arange(img_size)
    cols = torch.arange(img_size)

    if shape == "ellipse":
        # Center and radii in pixel coords
        cx = (bboxes[:, 0] * img_size).float()  # [N]
        cy = (bboxes[:, 1] * img_size).float()
        rx = (bboxes[:, 2] / 2 * img_size).float()
        ry = (bboxes[:, 3] / 2 * img_size).float()
        # Normalized distance: ((col - cx)/rx)^2 + ((row - cy)/ry)^2 <= 1
        dy = (rows.float().unsqueeze(0) - cy.unsqueeze(1)) / ry.unsqueeze(1)  # [N, H]
        dx = (cols.float().unsqueeze(0) - cx.unsqueeze(1)) / rx.unsqueeze(1)  # [N, W]
        dist = dy.unsqueeze(2) ** 2 + dx.unsqueeze(1) ** 2  # [N, H, W]
        mask = (dist <= 1.0).any(dim=0).long()
    else:
        # Vectorized coordinate computation: [N, 4] -> pixel xyxy clamped to image
        x1 = ((bboxes[:, 0] - bboxes[:, 2] / 2) * img_size).clamp(0, img_size).int()
        y1 = ((bboxes[:, 1] - bboxes[:, 3] / 2) * img_size).clamp(0, img_size).int()
        x2 = ((bboxes[:, 0] + bboxes[:, 2] / 2) * img_size).clamp(0, img_size).int()
        y2 = ((bboxes[:, 1] + bboxes[:, 3] / 2) * img_size).clamp(0, img_size).int()
        in_y = (rows.unsqueeze(0) >= y1.unsqueeze(1)) & (rows.unsqueeze(0) < y2.unsqueeze(1))
        in_x = (cols.unsqueeze(0) >= x1.unsqueeze(1)) & (cols.unsqueeze(0) < x2.unsqueeze(1))
        mask = (in_y.unsqueeze(2) & in_x.unsqueeze(1)).any(dim=0).long()
    return mask


def open_img(img_path: Path, canvas_size: int) -> TensorImage:
    """Open image at native resolution, place at random location on canvas.

    Image is loaded without resizing. If smaller than canvas_size, it is placed
    at a random offset within a zero-padded canvas. If larger, a random crop
    is taken. The offset/crop is stored in thread-local state so open_mask
    can use the same placement.
    """
    img = Image.open(img_path).convert("RGB")
    img = TF.to_tensor(img)  # [3, H, W] float32 [0, 1]
    _, h, w = img.shape

    _crop_state.img_h = h
    _crop_state.img_w = w

    if h > canvas_size or w > canvas_size:
        # Random crop from larger image
        crop_top = random.randint(0, max(0, h - canvas_size))
        crop_left = random.randint(0, max(0, w - canvas_size))
        _crop_state.mode = "crop"
        _crop_state.crop_top = crop_top
        _crop_state.crop_left = crop_left
        img = img[:, crop_top : crop_top + canvas_size, crop_left : crop_left + canvas_size]
        return TensorImage(img)

    if h == canvas_size and w == canvas_size:
        _crop_state.mode = "exact"
        return TensorImage(img)

    # Smaller than canvas — place at random offset on zero-padded canvas
    top = random.randint(0, canvas_size - h)
    left = random.randint(0, canvas_size - w)
    _crop_state.mode = "pad"
    _crop_state.top = top
    _crop_state.left = left
    canvas = torch.zeros(3, canvas_size, canvas_size, dtype=img.dtype)
    canvas[:, top : top + h, left : left + w] = img
    return TensorImage(canvas)


def open_mask(label_path: Path, canvas_size: int, ignore_index: int, shape: str = "box") -> TensorMask:
    """Load YOLO labels, rasterize at native resolution, place on canvas.

    Mask is rasterized at the image's native size, then cropped or padded
    to match the same placement used for the image. Padding regions are
    filled with ignore_index.
    """
    _, bboxes = load_labels(label_path)

    h = _crop_state.img_h
    w = _crop_state.img_w

    # Rasterize at native image size (use min dimension for square assumption)
    native_size = min(h, w)
    mask = rasterize_boxes(bboxes, native_size, shape=shape)

    if _crop_state.mode == "crop":
        # Random crop — same region as image
        ct = _crop_state.crop_top
        cl = _crop_state.crop_left
        mask = mask[ct : ct + canvas_size, cl : cl + canvas_size]
        return TensorMask(mask)

    if _crop_state.mode == "exact":
        return TensorMask(mask)

    # Pad mode — place on ignore-filled canvas at same offset as image
    top = _crop_state.top
    left = _crop_state.left
    canvas = torch.full(
        (canvas_size, canvas_size), fill_value=ignore_index, dtype=mask.dtype
    )
    canvas[top : top + h, left : left + w] = mask
    return TensorMask(canvas)


def label_func(img_path: Path, data_dir: Path) -> Path:
    """Map image path to YOLO label path."""
    return data_dir / "labels" / f"{img_path.stem}.txt"


def build_dataloaders(
    data_dir: Path,
    canvas_size: int = 1000,
    ignore_index: int = 99,
    bs: int = 6,
    batch_tfms: list | None = None,
    num_workers: int = 0,
    shape: str = "box",
) -> DataLoaders:
    """Build fastai DataLoaders using DataBlock for SKW segmentation.

    Images are loaded at native resolution and placed at random offsets on a
    canvas_size x canvas_size tensor. Mask padding is filled with ignore_index.
    """
    split_map = parse_splits(data_dir)
    images_dir = data_dir / "images"

    def get_items(source):
        return sorted(
            f
            for f in images_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def is_val(img_path):
        return split_map.get(img_path.stem) == "val"

    open_img_func = partial(open_img, canvas_size=canvas_size)
    open_mask_func = partial(open_mask, canvas_size=canvas_size, ignore_index=ignore_index, shape=shape)
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
