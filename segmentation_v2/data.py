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


def rasterize_boxes(
    bboxes: torch.Tensor, img_size: int, shape: str = "box"
) -> torch.Tensor:
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
        in_y = (rows.unsqueeze(0) >= y1.unsqueeze(1)) & (
            rows.unsqueeze(0) < y2.unsqueeze(1)
        )
        in_x = (cols.unsqueeze(0) >= x1.unsqueeze(1)) & (
            cols.unsqueeze(0) < x2.unsqueeze(1)
        )
        mask = (in_y.unsqueeze(2) & in_x.unsqueeze(1)).any(dim=0).long()
    return mask


def rasterize_centroids(
    bboxes: torch.Tensor, img_size: int, sigma: float = 3.0
) -> torch.Tensor:
    """Convert normalized xywh box centers to a Gaussian centroid heatmap [H, W].

    Args:
        bboxes: [N, 4] normalized xywh bounding boxes.
        img_size: Output heatmap size (square).
        sigma: Gaussian standard deviation in pixels.

    Returns:
        Float32 heatmap [H, W] with values in [0, 1].
    """
    heatmap = torch.zeros(img_size, img_size, dtype=torch.float32)
    if len(bboxes) == 0:
        return heatmap

    radius = int(3 * sigma + 0.5)  # 3-sigma window

    for i in range(len(bboxes)):
        cx = int(bboxes[i, 0].item() * img_size)
        cy = int(bboxes[i, 1].item() * img_size)

        # Clamp window to image bounds
        y0 = max(0, cy - radius)
        y1 = min(img_size, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(img_size, cx + radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue

        # Local Gaussian
        yy = torch.arange(y0, y1, dtype=torch.float32) - cy
        xx = torch.arange(x0, x1, dtype=torch.float32) - cx
        gy = torch.exp(-(yy**2) / (2 * sigma**2))
        gx = torch.exp(-(xx**2) / (2 * sigma**2))
        gaussian = gy.unsqueeze(1) * gx.unsqueeze(0)  # [h, w]

        # Max-merge to keep values in [0, 1]
        heatmap[y0:y1, x0:x1] = torch.max(heatmap[y0:y1, x0:x1], gaussian)

    return heatmap


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
        img = img[
            :, crop_top : crop_top + canvas_size, crop_left : crop_left + canvas_size
        ]
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


def open_mask(
    label_path: Path,
    canvas_size: int,
    ignore_index: int,
    shape: str = "box",
    centroid_sigma: float = 3.0,
) -> TensorMask:
    """Load YOLO labels, rasterize seg mask + centroid heatmap, place on canvas.

    Returns a [2, H, W] float tensor:
      - Channel 0: seg mask (0.0, 1.0, or ignore_index as float)
      - Channel 1: centroid Gaussian heatmap (0.0 to 1.0)

    Padding regions: seg channel filled with ignore_index, centroid channel with 0.
    """
    _, bboxes = load_labels(label_path)

    h = _crop_state.img_h
    w = _crop_state.img_w

    # Rasterize at native image size (min handles non-square edge cases)
    native_size = min(h, w)
    mask = rasterize_boxes(bboxes, native_size, shape=shape).float()
    centroids = rasterize_centroids(bboxes, native_size, sigma=centroid_sigma)

    if _crop_state.mode == "crop":
        ct = _crop_state.crop_top
        cl = _crop_state.crop_left
        mask = mask[ct : ct + canvas_size, cl : cl + canvas_size]
        centroids = centroids[ct : ct + canvas_size, cl : cl + canvas_size]
        return TensorMask(torch.stack([mask, centroids], dim=0))

    if _crop_state.mode == "exact":
        return TensorMask(torch.stack([mask, centroids], dim=0))

    # Pad mode — seg channel gets ignore_index fill, centroid channel gets 0
    top = _crop_state.top
    left = _crop_state.left
    seg_canvas = torch.full(
        (canvas_size, canvas_size), fill_value=float(ignore_index), dtype=torch.float32
    )
    seg_canvas[top : top + h, left : left + w] = mask
    cent_canvas = torch.zeros(canvas_size, canvas_size, dtype=torch.float32)
    cent_canvas[top : top + h, left : left + w] = centroids
    return TensorMask(torch.stack([seg_canvas, cent_canvas], dim=0))


def collect_v2_items(
    data_v2_dir: Path,
) -> list[tuple[Path, Path, str]]:
    """Collect (image_path, label_path, split) from data_v2 hierarchical structure.

    Scans initial/tiled/ and extra/*/tiled/ for images organized in
    train/val/test subdirectories. Label paths may not exist (negative tiles).
    """
    items: list[tuple[Path, Path, str]] = []
    tiled_dirs: list[Path] = []

    initial = data_v2_dir / "initial" / "tiled"
    if initial.exists():
        tiled_dirs.append(initial)

    extra = data_v2_dir / "extra"
    if extra.exists():
        for batch in sorted(extra.iterdir()):
            if not batch.is_dir():
                continue
            tiled = batch / "tiled"
            if tiled.exists():
                tiled_dirs.append(tiled)

    valid_splits = {"train", "val", "test"}
    img_exts = {".jpg", ".jpeg", ".png"}

    for tiled in tiled_dirs:
        images_dir = tiled / "images"
        labels_dir = tiled / "labels"
        if not images_dir.exists():
            continue
        for split_dir in sorted(images_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split = split_dir.name
            if split not in valid_splits:
                continue
            for img_path in sorted(split_dir.iterdir()):
                if img_path.suffix.lower() in img_exts:
                    label_path = labels_dir / split / f"{img_path.stem}.txt"
                    items.append((img_path, label_path, split))

    return items


def _has_labels(label_path: Path) -> bool:
    """Return True if label file exists and has at least one object."""
    if not label_path.exists():
        return False
    return bool(label_path.read_text().strip())


def build_dataloaders(
    data_dir: Path,
    data_v2_dir: Path | None = None,
    positive_oversample: int = 1,
    exclude_stems: set[str] | None = None,
    canvas_size: int = 1000,
    ignore_index: int = 99,
    train_bs: int = 6,
    val_bs: int | None = None,
    batch_tfms: list | None = None,
    num_workers: int = 0,
    shape: str = "box",
    centroid_sigma: float = 3.0,
    train_pct: float = 1.0,
    val_pct: float = 1.0,
) -> DataLoaders:
    """Build fastai DataLoaders using DataBlock for SKW segmentation.

    Images are loaded at native resolution and placed at random offsets on a
    canvas_size x canvas_size tensor. Mask padding is filled with ignore_index.
    Target is a [2, H, W] float tensor: channel 0 = seg mask, channel 1 = centroid heatmap.

    When data_v2_dir is provided, also loads images from the hierarchical
    initial/tiled/ + extra/*/tiled/ structure. Duplicates (by stem) are skipped.

    Args:
        positive_oversample: Repeat positive (has-object) train images this many
            times to rebalance against empty/negative tiles. Each repetition sees
            different augmentations. E.g. 5 = positives appear 5x per epoch.
        exclude_stems: Set of image stems to skip (e.g. bad/corrupt images).
    """
    all_items: list[tuple[Path, Path, bool]] = []  # (img_path, label_path, is_val)
    seen_stems: set[str] = set(exclude_stems or ())

    # Original flat dataset (data_dir/images/ + data_dir/labels/)
    split_map = parse_splits(data_dir)
    images_dir = data_dir / "images"
    if images_dir.exists():
        for f in sorted(images_dir.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                label_path = data_dir / "labels" / f"{f.stem}.txt"
                _is_val = split_map.get(f.stem) == "val"
                all_items.append((f, label_path, _is_val))
                seen_stems.add(f.stem)

    # data_v2 hierarchical dataset (skip stems already in original)
    if data_v2_dir is not None:
        for img_path, label_path, split in collect_v2_items(data_v2_dir):
            if img_path.stem not in seen_stems:
                all_items.append((img_path, label_path, split == "val"))
                seen_stems.add(img_path.stem)

    # Oversample positive train images to rebalance against negatives
    if positive_oversample > 1:
        train_pos = [t for t in all_items if not t[2] and _has_labels(t[1])]
        train_neg = [t for t in all_items if not t[2] and not _has_labels(t[1])]
        val_items = [t for t in all_items if t[2]]

        oversampled = train_pos * positive_oversample + train_neg
        random.shuffle(oversampled)
        all_items = oversampled + val_items

        pos_pct = len(train_pos) * positive_oversample / len(oversampled) * 100
        print(
            f"Oversampling: {len(train_pos)} pos x{positive_oversample} "
            f"= {len(train_pos) * positive_oversample} + {len(train_neg)} neg "
            f"= {len(oversampled)} train ({pos_pct:.0f}% pos), {len(val_items)} val"
        )

    # Subsample training data for quick experimentation
    if train_pct < 1.0 or val_pct < 1.0:
        train_items = [t for t in all_items if not t[2]]
        val_items = [t for t in all_items if t[2]]
        if train_pct < 1.0:
            n_keep = max(1, int(len(train_items) * train_pct))
            random.shuffle(train_items)
            all_train = len(train_items)
            train_items = train_items[:n_keep]
            print(f"train_pct={train_pct}: using {n_keep}/{all_train} train images")
        if val_pct < 1.0:
            n_keep = max(1, int(len(val_items) * val_pct))
            random.shuffle(val_items)
            all_val = len(val_items)
            val_items = val_items[:n_keep]
            print(f"val_pct={val_pct}: using {n_keep}/{all_val} val images")
        all_items = train_items + val_items

    # Build lookup dicts keyed by image path string
    label_lookup = {str(img): lbl for img, lbl, _ in all_items}
    val_lookup = {str(img): is_val for img, _, is_val in all_items}

    def get_items(source):
        return [img for img, _, _ in all_items]

    def is_val(img_path):
        return val_lookup.get(str(img_path), False)

    def get_label(img_path):
        return label_lookup[str(img_path)]

    open_img_func = partial(open_img, canvas_size=canvas_size)
    open_mask_func = partial(
        open_mask,
        canvas_size=canvas_size,
        ignore_index=ignore_index,
        shape=shape,
        centroid_sigma=centroid_sigma,
    )

    dblock = DataBlock(
        blocks=[
            TransformBlock([open_img_func]),
            TransformBlock([open_mask_func]),
        ],
        get_items=get_items,
        get_y=get_label,
        splitter=FuncSplitter(is_val),
        batch_tfms=batch_tfms or [],
    )

    dls = dblock.dataloaders(
        source=data_dir,
        bs=train_bs,
        val_bs=val_bs or train_bs,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return dls
