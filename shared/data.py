"""Shared data loading utilities for SKW dataset."""

from pathlib import Path, PureWindowsPath

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def parse_splits(data_dir: Path) -> dict[str, str]:
    """Parse train/val/test split files, matching Windows paths to local filenames.

    Returns {image_stem: split_name}. Priority: test > val > train.
    Unmatched images are assigned to train.
    """
    images_dir = data_dir / "images"
    actual_stems = {
        f.stem: f
        for f in images_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    split_map: dict[str, str] = {}
    # Process in priority order so test > val > train
    for split_name in ["train", "val", "test"]:
        split_file = data_dir / f"{split_name}.txt"
        if not split_file.exists():
            continue
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stem = PureWindowsPath(line).stem
                if stem in actual_stems:
                    split_map[stem] = split_name

    # Assign unmatched images to train
    for stem in actual_stems:
        if stem not in split_map:
            split_map[stem] = "train"

    return split_map


def load_labels(label_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load YOLO format labels.

    Returns (classes [N,1], bboxes [N,4]) or empty tensors if no labels.
    """
    if not label_path.exists():
        return torch.zeros(0, 1), torch.zeros(0, 4)
    content = label_path.read_text().strip()
    if not content:
        return torch.zeros(0, 1), torch.zeros(0, 4)

    data = []
    for line in content.split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = float(parts[0])
        xc, yc, w, h = (
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
        data.append([cls, xc, yc, w, h])

    if not data:
        return torch.zeros(0, 1), torch.zeros(0, 4)

    t = torch.tensor(data, dtype=torch.float32)
    return t[:, :1], t[:, 1:]  # cls [N,1], bboxes [N,4]


def letterbox(img: torch.Tensor, target_size: int, pad_value: float = 114 / 255):
    """Resize image preserving aspect ratio and pad to target_size.

    Args:
        img: [3, H, W] float32 tensor
        target_size: output will be [3, target_size, target_size]
        pad_value: fill value for padding (114/255 matches ultralytics gray)

    Returns:
        padded_img: [3, target_size, target_size]
        scale: resize scale factor
        pad_w: left padding in pixels
        pad_h: top padding in pixels
    """
    _, h, w = img.shape
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    img = TF.resize(img, [new_h, new_w], antialias=True)

    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    pad_bottom = target_size - new_h - pad_h
    pad_right = target_size - new_w - pad_w

    # F.pad order: (left, right, top, bottom)
    img = F.pad(img, (pad_w, pad_right, pad_h, pad_bottom), value=pad_value)

    return img, scale, pad_w, pad_h


def adjust_bboxes_for_letterbox(
    bboxes: torch.Tensor,
    orig_w: int,
    orig_h: int,
    target_size: int,
    scale: float,
    pad_w: int,
    pad_h: int,
) -> torch.Tensor:
    """Convert normalised xywh bboxes from original image space to letterboxed space.

    Input bboxes are normalised to [0,1] relative to orig image.
    Output bboxes are normalised to [0,1] relative to the padded target_size image.
    """
    if bboxes.shape[0] == 0:
        return bboxes
    bboxes = bboxes.clone()
    # Convert from original-normalised to pixel coords in resized space
    bboxes[:, 0] = bboxes[:, 0] * orig_w * scale + pad_w  # xc
    bboxes[:, 1] = bboxes[:, 1] * orig_h * scale + pad_h  # yc
    bboxes[:, 2] = bboxes[:, 2] * orig_w * scale  # w
    bboxes[:, 3] = bboxes[:, 3] * orig_h * scale  # h
    # Normalise to target_size
    bboxes[:, 0] /= target_size
    bboxes[:, 1] /= target_size
    bboxes[:, 2] /= target_size
    bboxes[:, 3] /= target_size
    return bboxes
