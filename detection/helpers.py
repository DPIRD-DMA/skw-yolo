"""Detection-specific visualization helpers."""

from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch


def plot_detection_batch(
    batch,
    image_idx: int = 0,
    class_names: Optional[dict[int, str]] = None,
    conf_threshold: float = 0.0,
):
    """Plot images from a detection batch with bounding boxes.

    Args:
        batch: (images [B,3,H,W], targets dict{batch_idx, cls, bboxes})
        image_idx: which image in the batch to display
        class_names: optional mapping {class_id: name}
        conf_threshold: minimum confidence for predicted boxes (0.0 for GT)
    """
    images, targets = batch
    if class_names is None:
        class_names = {0: "skw_0S", 1: "skw_1R"}

    colors = ["#FF4444", "#44FF44", "#4444FF", "#FFFF44"]

    img = images[image_idx].cpu()
    _, H, W = img.shape

    # Normalize to [0,1] for display
    rgb = img[:3].numpy().transpose(1, 2, 0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rgb)

    # Get bboxes for this image
    batch_idx = targets["batch_idx"]
    mask = batch_idx == image_idx
    cls = targets["cls"][mask].squeeze(-1).cpu().numpy()
    bboxes = targets["bboxes"][mask].cpu().numpy()  # xywh normalized

    for j in range(len(cls)):
        c = int(cls[j])
        xc, yc, w, h = bboxes[j]
        # Convert from normalized xywh to pixel xyxy
        x1 = (xc - w / 2) * W
        y1 = (yc - h / 2) * H
        box_w = w * W
        box_h = h * H

        color = colors[c % len(colors)]
        rect = patches.Rectangle(
            (x1, y1),
            box_w,
            box_h,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        label = class_names.get(c, str(c))
        ax.text(
            x1,
            y1 - 2,
            label,
            color=color,
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
        )

    n_boxes = len(cls)
    ax.set_title(f"Image {image_idx} | {n_boxes} boxes | size {H}x{W}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"Image shape: {images.shape}")
    print(f"Total boxes in batch: {len(targets['batch_idx'])}")
    print(f"Boxes in this image: {n_boxes}")


def plot_channel_histograms(img: torch.Tensor, title: str = "Per-channel distributions"):
    """Print per-channel stats and plot RGB histograms for a single image.

    Args:
        img: [3, H, W] tensor (any range).
        title: suptitle for the histogram figure.
    """
    img = img.detach().cpu()
    channel_names = ["Red", "Green", "Blue"]

    print(f"Image stats ({img.shape[1]}x{img.shape[2]})")
    print(f"{'Channel':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 52)
    for ch in range(img.shape[0]):
        c = img[ch]
        print(
            f"{channel_names[ch]:<10} {c.mean().item():>10.4f} {c.std().item():>10.4f}"
            f" {c.min().item():>10.4f} {c.max().item():>10.4f}"
        )
    print("-" * 52)
    print(
        f"{'All':.<10} {img.mean().item():>10.4f} {img.std().item():>10.4f}"
        f" {img.min().item():>10.4f} {img.max().item():>10.4f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors_hist = ["red", "green", "blue"]
    for ch, (ax, name, color) in enumerate(zip(axes, channel_names, colors_hist)):
        vals = img[ch].numpy().flatten()
        ax.hist(vals, bins=100, color=color, alpha=0.7)
        ax.set_title(f"{name} (mean={vals.mean():.3f}, std={vals.std():.3f})")
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Count")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()
