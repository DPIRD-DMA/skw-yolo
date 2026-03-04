"""Visualization and validation helpers for SKW segmentation v2."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

from metrics import _count_peaks, _gaussian_blur


def plot_batch(images, masks, centroids, n_show=4, downsample=4):
    """Visualize a training batch: RGB | R | G | B | Mask | Centroid | Overlay."""
    n_show = min(n_show, images.shape[0])
    s = downsample
    num_cols = 7
    fig, axs = plt.subplots(n_show, num_cols, figsize=(5 * num_cols, 4.5 * n_show))
    if n_show == 1:
        axs = np.expand_dims(axs, axis=0)

    channel_names = ["Red", "Green", "Blue"]
    for row in range(n_show):
        img_float = images[row, :, ::s, ::s].cpu().float()
        rgb = img_float[:3].numpy().transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        mask = masks[row, ::s, ::s].cpu().numpy()
        cent = centroids[row, ::s, ::s].cpu().numpy()

        axs[row, 0].imshow(rgb)
        if row == 0:
            axs[row, 0].set_title("RGB")
        axs[row, 0].axis("off")

        for ch in range(3):
            axs[row, ch + 1].imshow(img_float[ch].numpy(), cmap="gray")
            if row == 0:
                axs[row, ch + 1].set_title(channel_names[ch])
            axs[row, ch + 1].axis("off")

        axs[row, 4].imshow(mask, cmap="gray", interpolation="nearest", vmin=0, vmax=2)
        if row == 0:
            axs[row, 4].set_title("Mask")
        axs[row, 4].axis("off")

        axs[row, 5].imshow(cent, cmap="hot", interpolation="nearest", vmin=0, vmax=1)
        if row == 0:
            axs[row, 5].set_title("Centroids")
        axs[row, 5].axis("off")

        axs[row, 6].imshow(rgb)
        axs[row, 6].imshow(
            mask, alpha=0.4, cmap="Reds", interpolation="nearest", vmin=0, vmax=2
        )
        peak_y, peak_x = np.where(cent > 0.5)
        axs[row, 6].scatter(peak_x, peak_y, c="cyan", s=20, marker="x", linewidths=1)
        if row == 0:
            axs[row, 6].set_title("Overlay")
        axs[row, 6].axis("off")

    plt.tight_layout()
    plt.show()


def plot_centroid_targets(images, masks, centroids, centroid_sigma, nms_kernel, n_show=4):
    """Visualize centroid targets: RGB + seg mask outline + centroid heatmap."""
    n_show = min(n_show, images.shape[0])
    fig, axs = plt.subplots(n_show, 3, figsize=(18, 5 * n_show))
    if n_show == 1:
        axs = np.expand_dims(axs, axis=0)

    for row in range(n_show):
        img = images[row].cpu().float().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        mask = masks[row].cpu().numpy()
        cent = centroids[row].cpu().numpy()

        axs[row, 0].imshow(img)
        if mask.max() > 0:
            axs[row, 0].contour(mask, levels=[0.5], colors=["#FF4444"], linewidths=2)
        axs[row, 0].set_title("Image + seg mask" if row == 0 else "")
        axs[row, 0].axis("off")

        axs[row, 1].imshow(cent, cmap="hot", interpolation="nearest", vmin=0, vmax=1)
        if mask.max() > 0:
            axs[row, 1].contour(
                mask, levels=[0.5], colors=["cyan"], linewidths=1, linestyles="--"
            )
        axs[row, 1].set_title(
            f"Centroid target (sigma={centroid_sigma})" if row == 0 else ""
        )
        axs[row, 1].axis("off")

        peak_count = _count_peaks(centroids[row], threshold=0.5, nms_kernel=nms_kernel)
        axs[row, 2].imshow(img)
        axs[row, 2].imshow(cent, alpha=0.6, cmap="hot", vmin=0, vmax=1)
        if mask.max() > 0:
            axs[row, 2].contour(mask, levels=[0.5], colors=["cyan"], linewidths=2)
        axs[row, 2].set_title(
            f"Overlay ({peak_count} GT peaks)" if row == 0 else f"({peak_count} peaks)"
        )
        axs[row, 2].axis("off")

    fig.suptitle(
        f"Centroid ground truth targets (sigma={centroid_sigma}, NMS kernel={nms_kernel})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def plot_center_distance_weights(images, masks, weights, n_show=4):
    """Visualize center-distance weight maps overlaid on images."""
    n_show = min(n_show, images.shape[0])
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5), squeeze=False)
    for i in range(n_show):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        wmap = weights[i].cpu().numpy()

        axes[0, i].imshow(img)
        axes[0, i].imshow(wmap, alpha=0.5, cmap="hot", vmin=1.0, vmax=5)
        axes[0, i].set_title(f"Weight map {i} (max=5)")
        axes[0, i].axis("off")
    fig.suptitle("Center-distance weights (computed in loss function)", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_val_predictions(
    learner, dls, centroid_threshold, nms_kernel, n_images=8
):
    """Run inference on validation set and plot GT vs prediction side-by-side."""
    model = learner.model.cuda().float()
    model.eval()

    all_images, all_gt_mask, all_gt_cent = [], [], []
    all_pred_mask, all_pred_cent = [], []
    val_iter = iter(dls.valid)
    while len(all_images) < n_images:
        try:
            val_batch = next(val_iter)
            val_images, val_targets = val_batch[0], val_batch[1]
        except StopIteration:
            break
        with torch.no_grad():
            logits = model(val_images.cuda().float())
            seg_preds = logits[:, :2].argmax(dim=1).cpu()
            cent_preds = torch.sigmoid(logits[:, 2]).cpu()
        for i in range(val_images.shape[0]):
            if len(all_images) >= n_images:
                break
            all_images.append(val_images[i])
            all_gt_mask.append(val_targets[i, 0])
            all_gt_cent.append(val_targets[i, 1])
            all_pred_mask.append(seg_preds[i])
            all_pred_cent.append(cent_preds[i])

    for img_idx in range(len(all_images)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        img = all_images[img_idx].cpu().float().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        gt = all_gt_mask[img_idx].cpu().numpy()
        pred = all_pred_mask[img_idx].numpy()
        gt_cent = all_gt_cent[img_idx].cpu().numpy()
        pred_cent = all_pred_cent[img_idx].numpy()

        pred_cent_masked = pred_cent * pred.astype(np.float32)
        bg_overlay = np.ones_like(img)

        # GT peaks via NMS
        gt_count = _count_peaks(
            all_gt_cent[img_idx], threshold=0.5, nms_kernel=nms_kernel
        )
        gt_smoothed = (
            _gaussian_blur(all_gt_cent[img_idx].unsqueeze(0).unsqueeze(0))
            .squeeze()
            .cpu()
            .numpy()
        )
        gt_local_max = maximum_filter(gt_smoothed, size=nms_kernel) == gt_smoothed
        gt_peaks = gt_local_max & (gt_smoothed > 0.5)
        gt_py, gt_px = np.where(gt_peaks)

        # Pred peaks via NMS
        pred_cent_t = torch.from_numpy(pred_cent_masked)
        pred_count = _count_peaks(
            pred_cent_t, threshold=centroid_threshold, nms_kernel=nms_kernel
        )
        pred_smoothed = (
            _gaussian_blur(pred_cent_t.unsqueeze(0).unsqueeze(0))
            .squeeze()
            .cpu()
            .numpy()
        )
        pred_local_max = (
            maximum_filter(pred_smoothed, size=nms_kernel) == pred_smoothed
        )
        pred_peaks = pred_local_max & (pred_smoothed > centroid_threshold)
        pred_py, pred_px = np.where(pred_peaks)

        # Left: Ground truth
        ax1.imshow(img)
        ax1.imshow(bg_overlay, alpha=np.where(gt == 0, 0.4, 0.0))
        if gt.max() > 0:
            ax1.contour(gt, levels=[0.5], colors=["#FF4444"], linewidths=2)
        ax1.scatter(
            gt_px, gt_py, c="cyan", s=60, marker="o",
            edgecolors="black", linewidths=1, zorder=5,
        )
        ax1.set_title(f"Ground Truth ({gt_count} objects)")
        ax1.axis("off")

        # Right: Prediction
        ax2.imshow(img)
        ax2.imshow(bg_overlay, alpha=np.where(pred == 0, 0.4, 0.0))
        if pred.max() > 0:
            ax2.contour(pred, levels=[0.5], colors=["#4488FF"], linewidths=2)
        ax2.scatter(
            pred_px, pred_py, c="lime", s=60, marker="o",
            edgecolors="black", linewidths=1, zorder=5,
        )
        ax2.set_title(f"Prediction ({pred_count} objects)")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()
        print(f"Image {img_idx}: GT {gt_count} objects, Pred {pred_count} objects")
