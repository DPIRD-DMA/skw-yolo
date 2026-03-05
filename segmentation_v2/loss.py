"""Loss functions and training callbacks for binary segmentation."""

import torch
import torch.nn.functional as F
from fastai.callback.core import Callback

# Cached erosion kernel
_EROSION_KERNEL = None


def _get_erosion_kernel(device: torch.device) -> torch.Tensor:
    """Get or create the 3x3 erosion kernel (cached per device)."""
    global _EROSION_KERNEL
    if _EROSION_KERNEL is None or _EROSION_KERNEL.device != device:
        _EROSION_KERNEL = torch.ones(1, 1, 3, 3, device=device)
    return _EROSION_KERNEL


def _erode_distance(
    region: torch.Tensor, clip_distance: int, kernel: torch.Tensor
) -> torch.Tensor:
    """Compute erosion-based distance field for a binary region [B, 1, H, W].

    Returns distance map [B, H, W] with values in [0, clip_distance].
    """
    pad_size = clip_distance + 1
    padded = F.pad(region, (pad_size, pad_size, pad_size, pad_size), mode="replicate")

    current = padded
    dist_map = torch.zeros_like(padded)

    for d in range(1, clip_distance + 1):
        eroded = F.conv2d(current, kernel, padding=1)
        eroded = (eroded >= 9).float()
        at_d = current - eroded
        dist_map += at_d * d
        current = eroded

    # Interior pixels (survived all erosions) get max distance
    dist_map += current * clip_distance

    return dist_map[:, :, pad_size:-pad_size, pad_size:-pad_size].squeeze(1)


def center_distance_weights(
    mask: torch.Tensor,
    clip_distance: int = 3,
    class_ramps: dict[int, tuple[float, str]] | None = None,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Compute per-class distance-weighted pixel weights from a mask.

    Uses iterative morphological erosion (GPU-native) to compute distance
    from boundary for each specified class, then maps to weights.

    Args:
        mask: Mask [B, H, W] with class labels and optionally ignore_index.
        clip_distance: Max erosion iterations (determines gradient depth).
        class_ramps: Per-class weighting config. Dict mapping class_id to
            (max_weight, direction). direction is "center" (interior=max,
            boundary=1.0) or "edge" (boundary=max, interior=1.0).
            Default: {1: (5.0, "center")}.
        ignore_index: Label value to ignore (padding regions).

    Returns:
        Weight map [B, H, W] with values in [1.0, max(max_weights)].
    """
    if class_ramps is None:
        class_ramps = {1: (5.0, "center")}

    device = mask.device
    kernel = _get_erosion_kernel(device)
    weights = torch.ones(mask.shape, dtype=torch.float32, device=device)

    for cls_id, (max_w, direction) in class_ramps.items():
        cls_region = (mask == cls_id).float().unsqueeze(1)  # [B, 1, H, W]
        if cls_region.sum() == 0:
            continue

        dist_map = _erode_distance(cls_region, clip_distance, kernel)
        cls_mask = mask == cls_id

        if direction == "center":
            # boundary=1.0, interior=max_w
            weights[cls_mask] = 1.0 + (dist_map[cls_mask] / clip_distance) * (
                max_w - 1.0
            )
        else:  # "edge"
            # boundary=max_w, interior=1.0
            weights[cls_mask] = max_w - (dist_map[cls_mask] / clip_distance) * (
                max_w - 1.0
            )

    return weights


class DiceCELoss:
    """Combined Dice + center-weighted CE + centroid heatmap loss.

    Predictions: [B, 3, H, W] — channels 0-1 are seg logits, channel 2 is centroid logits.
    Targets: [B, 2, H, W] — channel 0 is seg mask (float), channel 1 is centroid heatmap.

    Args:
        dice_weight: Multiplier for the Dice loss term.
        ce_weight: Multiplier for the CE loss term.
        centroid_weight: Multiplier for the centroid heatmap loss term.
        class_weights: Optional per-class weights for CE (e.g. [1.0, 10.0]).
        clip_distance: Erosion depth for center-distance weighting.
        class_ramps: Per-class spatial weighting. Dict mapping class_id to
            (max_weight, direction). direction is "center" (interior=max) or
            "edge" (boundary=max). Default: {1: (5.0, "center")}.
        ignore_index: Label value to ignore in loss computation (padding regions).
    """

    def __init__(
        self,
        dice_weight: float = 10.0,
        ce_weight: float = 1.0,
        centroid_weight: float = 1.0,
        centroid_pos_weight: float = 20.0,
        class_weights: list[float] | None = None,
        clip_distance: int = 3,
        class_ramps: dict[int, tuple[float, str]] | None = None,
        ignore_index: int | None = None,
    ):
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.centroid_weight = centroid_weight
        self.centroid_pos_weight = centroid_pos_weight
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.clip_distance = clip_distance
        self.class_ramps = class_ramps
        self.ignore_index = ignore_index

    def __call__(self, pred, targ):
        # Split multi-channel target: [B, 2, H, W] → seg mask + centroid heatmap
        seg_mask = targ[:, 0].long()
        centroid_target = targ[:, 1]

        # Split predictions: [B, 3, H, W] → seg logits + centroid logits
        seg_logits = pred[:, :2]
        centroid_logits = pred[:, 2]

        # --- Segmentation loss ---
        w = self.class_weights
        if w is not None:
            w = w.to(pred.device)

        ignore = self.ignore_index if self.ignore_index is not None else -100

        if self.clip_distance > 0 and self.class_ramps is not None:
            with torch.no_grad():
                pixel_weights = center_distance_weights(
                    seg_mask, self.clip_distance, self.class_ramps, self.ignore_index
                )
            ce_unreduced = F.cross_entropy(
                seg_logits, seg_mask, weight=w, reduction="none", ignore_index=ignore,
            )
            ce = (ce_unreduced * pixel_weights).mean()
        else:
            ce = F.cross_entropy(
                seg_logits, seg_mask, weight=w, ignore_index=ignore,
            )

        pred_fg = F.softmax(seg_logits, dim=1)[:, 1]  # [B, H, W]
        targ_fg = (seg_mask == 1).float()

        if self.ignore_index is not None:
            valid = seg_mask != self.ignore_index
            pred_fg = pred_fg[valid]
            targ_fg = targ_fg[valid]

        inter = (pred_fg * targ_fg).sum()
        union = pred_fg.sum() + targ_fg.sum()
        dice = (2 * inter + 1) / (union + 1)

        seg_loss = self.ce_weight * ce + self.dice_weight * (1 - dice)

        # --- Centroid heatmap loss (BCE with pos_weight) ---
        if self.ignore_index is not None:
            valid_cent = seg_mask != self.ignore_index
            cent_logits_flat = centroid_logits[valid_cent]
            cent_target_flat = centroid_target[valid_cent]
        else:
            cent_logits_flat = centroid_logits.flatten()
            cent_target_flat = centroid_target.flatten()

        # Continuous pos_weight proportional to target intensity:
        # background (0) → 1.0, peak (1.0) → centroid_pos_weight, smooth in between
        pw = 1.0 + cent_target_flat * (self.centroid_pos_weight - 1.0)
        centroid_loss = F.binary_cross_entropy_with_logits(
            cent_logits_flat, cent_target_flat, weight=pw
        )

        return seg_loss + self.centroid_weight * centroid_loss


class EMACallback(Callback):
    """Exponential Moving Average of model weights.

    Maintains float32 shadow weights updated each training step.
    Swaps in EMA weights for validation, reverts for training.
    Replaces model with EMA version after training completes.
    """

    order = 65

    def __init__(self, decay: float = 0.9999):
        self.decay = decay

    def before_fit(self):
        self.ema_state = {
            k: v.clone().float() for k, v in self.learn.model.state_dict().items()
        }

    def after_batch(self):
        if self.training:
            d = self.decay
            with torch.no_grad():
                for k, v in self.learn.model.state_dict().items():
                    self.ema_state[k].mul_(d).add_(v.float(), alpha=1 - d)

    def before_validate(self):
        self._saved = {
            k: v.clone() for k, v in self.learn.model.state_dict().items()
        }
        self.learn.model.load_state_dict(
            {k: v.to(self._saved[k].dtype) for k, v in self.ema_state.items()}
        )

    def after_validate(self):
        self.learn.model.load_state_dict(self._saved)
        del self._saved

    def after_fit(self):
        self.learn.model.load_state_dict(
            {
                k: v.to(self.learn.model.state_dict()[k].dtype)
                for k, v in self.ema_state.items()
            }
        )
