"""Loss functions for binary segmentation with center-distance weighting."""

import torch
import torch.nn.functional as F

# Cached erosion kernel
_EROSION_KERNEL = None


def _get_erosion_kernel(device: torch.device) -> torch.Tensor:
    """Get or create the 3x3 erosion kernel (cached per device)."""
    global _EROSION_KERNEL
    if _EROSION_KERNEL is None or _EROSION_KERNEL.device != device:
        _EROSION_KERNEL = torch.ones(1, 1, 3, 3, device=device)
    return _EROSION_KERNEL


def _erode_distance(region: torch.Tensor, clip_distance: int, kernel: torch.Tensor) -> torch.Tensor:
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
            weights[cls_mask] = 1.0 + (dist_map[cls_mask] / clip_distance) * (max_w - 1.0)
        else:  # "edge"
            # boundary=max_w, interior=1.0
            weights[cls_mask] = max_w - (dist_map[cls_mask] / clip_distance) * (max_w - 1.0)

    return weights


class DiceCELoss:
    """Combined Dice + center-weighted CE loss for binary segmentation.

    Center-distance weighting is computed from the mask inside the loss
    function (no weight maps needed in the data pipeline).

    Args:
        dice_weight: Multiplier for the Dice loss term.
        ce_weight: Multiplier for the CE loss term.
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
        class_weights: list[float] | None = None,
        clip_distance: int = 3,
        class_ramps: dict[int, tuple[float, str]] | None = None,
        ignore_index: int | None = None,
    ):
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.clip_distance = clip_distance
        self.class_ramps = class_ramps
        self.ignore_index = ignore_index

    def __call__(self, pred, targ):
        targ = targ.long()

        # Compute center-distance pixel weights from mask
        with torch.no_grad():
            pixel_weights = center_distance_weights(
                targ, self.clip_distance, self.class_ramps, self.ignore_index
            )

        # Cross-entropy with class weights and spatial pixel weights
        w = self.class_weights
        if w is not None:
            w = w.to(pred.device)

        ce_unreduced = F.cross_entropy(
            pred, targ, weight=w, reduction="none",
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
        )
        ce = (ce_unreduced * pixel_weights).mean()

        # Dice (foreground class only, excluding ignore pixels)
        pred_fg = F.softmax(pred, dim=1)[:, 1]  # [B, H, W]
        targ_fg = (targ == 1).float()

        if self.ignore_index is not None:
            valid = targ != self.ignore_index
            pred_fg = pred_fg[valid]
            targ_fg = targ_fg[valid]

        inter = (pred_fg * targ_fg).sum()
        union = pred_fg.sum() + targ_fg.sum()
        dice = (2 * inter + 1) / (union + 1)

        return self.ce_weight * ce + self.dice_weight * (1 - dice)
