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


def center_distance_weights(
    mask: torch.Tensor,
    clip_distance: int = 3,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Compute center-weighted pixel weights from a binary mask.

    Uses iterative morphological erosion (GPU-native) to compute distance
    from boundary. Interior pixels (far from edges) get max_weight,
    boundary pixels get 1.0, background pixels get 1.0.

    Args:
        mask: Binary mask [B, H, W] with values {0, 1}.
        clip_distance: Max erosion iterations (determines gradient depth).
        max_weight: Weight at the deepest interior pixels.

    Returns:
        Weight map [B, H, W] with values in [1.0, max_weight].
    """
    device = mask.device
    kernel = _get_erosion_kernel(device)

    fg = (mask == 1).float().unsqueeze(1)  # [B, 1, H, W]

    # Pad to handle edge artifacts
    pad_size = clip_distance + 1
    fg_padded = F.pad(fg, (pad_size, pad_size, pad_size, pad_size), mode="replicate")

    # Iterative erosion to compute distance from boundary
    current = fg_padded
    dist_map = torch.zeros_like(fg_padded)

    for d in range(1, clip_distance + 1):
        eroded = F.conv2d(current, kernel, padding=1)
        eroded = (eroded >= 9).float()
        at_distance_d = current - eroded
        dist_map += at_distance_d * d
        current = eroded

    # Interior pixels (survived all erosions) get max distance
    dist_map += current * clip_distance

    # Crop back to original size
    dist_map = dist_map[:, :, pad_size:-pad_size, pad_size:-pad_size].squeeze(1)

    # Map distance to weight: 0 (boundary) → 1.0, clip_distance (interior) → max_weight
    # Background (dist=0 and mask=0) stays at 1.0
    weights = torch.ones_like(dist_map)
    fg_mask = mask == 1
    weights[fg_mask] = 1.0 + (dist_map[fg_mask] / clip_distance) * (max_weight - 1.0)

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
        max_weight: Max pixel weight at bbox centers.
    """

    def __init__(
        self,
        dice_weight: float = 10.0,
        ce_weight: float = 1.0,
        class_weights: list[float] | None = None,
        clip_distance: int = 3,
        max_weight: float = 5.0,
    ):
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.clip_distance = clip_distance
        self.max_weight = max_weight

    def __call__(self, pred, targ):
        targ = targ.long()

        # Compute center-distance pixel weights from mask
        with torch.no_grad():
            pixel_weights = center_distance_weights(
                targ, self.clip_distance, self.max_weight
            )

        # Cross-entropy with class weights and spatial pixel weights
        w = self.class_weights
        if w is not None:
            w = w.to(pred.device)

        ce_unreduced = F.cross_entropy(pred, targ, weight=w, reduction="none")
        ce = (ce_unreduced * pixel_weights).mean()

        # Dice (foreground class only)
        pred_fg = F.softmax(pred, dim=1)[:, 1]  # [B, H, W]
        targ_fg = (targ == 1).float()

        inter = (pred_fg * targ_fg).sum()
        union = pred_fg.sum() + targ_fg.sum()
        dice = (2 * inter + 1) / (union + 1)

        return self.ce_weight * ce + self.dice_weight * (1 - dice)
