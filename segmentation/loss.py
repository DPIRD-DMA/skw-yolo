"""Loss functions for binary segmentation."""

import torch
import torch.nn.functional as F


class DiceCenterWeightedCELoss:
    """Combined Dice + center-weighted CE loss for binary segmentation.

    CE is weighted per-pixel using a center weight map (high at bbox centers,
    fading to 1.0 at edges). Dice focuses on foreground overlap.

    Args:
        dice_weight: Multiplier for the Dice loss term.
        ce_weight: Multiplier for the CE loss term.
        class_weights: Optional per-class weights for CE (e.g. [1.0, 10.0]
            to upweight foreground). Moved to device automatically.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        class_weights: list[float] | None = None,
    ):
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )

    def __call__(self, pred, targ, pixel_weights=None):
        targ = targ.long()

        # Cross-entropy (with optional class weights + spatial pixel weights)
        w = self.class_weights
        if w is not None:
            w = w.to(pred.device)

        if pixel_weights is not None:
            # Per-pixel weighted CE: compute unreduced, multiply by spatial weights, then mean
            ce_unreduced = F.cross_entropy(pred, targ, weight=w, reduction="none")
            ce = (ce_unreduced * pixel_weights.to(pred.device)).mean()
        else:
            ce = F.cross_entropy(pred, targ, weight=w)

        # Dice (foreground class only)
        pred_fg = F.softmax(pred, dim=1)[:, 1]  # [B, H, W]
        targ_fg = (targ == 1).float()

        inter = (pred_fg * targ_fg).sum()
        union = pred_fg.sum() + targ_fg.sum()
        dice = (2 * inter + 1) / (union + 1)  # +1 smooth

        return self.ce_weight * ce + self.dice_weight * (1 - dice)
