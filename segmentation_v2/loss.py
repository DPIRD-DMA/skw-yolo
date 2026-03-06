"""Loss functions and training callbacks for binary segmentation."""

import torch
import torch.nn.functional as F
from fastai.callback.core import Callback


def lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovász-Softmax loss, flat version."""
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.shape[1]
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            fg_class = 1.0 - probas[:, 0]
        else:
            fg_class = probas[:, c]
        errors = (fg - fg_class).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=probas.device)

# Cached erosion kernel
_EROSION_KERNEL = None


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "none",
) -> torch.Tensor:
    """Focal cross-entropy loss: downweights easy examples, focuses on hard ones.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    """
    num_classes = logits.shape[1]
    # Replace ignore_index with 0 for gather (CE already handles ignore_index)
    safe_targets = targets.clone()
    if ignore_index >= 0:
        ignored = targets == ignore_index
        safe_targets[ignored] = 0

    ce = F.cross_entropy(logits, targets, weight=weight, ignore_index=ignore_index, reduction="none")
    p = F.softmax(logits, dim=1)
    p_t = p.gather(1, safe_targets.unsqueeze(1)).squeeze(1)

    # Ignored pixels get p_t=1 → focal_weight=0 → loss=0
    if ignore_index >= 0:
        p_t[ignored] = 1.0

    focal_weight = (1 - p_t) ** gamma
    loss = focal_weight * ce
    if reduction == "mean":
        return loss.mean()
    return loss


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
        lovasz_weight: float = 0.0,
        focal_gamma: float = 0.0,
        class_weights: list[float] | None = None,
        clip_distance: int = 3,
        class_ramps: dict[int, tuple[float, str]] | None = None,
        ignore_index: int | None = None,
    ):
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.centroid_weight = centroid_weight
        self.centroid_pos_weight = centroid_pos_weight
        self.lovasz_weight = lovasz_weight
        self.focal_gamma = focal_gamma
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

        if self.focal_gamma > 0:
            ce_unreduced = focal_cross_entropy(
                seg_logits, seg_mask, gamma=self.focal_gamma, weight=w, ignore_index=ignore,
            )
        else:
            ce_unreduced = F.cross_entropy(
                seg_logits, seg_mask, weight=w, reduction="none", ignore_index=ignore,
            )

        if self.clip_distance > 0 and self.class_ramps is not None:
            with torch.no_grad():
                pixel_weights = center_distance_weights(
                    seg_mask, self.clip_distance, self.class_ramps, self.ignore_index
                )
            ce = (ce_unreduced * pixel_weights).mean()
        else:
            ce = ce_unreduced.mean()

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

        # --- Lovász loss (directly optimizes IoU) ---
        if self.lovasz_weight > 0:
            probas = F.softmax(seg_logits, dim=1)  # [B, C, H, W]
            if self.ignore_index is not None:
                valid_lov = seg_mask != self.ignore_index
                # Flatten spatial dims, keeping classes
                probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, probas.shape[1])
                labels_flat = seg_mask.reshape(-1)
                valid_flat = valid_lov.reshape(-1)
                probas_flat = probas_flat[valid_flat]
                labels_flat = labels_flat[valid_flat]
            else:
                probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, probas.shape[1])
                labels_flat = seg_mask.reshape(-1)
            lovasz = lovasz_softmax_flat(probas_flat, labels_flat)
            seg_loss = seg_loss + self.lovasz_weight * lovasz

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
