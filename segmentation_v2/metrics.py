"""Segmentation metrics for SKW binary segmentation with centroid counting."""

import torch
import torch.nn.functional as F
from fastai.metrics import DiceMulti, JaccardCoeffMulti, Metric
from fastai.torch_core import flatten_check


class SegDiceMulti(DiceMulti):
    """Per-class Dice scores, ignoring padding pixels.

    Handles multi-channel targets: extracts channel 0 as seg mask.
    """

    def __init__(self, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

    def accumulate(self, learn):
        seg_logits = learn.pred[:, :2]
        seg_mask = learn.yb[0][:, 0]
        pred, targ = flatten_check(seg_logits.argmax(dim=self.axis), seg_mask)
        targ = targ.long()
        if self.ignore_index is not None:
            valid = targ != self.ignore_index
            pred = pred[valid]
            targ = targ[valid]
        for c in range(seg_logits.shape[self.axis]):
            p = torch.where(pred == c, 1, 0)
            t = torch.where(targ == c, 1, 0)
            c_inter = (p * t).float().sum().item()
            c_union = (p + t).float().sum().item()
            if c in self.inter:
                self.inter[c] += c_inter
                self.union[c] += c_union
            else:
                self.inter[c] = c_inter
                self.union[c] = c_union


class ForegroundIoU(JaccardCoeffMulti):
    """Foreground-only IoU (class 1), ignoring padding pixels.

    Handles multi-channel targets: extracts channel 0 as seg mask.
    """

    def __init__(self, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

    @property
    def name(self):
        return "fg_iou"

    def accumulate(self, learn):
        seg_logits = learn.pred[:, :2]
        seg_mask = learn.yb[0][:, 0]
        pred, targ = flatten_check(seg_logits.argmax(dim=self.axis), seg_mask)
        targ = targ.long()
        if self.ignore_index is not None:
            valid = targ != self.ignore_index
            pred = pred[valid]
            targ = targ[valid]
        c = 1  # foreground only
        p = torch.where(pred == c, 1, 0)
        t = torch.where(targ == c, 1, 0)
        c_inter = (p * t).float().sum().item()
        c_union = (p + t).float().sum().item()
        if c in self.inter:
            self.inter[c] += c_inter
            self.union[c] += c_union
        else:
            self.inter[c] = c_inter
            self.union[c] = c_union


class CentroidCountMAE(Metric):
    """Mean absolute error between predicted and ground-truth centroid counts.

    Counts GT centroids by thresholding heatmap peaks (>0.5).
    Counts predicted centroids via 3x3 local-max detection above a threshold.
    """

    def __init__(self, threshold: float = 0.3, nms_kernel: int = 41):
        self.threshold = threshold
        self.nms_kernel = nms_kernel
        self.reset()

    @property
    def name(self):
        return "cnt_mae"

    def reset(self):
        self.total_ae = 0.0
        self.n = 0

    def accumulate(self, learn):
        centroid_pred = torch.sigmoid(learn.pred[:, 2])  # [B, H, W]
        centroid_target = learn.yb[0][:, 1]  # [B, H, W]
        seg_pred = learn.pred[:, :2].argmax(dim=1)  # [B, H, W]

        for i in range(centroid_pred.shape[0]):
            gt_count = _count_peaks(
                centroid_target[i], threshold=0.5, nms_kernel=self.nms_kernel
            )
            # Mask centroid preds to predicted foreground to filter bg false positives
            masked_pred = centroid_pred[i] * seg_pred[i].float()
            pred_count = _count_peaks(
                masked_pred, threshold=self.threshold, nms_kernel=self.nms_kernel
            )

            self.total_ae += abs(pred_count - gt_count)
            self.n += 1

    @property
    def value(self):
        return self.total_ae / self.n if self.n > 0 else 0.0


class CentroidCountMAPE(Metric):
    """Mean absolute percentage error between predicted and GT centroid counts.

    Per-image: |pred - gt| / max(gt, 1). Images with gt=0 and pred=0 score 0%;
    images with gt=0 and pred>0 score 100%. Robust to "predict nothing" exploits
    unlike MAE, since missing all objects in an image = 100% error.
    """

    def __init__(self, threshold: float = 0.3, nms_kernel: int = 41):
        self.threshold = threshold
        self.nms_kernel = nms_kernel
        self.reset()

    @property
    def name(self):
        return "cnt_mape"

    def reset(self):
        self.total_ape = 0.0
        self.n = 0

    def accumulate(self, learn):
        centroid_pred = torch.sigmoid(learn.pred[:, 2])  # [B, H, W]
        centroid_target = learn.yb[0][:, 1]  # [B, H, W]
        seg_pred = learn.pred[:, :2].argmax(dim=1)  # [B, H, W]

        for i in range(centroid_pred.shape[0]):
            gt_count = _count_peaks(
                centroid_target[i], threshold=0.5, nms_kernel=self.nms_kernel
            )
            # Mask centroid preds to predicted foreground to filter bg false positives
            masked_pred = centroid_pred[i] * seg_pred[i].float()
            pred_count = _count_peaks(
                masked_pred, threshold=self.threshold, nms_kernel=self.nms_kernel
            )

            if gt_count == 0 and pred_count == 0:
                ape = 0.0
            elif gt_count == 0:
                ape = 1.0  # 100% error for false positives on empty image
            else:
                ape = abs(pred_count - gt_count) / gt_count

            self.total_ape += ape
            self.n += 1

    @property
    def value(self):
        return self.total_ape / self.n if self.n > 0 else 0.0


def _count_peaks(heatmap: torch.Tensor, threshold: float = 0.3, nms_kernel: int = 41) -> int:
    """Count local maxima in a 2D heatmap above a threshold.

    Uses max-pooling NMS: a pixel is a peak only if it equals the local
    maximum within the nms_kernel window AND exceeds threshold.
    nms_kernel should be scaled to the Gaussian sigma (e.g. 2*sigma+1).
    """
    # Ensure odd kernel for symmetric padding
    nms_kernel = nms_kernel if nms_kernel % 2 == 1 else nms_kernel + 1
    pad = nms_kernel // 2
    h = heatmap.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    local_max = F.max_pool2d(h, kernel_size=nms_kernel, stride=1, padding=pad)
    is_peak = (h == local_max) & (h > threshold)
    return int(is_peak.sum().item())
