"""Segmentation metrics."""

import torch
from fastai.metrics import JaccardCoeffMulti
from fastai.torch_core import flatten_check


class ForegroundIoU(JaccardCoeffMulti):
    """Foreground-only IoU — matches RasterIoU from the detection pipeline."""

    @property
    def name(self):
        return "fg_iou"

    def accumulate(self, learn):
        pred, targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.y)
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
