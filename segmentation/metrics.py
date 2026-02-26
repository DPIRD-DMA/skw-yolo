"""Segmentation metrics.

Metrics use learn.yb[0] (the mask tensor) instead of learn.y, because
yb may contain extra items like pixel weight maps.
"""

import torch
from fastai.metrics import DiceMulti, JaccardCoeffMulti
from fastai.torch_core import flatten_check


class SegDiceMulti(DiceMulti):
    """DiceMulti that reads only yb[0] (ignores extra yb items like weight maps)."""

    def accumulate(self, learn):
        pred, targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.yb[0])
        for c in range(learn.pred.shape[self.axis]):
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
    """Foreground-only IoU — matches RasterIoU from the detection pipeline."""

    @property
    def name(self):
        return "fg_iou"

    def accumulate(self, learn):
        pred, targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.yb[0])
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
