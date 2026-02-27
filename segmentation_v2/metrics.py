"""Segmentation metrics for SKW binary segmentation."""

import torch
from fastai.metrics import DiceMulti, JaccardCoeffMulti
from fastai.torch_core import flatten_check


class SegDiceMulti(DiceMulti):
    """Per-class Dice scores, ignoring padding pixels."""

    def __init__(self, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

    def accumulate(self, learn):
        pred, targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.yb[0])
        if self.ignore_index is not None:
            valid = targ != self.ignore_index
            pred = pred[valid]
            targ = targ[valid]
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
    """Foreground-only IoU (class 1), ignoring padding pixels."""

    def __init__(self, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

    @property
    def name(self):
        return "fg_iou"

    def accumulate(self, learn):
        pred, targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.yb[0])
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
