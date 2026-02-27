"""Mask-aware batch augmentations for segmentation (fastai RandTransform style)."""

import random

import torch
from fastai.torch_core import TensorImage, TensorMask
from fastai.vision.augment import RandTransform


class BatchFlip(RandTransform):
    """Randomly flip images and masks horizontally and/or vertically.

    All items in the batch receive the same flip to maintain spatial
    correspondence between images and masks.
    """

    split_idx = 0
    order = 5

    def __init__(self, p: float = 1.0, flip_vert: bool = True, flip_horiz: bool = True):
        super().__init__(p=p)
        self.flip_vert = flip_vert
        self.flip_horiz = flip_horiz
        self.p = p

    def before_call(self, b, split_idx):
        if random.random() < self.p:
            self.do = True
            self.do_horiz = self.flip_horiz and random.choice([True, False])
            self.do_vert = self.flip_vert and random.choice([True, False])
        else:
            self.do = False

    def encodes(self, x: TensorImage | TensorMask) -> TensorImage | TensorMask:
        if not self.do:
            return x
        if self.do_horiz:
            x = type(x)(torch.flip(x, dims=[-1]))
        if self.do_vert:
            x = type(x)(torch.flip(x, dims=[-2]))
        return x


class BatchRot90(RandTransform):
    """Randomly rotate batches by 0, 90, 180, or 270 degrees.

    All items in the batch receive the same rotation to maintain spatial
    correspondence between images and masks.
    """

    split_idx = 0
    order = 6

    def __init__(self, p: float = 1.0):
        super().__init__(p=p)
        self.p = p

    def before_call(self, b, split_idx):
        if random.random() < self.p:
            self.rot = random.choice([0, 1, 2, 3])
        else:
            self.rot = 0

    def encodes(self, x: TensorImage | TensorMask) -> TensorImage | TensorMask:
        return type(x)(x.rot90(self.rot, [-2, -1]))
