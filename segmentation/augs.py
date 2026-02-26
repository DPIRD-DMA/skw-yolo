"""Segmentation-specific augmentations (mask-aware geometric transforms)."""

import random

import torch
from fastai.callback.core import Callback
from fastai.torch_core import TensorImage


class MosaicSegCallback(Callback):
    """Mosaic augmentation for segmentation — combines 4 images and masks."""

    order = 58

    def __init__(
        self, p: float = 1.0, min_center: float = 0.25, max_center: float = 0.75
    ):
        self.p = p
        self.min_center = min_center
        self.max_center = max_center

    def before_batch(self):
        if not self.training or random.random() > self.p:
            return

        images = self.learn.xb[0]  # [B, C, H, W]
        masks = self.learn.yb[0]  # [B, H, W]
        B, C, H, W = images.shape
        device = images.device

        cx = random.randint(int(W * self.min_center), int(W * self.max_center))
        cy = random.randint(int(H * self.min_center), int(H * self.max_center))

        perm1 = torch.randperm(B, device=device)
        perm2 = torch.randperm(B, device=device)
        perm3 = torch.randperm(B, device=device)

        # Mosaic images
        mos_img = torch.empty_like(images)
        mos_img[:, :, :cy, :cx] = images[:, :, H - cy :, W - cx :]
        mos_img[:, :, :cy, cx:] = images[perm1, :, H - cy :, : W - cx]
        mos_img[:, :, cy:, :cx] = images[perm2, :, : H - cy, W - cx :]
        mos_img[:, :, cy:, cx:] = images[perm3, :, : H - cy, : W - cx]

        # Mosaic masks (same slicing, no channel dim)
        mos_mask = torch.empty_like(masks)
        mos_mask[:, :cy, :cx] = masks[:, H - cy :, W - cx :]
        mos_mask[:, :cy, cx:] = masks[perm1, H - cy :, : W - cx]
        mos_mask[:, cy:, :cx] = masks[perm2, : H - cy, W - cx :]
        mos_mask[:, cy:, cx:] = masks[perm3, : H - cy, : W - cx]

        self.learn.xb = (TensorImage(mos_img),)
        self.learn.yb = (mos_mask,)


class GeometricSegAugCallback(Callback):
    """Flip + rot90 applied to both images and segmentation masks."""

    order = 65

    def __init__(self, flip_p: float = 0.5, rot90_p: float = 0.5):
        self.flip_p = flip_p
        self.rot90_p = rot90_p

    def before_batch(self):
        if not self.training:
            return

        images = self.learn.xb[0]
        masks = self.learn.yb[0]

        if random.random() < self.flip_p:
            images = torch.flip(images, dims=[-1])
            masks = torch.flip(masks, dims=[-1])

        if random.random() < self.flip_p:
            images = torch.flip(images, dims=[-2])
            masks = torch.flip(masks, dims=[-2])

        if random.random() < self.rot90_p:
            k = random.choice([1, 2, 3])
            images = torch.rot90(images, k, dims=[-2, -1])
            masks = torch.rot90(masks, k, dims=[-2, -1])

        self.learn.xb = (TensorImage(images),)
        self.learn.yb = (masks,)
