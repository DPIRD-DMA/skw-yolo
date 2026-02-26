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

    def _mosaic_2d(self, t, perm1, perm2, perm3, cy, cx, H, W):
        """Apply mosaic slicing to a [B, H, W] tensor."""
        out = torch.empty_like(t)
        out[:, :cy, :cx] = t[:, H - cy :, W - cx :]
        out[:, :cy, cx:] = t[perm1, H - cy :, : W - cx]
        out[:, cy:, :cx] = t[perm2, : H - cy, W - cx :]
        out[:, cy:, cx:] = t[perm3, : H - cy, : W - cx]
        return out

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

        # Mosaic masks and weight maps (same slicing, no channel dim)
        mos_mask = self._mosaic_2d(masks, perm1, perm2, perm3, cy, cx, H, W)

        yb = (mos_mask,)
        if len(self.learn.yb) > 1:
            weight_maps = self.learn.yb[1]
            mos_weights = self._mosaic_2d(
                weight_maps, perm1, perm2, perm3, cy, cx, H, W
            )
            yb = (mos_mask, mos_weights)

        self.learn.xb = (TensorImage(mos_img),)
        self.learn.yb = yb


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
        weight_maps = self.learn.yb[1] if len(self.learn.yb) > 1 else None

        if random.random() < self.flip_p:
            images = torch.flip(images, dims=[-1])
            masks = torch.flip(masks, dims=[-1])
            if weight_maps is not None:
                weight_maps = torch.flip(weight_maps, dims=[-1])

        if random.random() < self.flip_p:
            images = torch.flip(images, dims=[-2])
            masks = torch.flip(masks, dims=[-2])
            if weight_maps is not None:
                weight_maps = torch.flip(weight_maps, dims=[-2])

        if random.random() < self.rot90_p:
            k = random.choice([1, 2, 3])
            images = torch.rot90(images, k, dims=[-2, -1])
            masks = torch.rot90(masks, k, dims=[-2, -1])
            if weight_maps is not None:
                weight_maps = torch.rot90(weight_maps, k, dims=[-2, -1])

        self.learn.xb = (TensorImage(images),)
        yb = (masks,) if weight_maps is None else (masks, weight_maps)
        self.learn.yb = yb


