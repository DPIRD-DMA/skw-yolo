"""Detection-specific augmentations (bbox-aware geometric transforms)."""

import random

import torch
from fastai.callback.core import Callback
from fastai.torch_core import TensorImage


class MosaicCallback(Callback):
    """YOLO-style mosaic augmentation that combines 4 images into one.

    Uses images already in the batch (no disk I/O). Each mosaic image keeps
    itself in one quadrant and draws the other 3 from random batch peers.
    A single random center per batch enables fully vectorized composition.
    """

    order = 58  # before image augs (62) and geometric augs (65)

    def __init__(
        self,
        p: float = 1.0,
        min_center: float = 0.25,
        max_center: float = 0.75,
    ):
        self.p = p
        self.min_center = min_center
        self.max_center = max_center

    def before_batch(self):
        if not self.training or random.random() > self.p:
            return

        images = self.learn.xb[0]  # [B, C, H, W]
        targets = self.learn.yb[0]
        B, C, H, W = images.shape
        device = images.device

        # Random center (shared across batch for vectorized slicing)
        cx = random.randint(int(W * self.min_center), int(W * self.max_center))
        cy = random.randint(int(H * self.min_center), int(H * self.max_center))

        # 3 random permutations to pair batch images
        perm1 = torch.randperm(B, device=device)
        perm2 = torch.randperm(B, device=device)
        perm3 = torch.randperm(B, device=device)

        # Vectorized mosaic composition — 4 slice ops, no Python loop over B
        mosaic = torch.empty_like(images)
        mosaic[:, :, :cy, :cx] = images[:, :, H - cy :, W - cx :]
        mosaic[:, :, :cy, cx:] = images[perm1, :, H - cy :, : W - cx]
        mosaic[:, :, cy:, :cx] = images[perm2, :, : H - cy, W - cx :]
        mosaic[:, :, cy:, cx:] = images[perm3, :, : H - cy, : W - cx]

        # Bbox adjustment — vectorized per quadrant (4 iters over all boxes)
        batch_idx = targets["batch_idx"]
        cls = targets["cls"]
        bboxes = targets["bboxes"]

        offsets = [(cx - W, cy - H), (cx, cy - H), (cx - W, cy), (cx, cy)]
        perms = [torch.arange(B, device=device), perm1, perm2, perm3]

        new_all_bboxes = []
        new_all_cls = []
        new_all_batch_idx = []

        for q in range(4):
            if bboxes.shape[0] == 0:
                continue
            ox, oy = offsets[q]
            perm = perms[q]

            # Inverse perm: original image k -> mosaic slot inv[k]
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(B, device=device)

            bb = bboxes
            x1 = (bb[:, 0] - bb[:, 2] / 2) * W + ox
            y1 = (bb[:, 1] - bb[:, 3] / 2) * H + oy
            x2 = (bb[:, 0] + bb[:, 2] / 2) * W + ox
            y2 = (bb[:, 1] + bb[:, 3] / 2) * H + oy

            x1 = x1.clamp(0, W)
            y1 = y1.clamp(0, H)
            x2 = x2.clamp(0, W)
            y2 = y2.clamp(0, H)

            bw = x2 - x1
            bh = y2 - y1
            valid = (bw > 2) & (bh > 2)

            if valid.any():
                x1, y1 = x1[valid], y1[valid]
                x2, y2 = x2[valid], y2[valid]
                bw, bh = x2 - x1, y2 - y1

                new_bb = torch.stack(
                    [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, bw / W, bh / H],
                    dim=1,
                )
                new_idx = inv_perm[batch_idx[valid].long()].float()

                new_all_bboxes.append(new_bb)
                new_all_cls.append(cls[valid])
                new_all_batch_idx.append(new_idx)

        self.learn.xb = (TensorImage(mosaic),)

        if new_all_bboxes:
            new_targets = {
                "batch_idx": torch.cat(new_all_batch_idx, 0),
                "cls": torch.cat(new_all_cls, 0),
                "bboxes": torch.cat(new_all_bboxes, 0),
            }
        else:
            new_targets = {
                "batch_idx": torch.zeros(0, device=device),
                "cls": torch.zeros(0, 1, device=device),
                "bboxes": torch.zeros(0, 4, device=device),
            }

        self.learn.yb = (new_targets,)


class GeometricBBoxAugCallback(Callback):
    """Applies geometric augmentations to both images and bounding boxes.

    Handles:
    - Random horizontal/vertical flips
    - Random 90-degree rotations

    Applied as a fastai Callback to access both xb (images) and yb (targets).
    """

    order = 65  # after image-only augs

    def __init__(self, flip_p: float = 0.5, rot90_p: float = 0.5):
        self.flip_p = flip_p
        self.rot90_p = rot90_p

    def before_batch(self):
        if not self.training:
            return

        images = self.learn.xb[0]  # [B, 3, H, W]
        targets = self.learn.yb[0]  # dict{batch_idx, cls, bboxes}

        bboxes = targets["bboxes"]  # [N, 4] xywh normalized

        # Random horizontal flip (entire batch)
        if random.random() < self.flip_p:
            images = torch.flip(images, dims=[-1])
            if bboxes.shape[0] > 0:
                bboxes = bboxes.clone()
                bboxes[:, 0] = 1.0 - bboxes[:, 0]  # flip x_center

        # Random vertical flip (entire batch)
        if random.random() < self.flip_p:
            images = torch.flip(images, dims=[-2])
            if bboxes.shape[0] > 0:
                bboxes = bboxes.clone()
                bboxes[:, 1] = 1.0 - bboxes[:, 1]  # flip y_center

        # Random 90-degree rotation
        if random.random() < self.rot90_p:
            k = random.choice([1, 2, 3])
            images = torch.rot90(images, k, dims=[-2, -1])
            if bboxes.shape[0] > 0:
                bboxes = bboxes.clone()
                for _ in range(k):
                    # 90-degree CCW: (x,y,w,h) -> (y, 1-x, h, w)
                    old_x = bboxes[:, 0].clone()
                    old_y = bboxes[:, 1].clone()
                    old_w = bboxes[:, 2].clone()
                    old_h = bboxes[:, 3].clone()
                    bboxes[:, 0] = old_y
                    bboxes[:, 1] = 1.0 - old_x
                    bboxes[:, 2] = old_h
                    bboxes[:, 3] = old_w

        targets = {**targets, "bboxes": bboxes}
        self.learn.xb = (TensorImage(images),)
        self.learn.yb = (targets,)
