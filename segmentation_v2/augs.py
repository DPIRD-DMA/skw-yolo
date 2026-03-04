"""Mask-aware batch augmentations for segmentation (fastai RandTransform style)."""

import random

import torch
import torchvision.transforms.functional as TF
from fastai.torch_core import TensorImage, TensorMask
from fastai.vision.augment import RandTransform
from typing import Tuple
import torch.nn.functional as F
import numpy as np


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


class BatchRotate(RandTransform):
    """Randomly rotate batches by a continuous angle.

    Fills introduced pixels with 0 for images and ignore_index for the
    seg mask channel (channel 0 of targets). The centroid heatmap channel
    (channel 1) is filled with 0.

    Args:
        max_angle: Maximum rotation in degrees (samples uniformly from [-max, +max]).
        ignore_index: Fill value for seg mask padding introduced by rotation.
        p: Probability of applying rotation (per batch).
    """

    split_idx = 0
    order = 7  # after BatchFlip (5) and BatchRot90 (6)

    def __init__(self, max_angle: float = 30.0, ignore_index: int = 99, p: float = 0.5):
        super().__init__(p=p)
        self.max_angle = max_angle
        self.ignore_index = ignore_index
        self.p = p

    def before_call(self, b, split_idx):
        self.angle = random.uniform(-self.max_angle, self.max_angle)
        self.do = random.random() < self.p

    def encodes(self, x: TensorImage | TensorMask) -> TensorImage | TensorMask:
        if not self.do or self.angle == 0:
            return x
        if isinstance(x, TensorMask):
            # Target is [B, 2, H, W]: channel 0 = seg mask, channel 1 = centroid heatmap
            seg = x[:, :1]
            cent = x[:, 1:]
            seg_rot = TF.rotate(seg, self.angle, fill=[float(self.ignore_index)])
            cent_rot = TF.rotate(cent, self.angle, fill=[0.0])
            return TensorMask(torch.cat([seg_rot, cent_rot], dim=1))
        return TensorImage(TF.rotate(x, self.angle, fill=[0.0]))


class RandomClipLargeImages(RandTransform):
    """
    Randomly crop batches of images and masks to a smaller size when they exceed the target dimensions.

    This transform applies random cropping to entire batches where images are larger than the desired
    output size. It first selects a random crop size between min_size and max_size, then chooses a
    random location within the image to extract the crop from. If the input images are smaller than
    the target crop size, no cropping is performed and the original batch is returned unchanged.

    The same crop size and coordinates are applied to all items in the batch to maintain spatial
    correspondence between images and masks. This transform is particularly useful after resampling
    operations that may produce varying image sizes, and helps speed up training by reducing the
    size of larger images.

    Cropping behaviour:

    ┌───────────────┐
    │  ┌──────────┐ │
    │  │ Random   │ │
    │  │ Crop     │ │
    │  │ Location │ │
    │  │          │ │
    │  └──────────┘ │
    └───────────────┘
    """

    split_idx = 0  # only apply to the training set
    order = 8  # should happen after normalisation and rotation

    def __init__(
        self,
        p: float = 1.0,
        min_size: int = 256,
        max_size: int = 256,
        ignore_index: int = 99,
        n_candidates: int = 10,
    ):
        super().__init__(p=p)

        self.min_size = min_size
        self.max_size = max_size
        self.ignore_index = ignore_index
        self.n_candidates = n_candidates
        self.do = True

    def before_call(self, b: Tuple[TensorImage, TensorMask], split_idx: int):
        image_size = b[0].shape[-1]

        new_size = random.randint(self.min_size, self.max_size)
        self.new_size = new_size

        if image_size < self.new_size:
            self.do = False
        else:
            self.do = True
            clip_max = image_size - self.new_size

            # Check for nodata regions introduced by rotation
            seg_mask = b[1][0, 0]  # first batch item, seg channel [H, W]
            has_nodata = (seg_mask == self.ignore_index).any().item()

            if not has_nodata:
                self.clip_x = random.randint(0, clip_max)
                self.clip_y = random.randint(0, clip_max)
                return

            # Sample multiple candidates, pick the one with least nodata
            best_x, best_y, best_nodata = 0, 0, float("inf")
            for _ in range(self.n_candidates):
                cx = random.randint(0, clip_max)
                cy = random.randint(0, clip_max)
                crop = seg_mask[cx : cx + new_size, cy : cy + new_size]
                nodata_count = (crop == self.ignore_index).sum().item()
                if nodata_count == 0:
                    best_x, best_y = cx, cy
                    break
                if nodata_count < best_nodata:
                    best_x, best_y, best_nodata = cx, cy, nodata_count

            self.clip_x = best_x
            self.clip_y = best_y

    def encodes(self, x: TensorImage | TensorMask) -> TensorImage | TensorMask:
        if not self.do:
            return x
        if isinstance(x, TensorImage):
            clipped_images = x[
                :,
                :,
                self.clip_x : self.clip_x + self.new_size,
                self.clip_y : self.clip_y + self.new_size,
            ]

            return TensorImage(clipped_images)
        if isinstance(x, TensorMask):
            clipped_masks = x[
                :,
                :,
                self.clip_x : self.clip_x + self.new_size,
                self.clip_y : self.clip_y + self.new_size,
            ]
            return TensorMask(clipped_masks)


class BatchResample(RandTransform):
    """
    Randomly resample images and masks to different scales using a plateau distribution.

    This transform applies random scaling to entire batches, where the scale factor is sampled
    from a three-zone plateau distribution that allows fine control over scale bias:
    - Linear fade-in from min_scale to plateau_min
    - Uniform sampling from plateau_min to plateau_max
    - Linear fade-out from plateau_max to max_scale

    The plateau distribution allows you to bias sampling towards specific scale ranges with
    linear probability transitions at the boundaries. When plateau bounds equal the min/max
    scales, the distribution becomes uniform (default behaviour).

    Images are resampled using random interpolation modes (bilinear/nearest) with randomly
    applied antialiasing, whilst masks use nearest neighbour to preserve discrete values.

    Probability Density

         │       ┌────────────┐
         │      ╱              ╲
         │    ╱  │            │  ╲
         │  ╱                      ╲
         │╱      │            │      ╲
         └────────────────────────────── Scale Factor
         │       │            │       │
    min_scale plat_min    plat_max  max_scale

    """

    order = 1  # the first thing we do as it reduces the batch size making all other transforms more efficient
    split_idx = 0  # only apply to the training set

    def __init__(
        self,
        p: float = 1.0,
        min_scale=0.5,
        max_scale=1.1,
        plateau_min=None,
        plateau_max=None,
    ):
        super().__init__(p=p)
        self.min_scale = min_scale
        self.max_scale = max_scale

        self._image_modes = [
            "bilinear",
            "nearest",
        ]
        self._antialias_modes = {"bilinear", "bicubic"}
        if plateau_min is None:
            plateau_min = min_scale
        if plateau_max is None:
            plateau_max = max_scale
        self.plateau_min = plateau_min
        self.plateau_max = plateau_max

    def _select_scale_factor(self) -> float:
        """Sample from plateau distribution: linear fade-in, uniform plateau, linear fade-out"""

        # Calculate ranges and areas
        lower_range = self.plateau_min - self.min_scale
        plateau_range = self.plateau_max - self.plateau_min
        upper_range = self.max_scale - self.plateau_max

        lower_area = lower_range / 2
        plateau_area = plateau_range
        upper_area = upper_range / 2

        total_area = lower_area + plateau_area + upper_area

        # Sample zone
        rand = random.random() * total_area

        if rand < lower_area:
            # Lower triangle
            u = random.random()
            return self.min_scale + lower_range * np.sqrt(u)
        elif rand < lower_area + plateau_area:
            # Plateau
            return random.uniform(self.plateau_min, self.plateau_max)
        else:
            # Upper triangle
            u = random.random()
            return self.max_scale - upper_range * np.sqrt(u)

    def before_call(self, batch: Tuple[TensorImage, TensorMask], split_idx: int):
        """Determine the target size before processing the batch"""
        original_size = batch[0].shape[-1]
        scale_factor = self._select_scale_factor()
        self.target_size = round(original_size * scale_factor)

    def _resample_image(self, image: TensorImage) -> TensorImage:
        """Resample image with random interpolation mode and antialiasing"""
        interpolation_mode = random.choice(self._image_modes)

        # Randomly apply antialiasing for modes that support it
        use_antialiasing = (
            interpolation_mode in self._antialias_modes and random.choice([True, False])
        )

        return TensorImage(
            F.interpolate(
                image,
                size=(self.target_size, self.target_size),
                mode=interpolation_mode,
                antialias=use_antialiasing,
            )
        )

    def _resample_mask(self, mask: TensorMask) -> TensorMask:
        """Resample mask using nearest neighbour to preserve discrete values"""
        # Add batch dimension, interpolate, then remove batch dimension
        if len(mask.shape) == 3:
            resampled = F.interpolate(
                mask.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode="nearest",
            )
            return TensorMask(resampled.squeeze(0))
        if len(mask.shape) == 4:
            resampled = F.interpolate(
                mask,
                size=(self.target_size, self.target_size),
                mode="nearest",
            )
            return TensorMask(resampled)
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    def encodes(self, x: TensorImage | TensorMask) -> TensorImage | TensorMask:
        """Apply appropriate resampling based on input type"""
        if isinstance(x, TensorImage):
            return self._resample_image(x)
        elif isinstance(x, TensorMask):
            return self._resample_mask(x)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")
