"""Shared image-only augmentations (no bbox/mask changes needed)."""

import random

import torch
import torch.nn.functional as F
from fastai.callback.core import Callback
from fastai.torch_core import TensorImage
from fastai.vision.augment import DisplayedTransform, RandTransform
from torchvision.transforms.functional import adjust_sharpness


class DynamicZScoreNormalize(DisplayedTransform):
    """Dynamically normalize images using Z-score normalization on non-zero pixels.

    Per-channel, per-image normalization then rescaled to target mean/std.
    Zero pixels (no-data) are preserved as zero after normalization.
    """

    order = 3

    def __init__(
        self,
        target_mean: float = 0.5,
        target_std: float = 0.75,
        no_data_value: float = 0.0,
    ):
        super().__init__(split_idx=None)
        self.target_mean = target_mean
        self.target_std = target_std
        self.no_data_value = no_data_value
        self.epsilon = 1e-8

    def encodes(self, x: TensorImage):
        mask = x != self.no_data_value
        valid_pixels = mask.sum(dim=(2, 3), keepdim=True)
        mean = (x * mask).sum(dim=(2, 3), keepdim=True) / valid_pixels.clamp(min=1)
        diff_sq = (x - mean) ** 2 * mask
        std = torch.sqrt(
            diff_sq.sum(dim=(2, 3), keepdim=True) / valid_pixels.clamp(min=1)
            + self.epsilon
        )
        normalized = (x - mean) / std  # z-score: mean=0, std=1
        normalized = normalized * self.target_std + self.target_mean  # rescale
        normalized = torch.where(mask, normalized, torch.zeros_like(x))
        return normalized


class DynamicMinMaxNormalize(DisplayedTransform):
    order = 3

    def __init__(
        self,
        no_data_value: float = 0.0,
    ):
        super().__init__(split_idx=None)
        self.no_data_value = no_data_value
        self.epsilon = 1e-8

    def encodes(self, x: TensorImage):
        mask = x != self.no_data_value
        masked = torch.where(mask, x, torch.full_like(x, float("inf")))
        ch_min = masked.amin(dim=(2, 3), keepdim=True)
        masked = torch.where(mask, x, torch.full_like(x, float("-inf")))
        ch_max = masked.amax(dim=(2, 3), keepdim=True)
        normalized = (x - ch_min) / (ch_max - ch_min + self.epsilon)
        normalized = torch.where(mask, normalized, torch.zeros_like(x))
        return normalized


class BatchResample(RandTransform):
    """Randomly resample images to different scales using a plateau distribution.

    Since YOLO bounding box coordinates are normalized (0-1), resizing the
    image does NOT require any bbox adjustment.
    """

    order = 1
    split_idx = 0

    def __init__(
        self,
        p: float = 1.0,
        min_scale: float = 0.5,
        max_scale: float = 1.2,
        plateau_min: float | None = None,
        plateau_max: float | None = None,
    ):
        super().__init__(p=p)
        self.min_scale = min_scale
        self.max_scale = max_scale
        if plateau_min is None:
            plateau_min = min_scale
        if plateau_max is None:
            plateau_max = max_scale
        self.plateau_min = plateau_min
        self.plateau_max = plateau_max

    def _select_scale_factor(self) -> float:
        lower_range = self.plateau_min - self.min_scale
        plateau_range = self.plateau_max - self.plateau_min
        upper_range = self.max_scale - self.plateau_max

        lower_area = lower_range / 2
        plateau_area = plateau_range
        upper_area = upper_range / 2
        total_area = lower_area + plateau_area + upper_area

        import numpy as np

        rand = random.random() * total_area
        if rand < lower_area:
            u = random.random()
            return self.min_scale + lower_range * np.sqrt(u)
        elif rand < lower_area + plateau_area:
            return random.uniform(self.plateau_min, self.plateau_max)
        else:
            u = random.random()
            return self.max_scale - upper_range * np.sqrt(u)

    def before_call(self, batch, split_idx):
        original_size = batch[0].shape[-1]
        scale_factor = self._select_scale_factor()
        # Round to nearest multiple of 32 (YOLO stride requirement)
        self.target_size = max(round(original_size * scale_factor / 32) * 32, 32)

    def encodes(self, x: TensorImage) -> TensorImage:
        interpolation_mode = random.choice(["bilinear", "nearest"])
        use_antialiasing = interpolation_mode == "bilinear" and random.choice(
            [True, False]
        )
        return TensorImage(
            F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode=interpolation_mode,
                antialias=use_antialiasing,
            )
        )


class RandomRectangle(RandTransform):
    """Randomly erase rectangular regions in images with random values.

    Only affects pixel values, no bbox adjustment needed.
    """

    import math

    order = 2
    split_idx = 0

    def __init__(
        self,
        p: float = 0.1,
        sl: float = 0.0,
        sh: float = 0.3,
        min_aspect: float = 0.3,
        max_count: int = 1,
        max_fill_value: float = 1.0,
    ):
        import math

        self.p = p
        self.sl = sl
        self.sh = sh
        self.min_aspect = min_aspect
        self.max_count = max_count
        self.max_fill_value = max_fill_value
        super().__init__(p=p)
        self.log_ratio = (math.log(min_aspect), math.log(1 / min_aspect))

    def _slice(self, area, sz: int) -> tuple[int, int]:
        import math

        bound = int(round(math.sqrt(area)))
        loc = random.randint(0, max(sz - bound, 0))
        return loc, loc + bound

    def _bounds(self, area, img_h, img_w):
        import math

        r_area = random.uniform(self.sl, self.sh) * area
        aspect = math.exp(random.uniform(*self.log_ratio))
        return self._slice(r_area * aspect, img_h) + self._slice(r_area / aspect, img_w)

    def encodes(self, x: TensorImage):
        count = random.randint(1, self.max_count)
        _, img_h, img_w = x.shape[-3:]
        area = img_h * img_w / count
        areas = [self._bounds(area, img_h, img_w) for _ in range(count)]

        for rl, rh, cl, ch in areas:
            chan = x.shape[-3]
            value_chan_count = random.randint(1, chan)
            value_chans = random.sample(range(chan), value_chan_count)
            fill_value = random.uniform(0, self.max_fill_value)
            for c in value_chans:
                x[:, c, rl:rh, cl:ch] = fill_value
        return x


class RandomSharpenBlur(RandTransform):
    """Randomly applies sharpening or blurring effects."""

    split_idx = 0
    order = 9

    def __init__(
        self,
        p: float = 1.0,
        min_factor: float = 0.0,
        max_factor: float = 2.0,
        per_sample_probability: float = 0.1,
    ):
        super().__init__(p=p)
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.per_sample_probability = per_sample_probability

    def encodes(self, x: TensorImage) -> TensorImage:
        x_out = x.clone()
        for idx, image in enumerate(x_out):
            if random.random() < self.per_sample_probability:
                sharpness_factor = random.uniform(self.min_factor, self.max_factor)
                image_min = image.min()
                image_max = image.max()
                image = (image - image_min) / (image_max - image_min + 1e-8)
                image = adjust_sharpness(image, sharpness_factor)
                image = image * (image_max - image_min) + image_min
                x_out[idx] = image
        return x_out


class ImageAugmentationCallback(Callback):
    """Applies image-only batch transforms that don't affect bounding boxes/masks.

    Uses before_batch hook so augmentations run before the model forward pass.
    Calls transform implementations directly (bypasses fastai type dispatch).
    """

    order = 62  # before geometric augs (order 65)

    def __init__(self, transforms: list):
        self.transforms = transforms
        self._split_idx = 0

    def before_batch(self):
        if not self.training:
            return

        images = TensorImage(self.learn.xb[0])

        for tfm in self.transforms:
            # Call before_call if it exists (for RandTransforms that need batch context)
            if hasattr(tfm, "before_call"):
                tfm.before_call((images, None), self._split_idx)
            # Check if the transform should run (RandTransform probability)
            if hasattr(tfm, "do") and not tfm.do:
                continue
            # Call the underlying implementation directly to avoid type dispatch issues
            images = TensorImage(tfm.encodes(TensorImage(images)))

        self.learn.xb = (images,)
