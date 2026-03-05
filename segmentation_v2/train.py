"""SKW Segmentation v2 — training script for iterative improvement."""

import sys
from pathlib import Path

import torch
from fastai.callback.schedule import fit_one_cycle  # noqa: F401 (patches Learner)
from fastai.callback.tracker import SaveModelCallback
from fastai.learner import Learner
from fastai.vision.all import ShowGraphCallback

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import build_dataloaders
from model import UnetWithCentroids
from loss import DiceCELoss, EMACallback
from metrics import ForegroundIoU, SegDiceMulti, CentroidCountMAE, CentroidCountMAPE
from augs import BatchFlip, BatchRot90, BatchRotate, RandomClipLargeImages, BatchResample
from validate import print_dataset_summary, validate_dataset
from shared.helpers import print_system_info
from shared.augs import RandomRectangle, RandomSharpenBlur, DynamicZScoreNormalize

# ── Configuration ────────────────────────────────────────────────────
data_dir = Path("/media/nick/4TB Working 7/Datasets/SKW")
data_v2_dir = Path("/media/nick/4TB Working 7/Datasets/SKW/data_v2")
model_type = "convnextv2_tiny.fcmae_ft_in22k_in1k"
model_version = "SKW_SEG_v2"

use_bf16 = True
use_compile = False  # dynamic shapes + smp model cause compile issues
canvas_size = 600  # training output tensor size
val_canvas_size = 792  # validation output tensor size (native resolution)
ignore_index = 99

train_bs = 24
val_bs = 8
learning_rate = 3e-4
epoch_count = 10
num_workers = 8

# Positive oversampling to rebalance against empty tiles
positive_oversample = 5

# Loss weights
dice_weight = 17.0
ce_weight = 1.0
clip_distance = 10  # reduced from 30 (fewer erosion iterations)
class_ramps = {
    1: (3.0, "edge"),   # boundary pixels weighted 3x — helps counting
    # bg ramp removed — was not helping metrics
}

# Centroid config
centroid_sigma = 12.0
centroid_weight = 1.0
centroid_pos_weight = 10.0
centroid_threshold = 0.3
nms_kernel = 61  # ~half median object diameter


def train():
    print_system_info()
    print(f"\nConfig: lr={learning_rate}, epochs={epoch_count}")
    print(f"  canvas: train={canvas_size}, val={val_canvas_size}")
    print(f"  dice_weight={dice_weight}, ce_weight={ce_weight}")
    print(f"  centroid_weight={centroid_weight}, pos_weight={centroid_pos_weight}")
    print(f"  encoder={model_type}")

    _ = print_dataset_summary(data_dir, data_v2_dir)
    exclude_stems = validate_dataset(data_dir, data_v2_dir, canvas_size)

    batch_tfms = [
        BatchFlip(),
        BatchRot90(),
        BatchRotate(max_angle=15.0, ignore_index=ignore_index, p=0.3),
        DynamicZScoreNormalize(target_mean=0.0, target_std=1.0),
        RandomRectangle(p=0.2, sl=0.02, sh=0.1, max_count=3),
        RandomSharpenBlur(min_factor=0.5, max_factor=1.5),
        RandomClipLargeImages(min_size=300, max_size=400),  # multi-scale
        BatchResample(min_scale=0.45, max_scale=1.1, plateau_min=0.5, plateau_max=1.0),
    ]

    dls = build_dataloaders(
        data_dir=data_dir,
        data_v2_dir=None,  # data_v2_dir,
        positive_oversample=positive_oversample,
        exclude_stems=exclude_stems,
        canvas_size=canvas_size,
        val_canvas_size=val_canvas_size,
        ignore_index=ignore_index,
        train_bs=train_bs,
        val_bs=val_bs,
        batch_tfms=batch_tfms,
        shape="ellipse",
        centroid_sigma=centroid_sigma,
        num_workers=num_workers,
    )

    model = UnetWithCentroids(
        encoder_name=f"tu-{model_type.split('.')[0]}",
        encoder_weights="imagenet",
        in_channels=3,
    )

    if use_compile:
        model = torch.compile(model, dynamic=True)

    callbacks = [
        ShowGraphCallback(),
        EMACallback(decay=0.99),
        SaveModelCallback(monitor="fg_iou", comp=lambda a, b: a > b),
    ]

    learner = Learner(
        dls=dls,
        model=model,
        loss_func=DiceCELoss(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            centroid_weight=centroid_weight,
            centroid_pos_weight=centroid_pos_weight,
            clip_distance=clip_distance,
            class_ramps=class_ramps,
            ignore_index=ignore_index,
        ),
        metrics=[
            SegDiceMulti(ignore_index=ignore_index),
            ForegroundIoU(ignore_index=ignore_index),
            CentroidCountMAE(threshold=centroid_threshold, nms_kernel=nms_kernel),
            CentroidCountMAPE(threshold=centroid_threshold, nms_kernel=nms_kernel),
        ],
        cbs=callbacks,
    )

    if use_bf16:
        learner = learner.to_bf16()

    learner.fit_one_cycle(epoch_count, lr_max=slice(learning_rate / 5, learning_rate))

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    train()
