"""SKW Segmentation v2 — training script for iterative improvement."""

import sys
from pathlib import Path

import torch
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.schedule import fit_one_cycle  # noqa: F401 (patches Learner)
from fastai.callback.tracker import SaveModelCallback
from fastai.learner import Learner
from fastai.callback.training import GradientAccumulation

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import build_dataloaders
from model import UnetWithCentroids
from loss import DiceCELoss
from metrics import ForegroundIoU, SegDiceMulti, CentroidCountMAE, CentroidCountMAPE
from augs import BatchFlip, BatchRot90
from shared.helpers import print_system_info
from shared.augs import RandomRectangle, RandomSharpenBlur, DynamicZScoreNormalize

# ── Configuration ────────────────────────────────────────────────────
data_dir = Path("/media/nick/4TB Working 7/Datasets/SKW")
model_type = "convnextv2_tiny.fcmae_ft_in22k_in1k"
model_version = "SKW_SEG_v2"

canvas_size = 792
ignore_index = 99
batch_size = 6
gradient_accumulation_batch_size = 16
learning_rate = 2e-4
epoch_count = 10

# Loss weights
dice_weight = 10.0
ce_weight = 1.0
clip_distance = 30
class_ramps = {
    1: (5.0, "center"),
    0: (2.0, "center"),
}

# Centroid config
centroid_sigma = 12.0
centroid_weight = 1.0
centroid_pos_weight = 10.0
centroid_threshold = 0.3
nms_kernel = 61  # ~half median object diameter; allows overlapping neighbors


def train():
    print_system_info()
    print(f"\nConfig: lr={learning_rate}, sigma={centroid_sigma}, nms_k={nms_kernel}")
    print(f"  centroid_weight={centroid_weight}, pos_weight={centroid_pos_weight}")
    print(f"  dice_weight={dice_weight}, ce_weight={ce_weight}")
    print(f"  clip_distance={clip_distance}")
    print(f"  encoder={model_type}")

    batch_tfms = [
        BatchFlip(),
        BatchRot90(),
        DynamicZScoreNormalize(target_mean=0.0, target_std=1.0),
        RandomRectangle(p=0.3, sl=0.5, sh=0.04, max_count=5),
        RandomSharpenBlur(min_factor=0.5, max_factor=1.5),
    ]

    dls = build_dataloaders(
        data_dir=data_dir,
        canvas_size=canvas_size,
        ignore_index=ignore_index,
        bs=batch_size,
        batch_tfms=batch_tfms,
        shape="ellipse",
        centroid_sigma=centroid_sigma,
    )

    model = UnetWithCentroids(
        encoder_name=f"tu-{model_type.split('.')[0]}",
        encoder_weights="imagenet",
        in_channels=3,
    )

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
        cbs=[
            MixedPrecision(amp_mode="bf16"),
            GradientAccumulation(gradient_accumulation_batch_size),
            SaveModelCallback(monitor="fg_iou", comp=lambda a, b: a > b),
        ],
    )

    learner.fit_one_cycle(epoch_count, lr_max=slice(learning_rate / 5, learning_rate))

    # Print final best
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    train()
