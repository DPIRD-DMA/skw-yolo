"""SKW Binary Segmentation Training Script (v2).

U-Net segmentation on the SKW drone dataset using fastai.
Converted from train.ipynb for iterative experimentation.
"""

import sys
from pathlib import Path

import torch
from fastai.callback.schedule import fit_one_cycle
from fastai.callback.tracker import SaveModelCallback
from fastai.learner import Learner
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.training import GradientAccumulation

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import build_dataloaders
from model import UnetWithCentroids
from loss import DiceCELoss, EMACallback, center_distance_weights
from metrics import ForegroundIoU, SegDiceMulti, CentroidCountMAE, CentroidCountMAPE
from augs import (
    BatchFlip,
    BatchRot90,
    BatchRotate,
    RandomClipLargeImages,
    BatchResample,
)
from validate import print_dataset_summary, validate_dataset
from shared.helpers import print_system_info
from shared.augs import (
    RandomRectangle,
    RandomSharpenBlur,
    DynamicZScoreNormalize,
)


def train():
    print_system_info()

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir = Path("/media/nick/4TB Working 7/Datasets/SKW")
    positive_oversample = 5
    train_pct = 1.0
    val_pct = 1.0

    # ── Training ──────────────────────────────────────────────────────────
    model_type = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    canvas_size = 792
    val_canvas_size = 792
    train_bs = 6
    val_bs = 6
    gradient_accumulation_bs = 24
    learning_rate = 3e-4
    epoch_count = 25
    num_workers = 8
    use_bf16 = True
    ema_decay = 0.995

    # ── Loss ──────────────────────────────────────────────────────────────
    dice_weight = 15.0
    ce_weight = 1.0
    lovasz_weight = 5.0
    focal_gamma = 1.0
    clip_distance = 10
    class_ramps = {1: (3.0, "edge")}
    ignore_index = 99

    # ── Centroid ──────────────────────────────────────────────────────────
    centroid_sigma = 12.0
    centroid_weight = 1.5
    centroid_pos_weight = 10.0
    centroid_threshold = 0.3
    nms_kernel = 61

    # ── Verify ────────────────────────────────────────────────────────────
    assert data_dir.exists(), f"Dataset not found: {data_dir}"
    print(f"Canvas: train={canvas_size}, val={val_canvas_size}")
    print(f"Loss: dice={dice_weight}, ce={ce_weight}, lovasz={lovasz_weight}, centroid={centroid_weight}")
    print(f"Epochs: {epoch_count}")

    print_dataset_summary(data_dir)
    exclude_stems = validate_dataset(data_dir, canvas_size)

    batch_tfms = [
        BatchResample(min_scale=0.5, max_scale=1.1, plateau_min=0.6, plateau_max=1.0),
        BatchFlip(),
        BatchRot90(),
        BatchRotate(max_angle=15.0, ignore_index=ignore_index, p=0.3),
        DynamicZScoreNormalize(target_mean=0.0, target_std=1.0),
        RandomRectangle(p=0.2, sl=0.02, sh=0.1, max_count=3),
        RandomSharpenBlur(min_factor=0.5, max_factor=1.5),
        RandomClipLargeImages(min_size=400, max_size=600),
    ]

    dls = build_dataloaders(
        data_dir=data_dir,
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
        train_pct=train_pct,
        val_pct=val_pct,
    )

    model = UnetWithCentroids(
        encoder_name=f"tu-{model_type.split('.')[0]}",
        encoder_weights="imagenet",
        in_channels=3,
    )

    callbacks = [
        GradientAccumulation(gradient_accumulation_bs),
        EMACallback(decay=ema_decay),
        SaveModelCallback(monitor="fg_iou"),
    ]

    learner = Learner(
        dls=dls,
        model=model,
        loss_func=DiceCELoss(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            centroid_weight=centroid_weight,
            centroid_pos_weight=centroid_pos_weight,
            lovasz_weight=lovasz_weight,
            focal_gamma=focal_gamma,
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

    learner.fit_one_cycle(
        n_epoch=epoch_count,
        lr_max=slice(learning_rate / 5, learning_rate),
    )

    # Print final metrics
    vals = learner.recorder.values[-1]
    metric_names = ["train_loss", "valid_loss", "seg_dice_multi", "fg_iou", "cnt_mae", "cnt_mape"]
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    for name, val in zip(metric_names, vals):
        print(f"  {name}: {val:.4f}")
    print("=" * 60)

    # Save final model
    save_path = Path(__file__).parent / "models" / "model.pth"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(learner.model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()
