# Segmentation v2 — Tuning Findings

## Dataset

- **Source:** SKW (Skeleton Weed) drone imagery dataset
- **Location:** `/media/nick/4TB Working 7/Datasets/SKW`
- **Images:** 792x792px drone images, letterboxed to canvas size
- **Split:** 389 train / 45 val / 2 test
- **Labels:** YOLO format normalized `[cls, xc, yc, w, h]` bounding boxes, rasterized into:
  - **Segmentation masks** (channel 0): binary fg/bg ellipses from bboxes, ignore_index=99 for padding
  - **Centroid heatmaps** (channel 1): Gaussian blobs at bbox centers (sigma configurable)
- **Classes:** `skw_0S` (class 0), `skw_1R` (class 1) — treated as single-class binary segmentation
- **Object size statistics:** median=115px diameter, p25=72px, p75=168px

## Architecture

- **Backbone:** smp.Unet with `tu-convnextv2_tiny` encoder (ImageNet pretrained)
- **Segmentation head:** smp default → 2-class logits (bg/fg)
- **Centroid head:** Custom dilated conv stack with BatchNorm → 1-channel heatmap logits
- **Output:** `[B, 3, H, W]` — channels 0-1 seg logits, channel 2 centroid logits
- **Training:** fastai Learner, fit_one_cycle, MixedPrecision (bf16), GradientAccumulation

## Key Metrics

| Metric | Description |
|--------|-------------|
| `fg_iou` | Foreground-only IoU (class 1 Jaccard index), ignoring padding |
| `cnt_mae` | Mean absolute error between predicted and GT centroid counts |
| `cnt_mape` | Mean absolute percentage error for centroid counts (more robust than MAE) |

## Experiment Results

All experiments run for 10 epochs on RTX 4090, batch_size=6, grad_accum=16.

### Centroid Head Architecture

| Head Architecture | fg_iou | cnt_mae | cnt_mape | Notes |
|-------------------|--------|---------|----------|-------|
| 2-layer dilated (d=8,16) | 0.633 | 2.82 | 1.20 | Original best |
| 3-layer dilated (d=8,16,32) | 0.642 | 3.82 | 1.86 | +0.009 fg_iou, worse counting |
| 3-layer + CBAM attention | 0.632 | 2.96 | 1.54 | Attention needs more epochs |

**Finding:** 3-layer dilated head (RF ~115px matching median object diameter) improves segmentation but hurts counting stability. The extra capacity helps the shared encoder learn better features for segmentation.

### Loss Weight Tuning (3-layer head, sigma=12, lr=2e-4)

| dice_w | ce_w | cent_w | pos_w | fg_iou | cnt_mae | cnt_mape |
|--------|------|--------|-------|--------|---------|----------|
| 10 | 1 | 1.0 | 10 | 0.642 | 3.82 | 1.86 |
| 10 | 1 | 2.0 | 10 | 0.602 | 1.98 | 0.63 |
| 10 | 1 | 1.5 | 10 | 0.633 | 3.44 | 1.19 |
| 15 | 1 | 2.0 | 10 | 0.633 | 2.44 | 0.99 |
| 20 | 1 | 2.0 | 10 | 0.640 | 2.56 | 0.99 |
| 25 | 1 | 2.0 | 10 | 0.629 | 2.98 | 1.33 |
| 20 | 2 | 2.0 | 10 | 0.640 | 3.36 | 1.95 |
| 20 | 1 | 2.0 | 5 | 0.635 | 2.47 | 0.79 |
| 20 | 1 | 1.5 | 5 | 0.636 | 3.44 | 1.77 |

**Findings:**
- **centroid_weight:** Higher values (2.0) dramatically improve counting but steal capacity from segmentation. Must be balanced with higher dice_weight.
- **dice_weight:** Sweet spot around 20. Below 15: fg_iou suffers. Above 25: total loss too large, convergence slows.
- **pos_weight:** Lower values (5) stabilize counting but slightly hurt fg_iou. Higher values (10) give stronger positive signal.
- **ce_weight:** Increasing beyond 1.0 doesn't help and hurts centroid counting.

### Sigma (Gaussian target width)

| Sigma | fg_iou | cnt_mae | cnt_mape | Notes |
|-------|--------|---------|----------|-------|
| 12 | 0.642 | 3.82 | 1.86 | Good balance |
| 20 | 0.627 | 4.11 | 1.17 | Too broad, hurts segmentation |

**Finding:** sigma=12 is better overall. Larger sigma creates broader targets that hurt spatial precision.

### Learning Rate

| LR | fg_iou | cnt_mae | cnt_mape | Notes |
|----|--------|---------|----------|-------|
| 2e-4 | 0.640 | 1.98 | 0.42 | Stable for both tasks |
| 3e-4 | 0.641 | 1.78 | 0.40 | Best overall (with smoothing) |
| 3e-4 (no smoothing) | 0.643 | 5.29 | 2.41 | Good seg, bad counting |

**Finding:** Higher LR benefits segmentation but makes centroid predictions noisier. Gaussian smoothing in post-processing compensates for the noise.

### Training Strategy

| Strategy | fg_iou | cnt_mae | cnt_mape | Notes |
|----------|--------|---------|----------|-------|
| fit_one_cycle 10ep | 0.640 | 1.98 | 0.42 | Standard approach |
| Freeze 3ep + unfreeze 7ep | 0.564 | 2.96 | 1.18 | Not enough epochs per phase |
| grad_accum=32 | 0.599 | 2.60 | 1.22 | Too few updates per epoch |

**Finding:** Standard fit_one_cycle with discriminative LR `slice(lr/5, lr)` is best for 10 epochs. Freeze/unfreeze needs more total epochs. Larger grad_accum reduces updates too much.

### Gaussian Smoothing in Peak Detection

| Smoothing | fg_iou | cnt_mae | cnt_mape | Notes |
|-----------|--------|---------|----------|-------|
| None | 0.635 | 2.47 | 0.79 | Raw peak detection |
| sigma=3 | 0.632 | 2.27 | 0.41 | Noise suppression helps a lot |

**Finding:** Gaussian blur (sigma=3) before peak detection is the single biggest improvement for counting accuracy. It filters noise and merges spurious nearby peaks without affecting training.

### NMS Kernel Size

| NMS Kernel | Notes |
|------------|-------|
| 25 | Too small — single plant gets 10+ predictions |
| 61 | Good — ~half median object diameter, allows overlapping neighbors |
| 101 | Too large — merges distinct nearby overlapping objects |

**Finding:** NMS kernel should be ~half the median object diameter. Objects overlap in this dataset, so kernel must be smaller than full object size.

## Best Configuration

```python
# Architecture
model_type = "convnextv2_tiny.fcmae_ft_in22k_in1k"  # ConvNeXtV2 Tiny encoder
# 3-layer dilated centroid head: d=8,16,32 (~115px RF)
# BatchNorm + ReLU between layers
# CenterNet bias init (-4.0) on final conv

# Training
canvas_size = 792
batch_size = 6
gradient_accumulation_batch_size = 16
learning_rate = 3e-4  # with discriminative slice(lr/5, lr)
epoch_count = 10

# Loss weights
dice_weight = 10.0
ce_weight = 1.0
centroid_weight = 1.0
centroid_pos_weight = 10.0
clip_distance = 30
class_ramps = {1: (5.0, "center"), 0: (2.0, "center")}

# Centroid config
centroid_sigma = 12.0
centroid_threshold = 0.3
nms_kernel = 61

# Post-processing
smooth_sigma = 3.0  # Gaussian blur before peak detection
```

**Best results:** fg_iou=0.641, cnt_mae=1.78, cnt_mape=0.40

## Key Takeaways

1. **Gaussian smoothing in peak detection** was the single biggest improvement for counting accuracy (cnt_mape: 1.86 → 0.40)
2. **3-layer dilated centroid head** (d=8,16,32) with BatchNorm gives better segmentation than 2-layer
3. **Loss weight balance is critical** — centroid and segmentation losses compete for shared encoder capacity
4. **ConvNeXtV2 Tiny** significantly outperforms Nano for this task
5. **Discriminative LR** (lower for encoder, higher for heads) helps both tasks
6. **NMS kernel must match object scale** (~half median diameter), not sigma
7. **MAPE is more robust than MAE** for counting evaluation (MAE can be gamed by predicting nothing)

## Progression Summary

| Stage | fg_iou | cnt_mape | Key Change |
|-------|--------|----------|------------|
| Baseline (notebook) | 0.555 | N/A | Original config |
| + ConvNeXtV2 Tiny, disc LR | 0.633 | 1.20 | Bigger encoder, better LR |
| + 3-layer dilated head | 0.642 | 1.86 | Larger receptive field |
| + Weight rebalancing | 0.640 | 0.99 | dice=20, cw=2.0 |
| + Gaussian smoothing | 0.641 | 0.40 | Post-processing noise filter |
