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
# smp.Unet (plain, no attention — SCSE attention tested but hurt fg_iou)
# 3-layer dilated centroid head: d=8,16,32 (~115px RF)
# BatchNorm + ReLU between layers
# CenterNet bias init (-4.0) on final conv

# Training
canvas_size = 792
batch_size = 6
gradient_accumulation_batch_size = 24
learning_rate = 3e-4  # with discriminative slice(lr/5, lr)
epoch_count = 25  # peaks at ~12-13 epochs, SaveModelCallback saves best
ema_decay = 0.995  # slower EMA for stable validation

# Loss weights
dice_weight = 15.0
ce_weight = 1.0
lovasz_weight = 5.0  # Lovász-Softmax loss directly optimizes IoU
focal_gamma = 1.0  # Focal CE: downweight easy pixels, focus on hard boundaries
centroid_weight = 1.5
centroid_pos_weight = 10.0
clip_distance = 10
class_ramps = {1: (3.0, "edge")}  # boundary pixel upweighting

# Centroid config
centroid_sigma = 12.0
centroid_threshold = 0.3
nms_kernel = 61

# Post-processing
smooth_sigma = 2.0  # Gaussian blur before peak detection (in _count_peaks)

# Augmentations
# BatchResample(min_scale=0.5, max_scale=1.1) — critical for generalization
# BatchFlip, BatchRot90, BatchRotate(15°, p=0.3)
# DynamicZScoreNormalize, RandomRectangle, RandomSharpenBlur
# RandomClipLargeImages(400-600) — after resample
```

**Best results (balanced):** fg_iou=0.651, cnt_mae=1.40, cnt_mape=0.31 (focal_gamma=1.0)
**Best results (counting):** fg_iou=0.646, cnt_mae=0.87, cnt_mape=0.26 (focal_gamma=2.0)

## Key Takeaways

1. **Lovász-Softmax loss** was the biggest improvement for fg_iou (+0.015 over Dice-only), directly optimizing IoU
2. **Focal cross-entropy** (gamma=1.0) improves counting without hurting fg_iou; higher gamma (2.0) dramatically improves counting (cnt_mape: 0.34→0.26) but reduces fg_iou by ~0.005
3. **Gaussian smoothing in peak detection** was the single biggest early improvement for counting (cnt_mape: 1.86 → 0.40)
4. **Training at native 792px resolution** significantly outperforms 600px with cropping (fg_iou: 0.636 → 0.651)
5. **3-layer dilated centroid head** (d=8,16,32) with BatchNorm gives better segmentation than 2-layer
6. **Loss weight balance is critical** — centroid and segmentation losses compete for shared encoder capacity
7. **centroid_weight=1.5** is the sweet spot; 2.0 steals too much from segmentation (fg_iou drops 0.003)
8. **EMA decay=0.995** (slower) outperforms 0.99 (faster), giving more stable validation metrics
9. **ConvNeXtV2 Tiny** significantly outperforms Nano for this task
10. **Discriminative LR** (lower for encoder, higher for heads) helps both tasks
11. **NMS kernel must match object scale** (~half median diameter), not sigma
12. **MAPE is more robust than MAE** for counting evaluation (MAE can be gamed by predicting nothing)
13. **BatchResample augmentation** is important for spatial resolution generalization even at full resolution

### What Helped

| Change | fg_iou Impact | cnt_mape Impact | Notes |
|--------|---------------|-----------------|-------|
| Lovász loss (weight=5) | +0.015 | — | Direct IoU optimization |
| 792px training resolution | +0.010 | +0.05 | Match val resolution |
| Focal CE (gamma=1.0) | +0.000 | +0.03 | Focus on hard pixels |
| Focal CE (gamma=2.0) | -0.005 | +0.08 | Best counting, slight fg_iou cost |
| centroid_weight=1.5 | — | +0.11 | Better counting focus |
| EMA decay 0.995 | +0.005 | — | More stable validation |
| dice_weight=15 (with Lovász) | +0.005 | — | Let Lovász handle IoU |

### What Did Not Help

| Change | fg_iou Impact | cnt_mape Impact | Notes |
|--------|---------------|-----------------|-------|
| SCSE decoder attention | -0.011 | — | Extra params don't help this task |
| Weight decay 0.01 | -0.005 | +0.01 | Regularization hurts peak performance |
| Stronger augmentations | -0.005 | +0.01 | Too aggressive for 434-image dataset |
| dice_weight=17 (with Lovász=5) | -0.005 | — | Competes with Lovász; 15 is better |
| centroid_weight=2.0 | -0.003 | +0.03 | Steals too much from segmentation |
| Focal gamma=1.5 | -0.004 | — | Worse than both 1.0 and 2.0 |
| ConvNeXtV2 Base encoder | OOM | — | Too large for 792px on RTX 4090 |
| UNet++ decoder | crash | — | Incompatible with 792px (not div by 16) |

### Focal Gamma Sweep

| Gamma | fg_iou | cnt_mape | cnt_mae | Notes |
|-------|--------|----------|---------|-------|
| 0.0 | 0.651 | 0.337 | 1.47 | No focal (iter 3) |
| 1.0 | **0.651** | 0.311 | 1.40 | Best balance |
| 1.5 | 0.647 | 0.327 | 1.42 | Worse at both |
| 2.0 | 0.646 | **0.258** | **0.87** | Best counting |

## Progression Summary

| Stage | fg_iou | cnt_mape | Key Change |
|-------|--------|----------|------------|
| Baseline (notebook) | 0.555 | N/A | Original config |
| + ConvNeXtV2 Tiny, disc LR | 0.633 | 1.20 | Bigger encoder, better LR |
| + 3-layer dilated head | 0.642 | 1.86 | Larger receptive field |
| + Weight rebalancing | 0.640 | 0.99 | dice=20, cw=2.0 |
| + Gaussian smoothing | 0.641 | 0.40 | Post-processing noise filter |
| + 792px + Lovász (w=3) | 0.642 | 0.33 | Full resolution + IoU loss |
| + Lovász w=5 + EMA 0.995 | 0.651 | 0.34 | Stronger IoU optimization |
| + Focal CE (gamma=1.0) | **0.651** | **0.31** | Focus on hard pixels |
| (alt) Focal CE (gamma=2.0) | 0.646 | **0.26** | Best counting variant |
