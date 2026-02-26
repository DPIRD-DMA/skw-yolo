# SKW_YOLO - Skeleton Weed Detection

Object detection (YOLO26) and binary segmentation (U-Net) pipelines for skeleton weed identification in drone imagery, built on **fastai**.

## Project Structure

```
shared/              Shared data loading, image augmentations, helpers
detection/           YOLO26 object detection pipeline
segmentation/        U-Net binary segmentation pipeline
models/              Trained checkpoint outputs
```

## Setup

```bash
uv sync
```

## Dataset

Set `data_dir` in each notebook's config cell to point to your local SKW dataset directory.

- 434 images (792x792), YOLO normalized bbox labels
- 2 classes: `skw_0S`, `skw_1R` (or single-class mode)
- Split: 389 train / 45 val / 2 test

## Detection

YOLO26 (ultralytics) wrapped for fastai with E2E loss, EMA, and mosaic augmentation.

```bash
# Interactive notebook
cd detection && uv run jupyter notebook train.ipynb

# Script training
uv run python detection/train.py

# Hyperparameter search
uv run python detection/optuna_search.py
```

## Segmentation

smp U-Net with ConvNeXt encoder. YOLO bbox labels are rasterized into binary masks for pixel-wise training. ForegroundIoU is directly comparable to detection's RasterIoU.

```bash
cd segmentation && uv run jupyter notebook train_seg.ipynb
```

## Development

```bash
uv run ruff format .
uv run ruff check . --fix
```
