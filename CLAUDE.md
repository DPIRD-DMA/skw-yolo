# SKW_YOLO - Skeleton Weed Detection

YOLO26-based object detection and U-Net binary segmentation for the SKW drone dataset, using **fastai** as the training framework.

## Tooling

- **Python 3.13** managed with **uv** (`uv run`, `uv add`, `uv sync`)
- **ruff** for linting and formatting (`ruff check`, `ruff format`)
- Run scripts: `uv run python detection/train.py`, `uv run jupyter notebook`

## Project Structure

```
shared/                # Shared utilities used by both pipelines
├── data.py            # DATA_DIR, parse_splits, load_labels, letterbox, adjust_bboxes_for_letterbox
├── augs.py            # Image-only augs: normalize, rect erase, sharpen, resample, ImageAugmentationCallback
└── helpers.py         # print_system_info

detection/             # YOLO26 object detection pipeline
├── train.py           # Training entry point (callable from Optuna or standalone)
├── train.ipynb        # Interactive training notebook
├── model.py           # YOLO26ForFastai nn.Module wrapper
├── dataset.py         # SKWDetectionDataset, DataLoaders, collation, get_class_info
├── loss.py            # YOLO26Loss + callbacks (EMA, E2ELossDecay, GradAccum)
├── metrics.py         # YOLOmAP50, RasterIoU
├── augs.py            # Detection augs: MosaicCallback, GeometricBBoxAugCallback
├── helpers.py         # plot_detection_batch
└── optuna_search.py   # Hyperparameter optimization with Optuna

segmentation/          # U-Net binary segmentation pipeline
├── train_seg.ipynb    # Interactive training notebook
├── dataset.py         # SKWSegDataset, rasterize_boxes, build_seg_dataloaders
├── loss.py            # DiceCELoss (Dice + CE combined)
├── metrics.py         # ForegroundIoU (fg-only pixel IoU, comparable to RasterIoU)
└── augs.py            # Seg augs: MosaicSegCallback, GeometricSegAugCallback

yolo26*.pt             # Pretrained COCO weights (n/s/m/l/x)
models/                # Trained checkpoint outputs (both pipelines)
```

## Dataset

- **Location:** defined as `data_dir` in each notebook's config cell and `DATA_DIR` in `shared/data.py`
- **Classes:** `skw_0S` (class 0), `skw_1R` (class 1) — or single-class mode (`skw`)
- **Format:** YOLO normalized [cls, xc, yc, w, h] labels
- **Images:** 792x792 drone images, letterboxed to training size
- **Split:** 389 train / 45 val / 2 test

## Architecture

### Detection
- YOLO26 (ultralytics `DetectionModel`) wrapped for fastai `Learner` compatibility
- Pretrained COCO weights with partial weight matching (80→2 classes)
- E2E (end-to-end) loss: one2many + one2one branches with decay schedule
- EMA (exponential moving average) of model weights
- Gradient accumulation for effective larger batch sizes

### Segmentation
- smp.Unet with ConvNeXt encoder, pretrained ImageNet weights
- Binary segmentation: YOLO bbox labels rasterized into pixel masks
- DiceCELoss for class-imbalanced binary segmentation
- ForegroundIoU metric directly comparable to detection RasterIoU

## Key Conventions

- Bboxes stored as normalized xywh, converted to pixel xyxy for IoU computation
- Batch annotations flattened with `batch_idx` tensor for reassembly
- Letterbox padding uses 114/255 gray fill (ultralytics standard)
- Callback ordering: GradAccum(10) → Mosaic(58) → LossDecay(60) → ImageAugs(62) → EMA/GeomAugs(65) → Optuna(70)
- Training targets RTX 4090, CUDA, bf16 mixed precision
- Notebooks run from their own directory with `sys.path.insert(0, parent)` for shared imports

## Common Tasks

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check . --fix

# Run detection training
uv run python detection/train.py

# Run detection hyperparameter search
uv run python detection/optuna_search.py

# Add a dependency
uv add <package>
```
