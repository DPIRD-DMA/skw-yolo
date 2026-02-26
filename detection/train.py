"""Training function for SKW YOLO26, callable from Optuna or standalone."""

import gc
import sys
from pathlib import Path

import torch
from fastai.callback.core import Callback, CancelFitException
from fastai.vision.all import Learner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.augs import (
    BatchResample,
    DynamicMinMaxNormalize,
    DynamicZScoreNormalize,
    ImageAugmentationCallback,
    RandomRectangle,
    RandomSharpenBlur,
)

from augs import GeometricBBoxAugCallback, MosaicCallback
from dataset import build_dataloaders, get_class_info
from loss import (
    DetectionGradientAccumulation,
    E2ELossDecayCallback,
    EMACallback,
    YOLO26Loss,
)
from metrics import YOLOmAP50
from model import YOLO26ForFastai

MODEL_BATCH_SIZES = {"n": 16, "s": 16, "m": 8, "l": 4, "x": 2}


class OptunaCallback(Callback):
    """Reports per-epoch mAP50 to Optuna and prunes underperforming trials."""

    order = 70

    def __init__(self, trial):
        self.trial = trial

    def after_epoch(self):
        if not self.recorder.values:
            return
        # recorder.values rows: [train_loss, valid_loss, metric0, ...]
        # mAP50 is the first (and only) metric at index 2
        epoch_metrics = self.recorder.values[-1]
        if len(epoch_metrics) >= 3:
            map50 = float(epoch_metrics[2])
            self.trial.report(map50, self.epoch)
            if self.trial.should_prune():
                raise CancelFitException()


def train(config: dict, trial=None) -> float:
    """Train a YOLO26 model with the given config.

    Returns val mAP50 as a float (0.0 on failure or pruning).
    """
    single_class = True
    use_bf16 = True
    num_workers = 8

    data_dir = Path(config["data_dir"])
    model_size = config["model_size"]
    img_size = config["img_size"]
    epoch_count = config["epoch_count"]
    lr_max = config["lr_max"]
    pos_weight = config["pos_weight"]
    ema_decay = config["ema_decay"]
    n_acc = config["n_acc"]
    mosaic_p = config["mosaic_p"]
    flip_p = config["flip_p"]
    rot90_p = config["rot90_p"]
    rect_p = config["rect_p"]
    rect_sl = config["rect_sl"]
    sharpen_min = config["sharpen_min_factor"]
    sharpen_max = config["sharpen_max_factor"]
    norm_type = config["norm_type"]
    use_resample = config["use_batch_resample"]

    batch_size = MODEL_BATCH_SIZES[model_size]
    _, num_classes = get_class_info(single_class)

    learn = None
    try:
        dls = build_dataloaders(
            data_dir=data_dir,
            img_size=img_size,
            bs=batch_size,
            num_workers=num_workers,
            single_class=single_class,
        )

        model = YOLO26ForFastai(
            model_size=model_size,
            nc=num_classes,
            pretrained=True,
            epochs=epoch_count,
        )

        loss_func = YOLO26Loss(model, pos_weight=pos_weight)

        # Image-only augmentations
        image_augs = []
        if use_resample:
            image_augs.append(
                BatchResample(
                    min_scale=0.5, max_scale=1.2, plateau_min=0.7, plateau_max=1.0
                )
            )
        image_augs.append(RandomRectangle(p=rect_p, sl=rect_sl, sh=0.4))
        image_augs.append(
            RandomSharpenBlur(min_factor=sharpen_min, max_factor=sharpen_max)
        )
        if norm_type == "minmax":
            image_augs.append(DynamicMinMaxNormalize())
        elif norm_type == "zscore":
            image_augs.append(DynamicZScoreNormalize())

        callbacks = [
            MosaicCallback(p=mosaic_p),
            ImageAugmentationCallback(image_augs),
            GeometricBBoxAugCallback(flip_p=flip_p, rot90_p=rot90_p),
            E2ELossDecayCallback(),
            EMACallback(decay=ema_decay, tau=2000),
            DetectionGradientAccumulation(n_acc=n_acc),
        ]
        if trial is not None:
            callbacks.append(OptunaCallback(trial))

        learn = Learner(
            dls=dls,
            model=model,
            loss_func=loss_func,
            metrics=[YOLOmAP50(nc=num_classes)],
            cbs=callbacks,
        )

        if use_bf16:
            learn = learn.to_bf16()

        learn.fit_one_cycle(n_epoch=epoch_count, lr_max=lr_max)

        # Extract final val mAP50
        # recorder.values rows: [train_loss, valid_loss, mAP50]
        if learn.recorder.values:
            final_map50 = float(learn.recorder.values[-1][2])
        else:
            final_map50 = 0.0
        return final_map50

    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0
    finally:
        if learn is not None:
            del learn
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    config = {
        "model_size": "s",
        "img_size": 800,
        "epoch_count": 10,
        "lr_max": 1e-4,
        "pos_weight": 6.0,
        "ema_decay": 0.9999,
        "n_acc": 16,
        "mosaic_p": 0.2,
        "flip_p": 0.5,
        "rot90_p": 0.5,
        "rect_p": 0.3,
        "rect_sl": 0.05,
        "sharpen_min_factor": 0.5,
        "sharpen_max_factor": 1.5,
        "norm_type": "minmax",
        "use_batch_resample": False,
    }
    map50 = train(config)
    print(f"Final mAP50: {map50:.4f}")
