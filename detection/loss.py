"""Loss function bridging fastai <-> ultralytics E2ELoss."""

from typing import Any

import torch
from fastai.callback.core import Callback, CancelBatchException
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import ModelEMA


class YOLO26Loss:
    """Loss wrapper for fastai Learner.

    fastai calls: loss_func(model_output, targets)
    E2ELoss expects: loss(preds, batch_dict)

    The model output in training mode is the raw prediction dict.
    The targets come from our custom collate as a dict with
    batch_idx, cls, bboxes keys.

    E2ELoss is created lazily on first call so that it picks up
    the correct device (model must be on CUDA first).
    """

    def __init__(self, model, pos_weight: float = 1.0):
        """model: YOLO26ForFastai instance.

        E2ELoss reads model.model[-1] for Detect head attributes.

        pos_weight: weight for positive class in BCE classification loss.
            >1 increases cost of false negatives (missed detections),
            pushing the model to predict more aggressively.
            e.g. 3.0 means missing an object costs 3x more than a false alarm.
        """
        self._model = model
        self._pos_weight = pos_weight
        self._e2e_loss: E2ELoss | None = None
        self._device: torch.device | None = None

    def _get_loss(self, device: torch.device) -> E2ELoss:
        """Create or move E2ELoss to the correct device."""
        if self._e2e_loss is None or self._device != device:
            self._e2e_loss = E2ELoss(self._model)
            if self._pos_weight != 1.0:
                pw = torch.tensor([self._pos_weight], device=device)
                for branch in (self._e2e_loss.one2many, self._e2e_loss.one2one):
                    branch.bce = torch.nn.BCEWithLogitsLoss(
                        reduction="none", pos_weight=pw
                    )
            self._device = device
        return self._e2e_loss

    def __call__(self, preds: Any, targets: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss.

        preds: model output
            - training: dict{one2many, one2one}
            - eval: tuple(decoded, dict)
        targets: dict{batch_idx, cls, bboxes}
        """
        # Find the device from predictions
        if isinstance(preds, dict):
            ref_tensor = preds["one2many"]["boxes"]
        else:
            ref_tensor = preds[0]
        device = ref_tensor.device

        # Get loss function on correct device
        e2e_loss = self._get_loss(device)

        # Move targets to correct device
        batch = {k: v.to(device) for k, v in targets.items()}

        # E2ELoss returns [3] tensor (box, cls, dfl) already scaled by batch_size.
        # Ultralytics trainer sums these components to get the final scalar.
        total_loss, _ = e2e_loss(preds, batch)
        return total_loss.sum()

    def update(self):
        """Decay o2m/o2o weights. Call once per epoch."""
        if self._e2e_loss is not None:
            self._e2e_loss.update()


class E2ELossDecayCallback(Callback):
    """Updates E2ELoss one2many/one2one weight ratio each epoch."""

    order = 60

    def after_epoch(self):
        if hasattr(self.learn.loss_func, "update"):
            self.learn.loss_func.update()


class EMACallback(Callback):
    """Exponential Moving Average for model weights.

    Uses ultralytics' ModelEMA: maintains a shadow copy of the model
    that's updated each training step. Swaps in the EMA weights for
    validation and saves, then swaps back for training.
    """

    order = 65  # after loss decay, before metrics

    def __init__(self, decay: float = 0.9999, tau: int = 2000):
        self.decay = decay
        self.tau = tau
        self.ema = None

    def before_fit(self):
        self.ema = ModelEMA(self.learn.model, decay=self.decay, tau=self.tau)

    def after_batch(self):
        if self.training:
            self.ema.update(self.learn.model)

    def before_validate(self):
        # Swap in EMA weights for validation
        self._model_state = {
            k: v.clone() for k, v in self.learn.model.state_dict().items()
        }
        self.learn.model.load_state_dict(self.ema.ema.state_dict())

    def after_validate(self):
        # Swap back training weights
        self.learn.model.load_state_dict(self._model_state)
        del self._model_state

    def after_fit(self):
        # After training, replace model with EMA version (final export)
        self.learn.model.load_state_dict(self.ema.ema.state_dict())


class DetectionGradientAccumulation(Callback):
    """Gradient accumulation that gets batch size from xb (images) instead of yb.

    fastai's built-in GradientAccumulation uses find_bs(yb) which fails
    with our dict-based targets (finds bbox count, not batch size).
    """

    order = 10  # same relative order as fastai's GradientAccumulation
    run_valid = False

    def __init__(self, n_acc: int = 32):
        self.n_acc = n_acc

    def before_fit(self):
        self.count = 0

    def _find_bs(self):
        return self.learn.xb[0].shape[0]

    def after_loss(self):
        self.learn.loss_grad /= self.n_acc / self._find_bs()

    def before_step(self):
        self.learn.loss_grad *= self.n_acc / self._find_bs()
        self.count += self._find_bs()
        if self.count < self.n_acc:
            raise CancelBatchException()
        else:
            self.count = 0
