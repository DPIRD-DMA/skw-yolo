"""YOLO26 model wrapper for fastai Learner."""

import torch.nn as nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import intersect_dicts


class YOLO26ForFastai(nn.Module):
    """Wraps ultralytics DetectionModel for use with fastai Learner.

    Key requirements for E2ELoss compatibility:
    - model.model[-1] must return the Detect head
    - model.args must contain hyperparameters (box, cls, dfl, epochs)
    - model.stride must be available

    Forward pass:
    - Training: returns pred dict {one2many, one2one}
    - Eval: returns (decoded [B,300,6], raw_dict)
    """

    def __init__(
        self,
        model_size: str = "n",
        nc: int = 2,
        pretrained: bool = True,
        epochs: int = 30,
    ):
        super().__init__()

        # Build fresh model with correct nc
        cfg = f"yolo26{model_size}.yaml"
        self._det_model = DetectionModel(cfg, ch=3, nc=nc, verbose=False)

        if pretrained:
            self._load_coco_weights(model_size)

        # Enable gradients (frozen by default from ultralytics loading)
        for p in self._det_model.parameters():
            p.requires_grad_(True)

        # Set args that E2ELoss reads
        args = get_cfg()
        args.epochs = epochs
        self._det_model.args = args

    def _load_coco_weights(self, model_size: str):
        """Load COCO pretrained weights with partial matching for different nc."""
        coco_model = YOLO(f"yolo26{model_size}.pt").model
        coco_sd = coco_model.float().state_dict()
        model_sd = self._det_model.state_dict()
        matched = intersect_dicts(coco_sd, model_sd)
        self._det_model.load_state_dict(matched, strict=False)
        print(f"Loaded {len(matched)}/{len(model_sd)} pretrained weights")

    @property
    def model(self):
        """Expose nn.Sequential so model.model[-1] returns Detect head."""
        return self._det_model.model

    @model.setter
    def model(self, value):
        self._det_model.model = value

    @property
    def args(self):
        return self._det_model.args

    @args.setter
    def args(self, value):
        self._det_model.args = value

    @property
    def stride(self):
        return self._det_model.stride

    @property
    def end2end(self):
        return self._det_model.end2end

    def forward(self, x):
        return self._det_model(x)

    def train(self, mode=True):
        self._det_model.train(mode)
        return super().train(mode)

    def eval(self):
        self._det_model.eval()
        return super().eval()
