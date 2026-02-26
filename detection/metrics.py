"""mAP metric for fastai Learner with YOLO26."""

import numpy as np
import torch
from fastai.metrics import JaccardCoeffMulti, Metric
from ultralytics.utils.metrics import box_iou


class RasterIoU(JaccardCoeffMulti):
    """Foreground IoU between rasterized GT and predicted bounding boxes.

    Rasterizes all boxes into binary masks (object vs background) on the
    image grid and computes foreground (class 1) IoU via JaccardCoeffMulti.
    """

    def __init__(self, conf_thresh: float = 0.25):
        super().__init__(axis=1)
        self.conf_thresh = conf_thresh

    @property
    def name(self):
        return "raster_iou"

    def accumulate(self, learn):
        preds = learn.pred
        targets = learn.yb[0]

        if isinstance(preds, tuple):
            decoded = preds[0]  # [B, 300, 6] = x1,y1,x2,y2,conf,cls
        else:
            return

        batch_idx = targets["batch_idx"]
        gt_bboxes = targets["bboxes"]  # [N, 4] xywh normalized

        bs = decoded.shape[0]
        img_size = learn.xb[0].shape[-1]

        gt_masks = torch.zeros(bs, img_size, img_size, device=decoded.device)
        pred_masks = torch.zeros(bs, img_size, img_size, device=decoded.device)

        for i in range(bs):
            # Rasterize GT boxes (normalized xywh -> pixel xyxy)
            for box in gt_bboxes[batch_idx == i]:
                x1 = int(max(0, (box[0] - box[2] / 2) * img_size))
                y1 = int(max(0, (box[1] - box[3] / 2) * img_size))
                x2 = int(min(img_size, (box[0] + box[2] / 2) * img_size))
                y2 = int(min(img_size, (box[1] + box[3] / 2) * img_size))
                gt_masks[i, y1:y2, x1:x2] = 1

            # Rasterize predicted boxes above confidence threshold
            det = decoded[i]
            det = det[det[:, 4] > self.conf_thresh]
            for box in det:
                x1 = int(max(0, box[0].item()))
                y1 = int(max(0, box[1].item()))
                x2 = int(min(img_size, box[2].item()))
                y2 = int(min(img_size, box[3].item()))
                pred_masks[i, y1:y2, x1:x2] = 1

        # Accumulate intersection/union for foreground only (class 1)
        pred_flat = pred_masks.flatten()
        targ_flat = gt_masks.flatten()
        c = 1
        p = (pred_flat == c).float()
        t = (targ_flat == c).float()
        c_inter = (p * t).sum().item()
        c_union = (p + t).sum().item()
        if c in self.inter:
            self.inter[c] += c_inter
            self.union[c] += c_union
        else:
            self.inter[c] = c_inter
            self.union[c] = c_union


class YOLOmAP50(Metric):
    """Computes mAP@50 during fastai validation."""

    def __init__(self, nc: int = 2):
        self.nc = nc
        self.iou_threshold = 0.5

    @property
    def name(self):
        return "mAP50"

    def reset(self):
        self.stats: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

    def accumulate(self, learn):
        """Called after each validation batch."""
        preds = learn.pred
        targets = learn.yb[0]

        # In eval mode, preds = (decoded_y [B,300,6], raw_dict)
        if isinstance(preds, tuple):
            decoded = preds[0]  # [B, 300, 6] = x1,y1,x2,y2,conf,cls
        else:
            return

        batch_idx = targets["batch_idx"]
        gt_cls = targets["cls"]  # [N, 1]
        gt_bboxes = targets["bboxes"]  # [N, 4] xywh normalized

        bs = decoded.shape[0]
        img_size = learn.xb[0].shape[-1]

        # Convert gt bboxes from normalized xywh to pixel xyxy
        gt_bboxes_xyxy = self._xywh_norm_to_xyxy_pixel(gt_bboxes, img_size)

        for i in range(bs):
            # Get predictions for this image
            det = decoded[i]  # [300, 6]
            mask = det[:, 4] > 0.001
            det = det[mask]

            # Get ground truth for this image
            gt_mask = batch_idx == i
            gt_cls_i = gt_cls[gt_mask].squeeze(-1)  # [M]
            gt_boxes_i = gt_bboxes_xyxy[gt_mask]  # [M, 4]

            if len(det) == 0:
                if len(gt_cls_i) > 0:
                    self.stats.append(
                        (
                            torch.zeros(0, dtype=torch.bool),
                            torch.zeros(0),
                            torch.zeros(0),
                            gt_cls_i.cpu(),
                        )
                    )
                continue

            pred_boxes = det[:, :4]  # [D, 4] xyxy pixel
            pred_conf = det[:, 4]
            pred_cls = det[:, 5]

            if len(gt_cls_i) == 0:
                self.stats.append(
                    (
                        torch.zeros(len(det), dtype=torch.bool),
                        pred_conf.cpu(),
                        pred_cls.cpu(),
                        torch.zeros(0),
                    )
                )
                continue

            # Compute IoU and match
            iou = box_iou(gt_boxes_i, pred_boxes)  # [M, D]
            tp = self._match_predictions(pred_cls, gt_cls_i, iou)

            self.stats.append(
                (
                    tp.cpu(),
                    pred_conf.cpu(),
                    pred_cls.cpu(),
                    gt_cls_i.cpu(),
                )
            )

    def _match_predictions(
        self,
        pred_cls: torch.Tensor,
        gt_cls: torch.Tensor,
        iou: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy IoU matching. Returns TP boolean tensor for predictions."""
        correct = torch.zeros(len(pred_cls), dtype=torch.bool, device=pred_cls.device)
        iou_match = iou >= self.iou_threshold
        matched_gt: set[int] = set()

        # Sort predictions by IoU (greedy matching)
        for pred_idx in range(len(pred_cls)):
            best_iou = -1.0
            best_gt = -1
            for gt_idx in range(len(gt_cls)):
                if (
                    gt_idx not in matched_gt
                    and iou_match[gt_idx, pred_idx]
                    and pred_cls[pred_idx] == gt_cls[gt_idx]
                    and iou[gt_idx, pred_idx] > best_iou
                ):
                    best_iou = iou[gt_idx, pred_idx].item()
                    best_gt = gt_idx

            if best_gt >= 0:
                correct[pred_idx] = True
                matched_gt.add(best_gt)

        return correct

    @staticmethod
    def _xywh_norm_to_xyxy_pixel(xywh: torch.Tensor, img_size: int) -> torch.Tensor:
        """Convert normalized xywh to pixel xyxy."""
        if xywh.shape[0] == 0:
            return torch.zeros(0, 4, device=xywh.device)
        xyxy = torch.zeros_like(xywh)
        xyxy[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2) * img_size
        xyxy[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2) * img_size
        xyxy[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2) * img_size
        xyxy[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2) * img_size
        return xyxy

    @property
    def value(self):
        if not self.stats:
            return 0.0

        all_tp = [s[0] for s in self.stats if len(s[0]) > 0]
        all_conf = [s[1] for s in self.stats if len(s[1]) > 0]
        all_pred_cls = [s[2] for s in self.stats if len(s[2]) > 0]
        all_target_cls = [s[3] for s in self.stats if len(s[3]) > 0]

        if not all_tp or not all_target_cls:
            return 0.0

        tp = torch.cat(all_tp)
        conf = torch.cat(all_conf)
        pred_cls = torch.cat(all_pred_cls)
        target_cls = torch.cat(all_target_cls)

        if len(tp) == 0 or len(target_cls) == 0:
            return 0.0

        # Sort by confidence
        sort_idx = torch.argsort(conf, descending=True)
        tp = tp[sort_idx].numpy()
        pred_cls = pred_cls[sort_idx].numpy()
        target_cls = target_cls.numpy()

        ap_sum = 0.0
        n_classes_found = 0

        for c in range(self.nc):
            mask = pred_cls == c
            n_gt = (target_cls == c).sum()
            n_pred = mask.sum()

            if n_gt == 0 or n_pred == 0:
                continue

            tp_c = tp[mask].astype(np.float64).cumsum()
            fp_c = (~tp[mask].astype(bool)).astype(np.float64).cumsum()

            recall = tp_c / n_gt
            precision = tp_c / (tp_c + fp_c)

            ap = self._compute_ap(recall, precision)
            ap_sum += ap
            n_classes_found += 1

        return ap_sum / max(n_classes_found, 1)

    @staticmethod
    def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
        """Compute AP using all-point interpolation."""
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Monotonically decreasing precision
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        # Find recall change points
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # Sum area under curve
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return float(ap)
