# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
from typing import List, Tuple, Optional, Union

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class DetectionPredictor(BasePredictor):
    def get_annotator(self, img: torch.Tensor) -> Annotator:
        """Initialize the Annotator for the given image."""
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """Preprocess the image for prediction."""
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # Convert uint8 to fp16/32
        img /= 255  # Normalize to range [0.0, 1.0]
        return img

    def postprocess(self, preds: List[torch.Tensor], img: torch.Tensor, orig_img: torch.Tensor) -> List[torch.Tensor]:
        """Postprocess predictions, including NMS and scaling."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx: int, preds: List[torch.Tensor], batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> str:
        """Write the results to file and/or display them."""
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # Expand for batch dimension

        self.seen += 1
        im0 = im0.copy()
        frame = self.dataset.count if self.webcam else getattr(self.dataset, 'frame', 0)
        self.data_path = p

        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # Log image size
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        log_string += self._format_detection_log(det)

        if len(det) > 0:
            self._process_detections(det, im0)

        return log_string

    def _format_detection_log(self, det: torch.Tensor) -> str:
        """Format the detection log string."""
        log_string = ""
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # Count detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def _process_detections(self, det: torch.Tensor, im0: torch.Tensor) -> None:
        """Process and save detections."""
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            cls = int(cls)
            if self.args.save_txt:
                self._save_txt(xyxy, conf, cls, gn)

            if self.args.save or self.args.save_crop or self.args.show:
                self._annotate_image(xyxy, conf, cls, im0)
                
            if self.args.save_crop:
                self._save_crop(xyxy, cls, im0)

    def _save_txt(self, xyxy: List[float], conf: float, cls: int, gn: torch.Tensor) -> None:
        """Save detection results to a text file."""
        xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # Normalized xywh
        line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # Label format
        with open(f'{self.txt_path}.txt', 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def _annotate_image(self, xyxy: List[float], conf: float, cls: int, im0: torch.Tensor) -> None:
        """Add bounding box and label to the image."""
        label = None if self.args.hide_labels else (
            self.model.names[cls] if self.args.hide_conf else f'{self.model.names[cls]} {conf:.2f}')
        self.annotator.box_label(xyxy, label, color=colors(cls, True))

    def _save_crop(self, xyxy: List[float], cls: int, im0: torch.Tensor) -> None:
        """Save cropped image."""
        imc = im0.copy()
        save_one_box(xyxy,
                     imc,
                     file=self.save_dir / 'crops' / self.model.model.names[cls] / f'{self.data_path.stem}.jpg',
                     BGR=True)


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    """Main function to run predictions."""
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # Check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()

