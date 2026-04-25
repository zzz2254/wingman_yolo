import logging
import threading
from typing import Optional

import cv2
import numpy as np
import torch

from core.config import AppConfig
from core.event_bus import EventBus

log = logging.getLogger(__name__)


class DetectionEngine:

    def __init__(self, config: AppConfig, event_bus: EventBus):
        self._config = config
        self._event_bus = event_bus
        self._model = None
        self._should_stop = threading.Event()
        self._latest_detections: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    @property
    def latest_detections(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_detections is None:
                return None
            return self._latest_detections.copy()

    def _load_model(self):
        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.utils.ops import non_max_suppression
        from utils.torch_utils import select_device

        device = select_device(self._config.device)

        model = AutoBackend(
            self._config.weights,
            device=device,
            dnn=self._config.dnn,
            data=self._config.data,
            fp16=self._config.half,
        )
        model.warmup(imgsz=(1, 3, *self._config.imgsz))
        self._model = model
        self._non_max_suppression = non_max_suppression
        log.info('model loaded: %s on %s', self._config.weights, device)

    def _preprocess(self, im_chw: np.ndarray) -> torch.Tensor:
        im = torch.from_numpy(im_chw).to(self._model.device)
        im = im.half() if self._model.fp16 else im.float()
        im /= 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)
        return im

    def _postprocess(self, pred: torch.Tensor) -> np.ndarray:
        pred = self._non_max_suppression(
            pred, self._config.conf_thres, self._config.iou_thres,
            self._config.classes, self._config.agnostic_nms,
            max_det=self._config.max_det,
        )
        return pred[0].cpu().numpy()

    @torch.no_grad()
    def infer(self, im_chw: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(im_chw)
        pred = self._model(tensor, augment=self._config.augment, visualize=False)
        return self._postprocess(pred)

    def update_detections(self, im_chw: np.ndarray) -> Optional[np.ndarray]:
        try:
            detections = self.infer(im_chw)
        except Exception:
            log.exception('inference error')
            return None
        with self._lock:
            self._latest_detections = detections
        self._event_bus.publish('detect.result', detections=detections.copy())
        return detections

    def run(self, capture):
        self._load_model()
        self._should_stop.clear()
        log.info('detection engine started')

        for im, im0 in capture:
            if self._should_stop.is_set():
                break
            detections = self.update_detections(im)
            if self._config.view_img:
                self._show_frame(im0, detections)
        cv2.destroyAllWindows()
        log.info('detection engine stopped')

    def _show_frame(self, im0: np.ndarray, detections: Optional[np.ndarray]):
        canvas = im0.copy()
        if detections is not None:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                color = (0, 255, 0) if int(cls) == self._config.enemy_label else (255, 0, 0)
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.imshow('AL_Yolo_Detect', canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            self._should_stop.set()

    def stop(self):
        self._should_stop.set()
