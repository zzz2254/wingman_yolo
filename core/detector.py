import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.config import AppConfig
from core.display import Display
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
        self._display = Display(on_esc=self.stop)

        self._event_bus.subscribe('config.reloaded', self._on_config_reloaded)

    def _on_config_reloaded(self):
        self._display.set_show_fps(self._config.show_fps)

    @property
    def latest_detections(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_detections

    def _resolve_weights(self) -> str:
        weights = Path(self._config.weights)
        if not self._config.use_tensorrt:
            return str(weights)

        engine_path = weights.with_suffix('.engine')
        if engine_path.exists():
            log.info('using existing TensorRT engine: %s', engine_path)
            return str(engine_path)

        log.info('TensorRT engine not found, attempting export...')
        try:
            import os as _os
            _os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

            from ultralytics import YOLO
            yolo = YOLO(str(weights))
            exported = yolo.export(
                format='engine',
                half=self._config.half,
                imgsz=self._config.imgsz,
                device=self._config.device,
                dynamic=False,
                simplify=True,
                verbose=False,
            )
            log.info('TensorRT export succeeded: %s', exported)
            return str(exported)
        except Exception:
            log.exception('TensorRT export failed, falling back to PyTorch')
            return str(weights)

    def _load_model(self):
        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.utils.ops import non_max_suppression

        weights_path = self._resolve_weights()
        device = torch.device(
            self._config.device if torch.cuda.is_available()
            else 'cpu'
        )

        model = AutoBackend(
            weights_path,
            device=device,
            dnn=self._config.dnn,
            data=self._config.data,
            fp16=self._config.half,
        )
        imgsz = self._config.imgsz
        if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
            m_yaml = getattr(model.model, 'yaml', None) or {}
            imgsz = m_yaml.get('imgsz', imgsz) if isinstance(m_yaml, dict) else imgsz

        model.warmup(imgsz=(1, 3, *imgsz))
        self._model = model
        self._non_max_suppression = non_max_suppression
        log.info('model loaded: %s on %s', weights_path, device)

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
        self._event_bus.publish('detect.result', detections=detections)
        return detections

    def run(self, capture):
        self._load_model()
        self._should_stop.clear()
        log.info('detection engine started')

        if self._config.view_img:
            self._display.start()

        for im, im0 in capture:
            if self._should_stop.is_set():
                break
            detections = self.update_detections(im)
            if self._config.view_img:
                self._show_frame(im0, detections)
        self._display.stop()
        log.info('detection engine stopped')

    def _show_frame(self, im0: np.ndarray, detections: Optional[np.ndarray]):
        canvas = im0.copy()
        if detections is not None:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                color = (0, 255, 0) if int(cls) == self._config.enemy_label else (255, 0, 0)
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        self._display.show(canvas)

    def stop(self):
        self._should_stop.set()
