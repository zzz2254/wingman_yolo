import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision

from core.config import AppConfig
from core.overlay import DetectionOverlay
from core.event_bus import EventBus
from core.tracker import DetectionTracker

log = logging.getLogger(__name__)


class DetectionEngine:

    def __init__(self, config: AppConfig, event_bus: EventBus):
        self._config = config
        self._event_bus = event_bus
        self._model = None
        self._should_stop = threading.Event()
        self._latest_detections: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._overlay = DetectionOverlay()

        # 时序平滑跟踪器
        self._tracker = DetectionTracker(alpha=0.4, iou_threshold=0.3)

        # 双缓冲流水线: 独立捕获线程,使 CPU grab 与 GPU infer 重叠
        self._capture_thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[tuple] = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()

        # CUDA Stream: 异步 H2D 传输,减少 CPU 等待
        self._cuda_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )

        self._event_bus.subscribe('config.reloaded', self._on_config_reloaded)

    def _on_config_reloaded(self):
        self._overlay.set_enemy_label(self._config.enemy_label)

    @property
    def latest_detections(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_detections

    def _resolve_weights(self) -> str:
        weights = Path(self._config.weights)
        if not self._config.use_tensorrt:
            return str(weights)

        # 检查已存在的 TensorRT 引擎 (.engine 或 .trt)
        for ext in ('.engine', '.trt'):
            engine_path = weights.with_suffix(ext)
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

        weights_path = self._resolve_weights()
        device_str = self._config.device
        if device_str.isdigit():
            device_str = f'cuda:{device_str}'
        device = torch.device(
            device_str if torch.cuda.is_available()
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

        # 同步有效 imgsz 到 config,确保 AimController/TargetSelector/Overlay 使用一致坐标空间
        effective = tuple(imgsz) if isinstance(imgsz, (list, tuple)) else (imgsz, imgsz)
        if effective != tuple(self._config.imgsz):
            log.info('model imgsz %s differs from config %s, syncing to config',
                     effective, self._config.imgsz)
            self._config.imgsz = effective

        model.warmup(imgsz=(1, 3, *imgsz))
        self._model = model
        log.info('model loaded: %s on %s', weights_path, device)

    def _preprocess(self, im_chw: np.ndarray) -> torch.Tensor:
        im = torch.from_numpy(im_chw).to(
            self._model.device, non_blocking=True, dtype=torch.float32,
        )
        if self._model.fp16:
            im = im.half()
        im /= 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)
        return im

    def _postprocess(self, pred: torch.Tensor) -> np.ndarray:
        """后处理：置信度过滤 + GPU NMS（torchvision.ops.nms），消除 CPU-GPU 同步点。

        兼容 CPU/GPU，`torchvision.ops.nms` 两端均可用。
        """
        # 调试：记录每次调用
        self._detect_log_counter = getattr(self, '_detect_log_counter', 0) + 1

        if not isinstance(pred, torch.Tensor):
            log.warning('frame #%d: pred is %s (not a tensor), converting',
                        self._detect_log_counter, type(pred).__name__)
            pred = torch.tensor(pred) if isinstance(pred, (list, tuple)) else pred

        if pred.shape[0] == 0:
            if self._detect_log_counter % 30 == 1:
                log.warning('frame #%d: raw pred shape is %s (0 predictions from model)',
                            self._detect_log_counter, pred.shape)
            return np.empty((0, 6), dtype=np.float32)

        pred = pred[0]  # (6, N) → (N, 6): batch dim → anchors×features
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        # YOLO11: 列 0-3=cxcywh, 列 4+=类别概率（含 objectness）
        boxes = pred[:, :4]  # cxcywh
        cls_conf, cls_id = pred[:, 4:].max(dim=1, keepdim=True)
        score = cls_conf.flatten()

        # 置信度过滤
        mask = score > self._config.conf_thres
        if not mask.any():
            if self._detect_log_counter % 30 == 1:
                max_conf = score.max().item()
                log.info('frame #%d: raw %d preds, max conf=%.3f < thres=%.2f, all filtered',
                         self._detect_log_counter, score.numel(), max_conf, self._config.conf_thres)
            return np.empty((0, 6), dtype=np.float32)

        boxes = boxes[mask]
        score = score[mask]
        cls_id = cls_id[mask]

        # cxcywh → xyxy
        half_w = boxes[:, 2] / 2
        half_h = boxes[:, 3] / 2
        boxes_xyxy = torch.empty_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - half_w   # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - half_h   # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + half_w   # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + half_h   # y2

        # agnostic NMS：类别偏移后再做 NMS
        if self._config.agnostic_nms:
            offset = cls_id.float() * 7680
            nms_boxes = boxes_xyxy.clone()
            nms_boxes[:, :2] += offset
            nms_boxes[:, 2:] += offset
        else:
            nms_boxes = boxes_xyxy

        keep = torchvision.ops.nms(nms_boxes, score, self._config.iou_thres)

        if len(keep) > self._config.max_det:
            keep = keep[:self._config.max_det]

        # 输出：x1, y1, x2, y2, conf, cls
        result = torch.cat([
            boxes_xyxy[keep],
            score[keep].unsqueeze(1),
            cls_id[keep].float(),
        ], dim=1)

        # 调试：统计检测结果
        self._detect_log_counter = getattr(self, '_detect_log_counter', 0) + 1
        n_dets = len(keep)
        if n_dets > 0:
            cls_counts = {}
            for cid in cls_id[keep].int().flatten().tolist():
                cls_counts[cid] = cls_counts.get(cid, 0) + 1
            cls_str = ' '.join(f'cls{k}={v}' for k, v in sorted(cls_counts.items()))
            log.info('frame #%d: %d detections found (max conf=%.3f, %s)',
                     self._detect_log_counter, n_dets, score[keep].max().item(), cls_str)
        elif self._detect_log_counter % 60 == 1:
            log.info('frame #%d: 0 detections (no pred above conf=%.2f)',
                     self._detect_log_counter, self._config.conf_thres)

        return result.cpu().numpy()

    @torch.no_grad()
    def infer(self, im_chw: np.ndarray) -> np.ndarray:
        if self._cuda_stream is not None:
            with torch.cuda.stream(self._cuda_stream):
                tensor = self._preprocess(im_chw)
                pred = self._model(tensor, augment=self._config.augment, visualize=False)
            torch.cuda.synchronize()
        else:
            tensor = self._preprocess(im_chw)
            pred = self._model(tensor, augment=self._config.augment, visualize=False)
        while isinstance(pred, (list, tuple)):
            if len(pred) == 0:
                log.warning('model returned empty %s', type(pred).__name__)
                return np.empty((0, 6), dtype=np.float32)
            pred = pred[0]
        return self._postprocess(pred)

    def update_detections(self, im_chw: np.ndarray, capture_ts: float = 0.0) -> Optional[np.ndarray]:
        try:
            detections = self.infer(im_chw)
        except Exception:
            log.exception('inference error')
            return None
        detections = self._tracker.update(detections)
        with self._lock:
            self._latest_detections = detections
        self._event_bus.publish('detect.result', detections=detections, capture_timestamp=capture_ts)
        return detections

    def _capture_worker(self, capture):
        """捕获线程: 独立运行,使 CPU 截图与 GPU 推理重叠。"""
        for im, im0 in capture:
            if self._should_stop.is_set():
                break
            ts = time.perf_counter()
            with self._frame_lock:
                self._latest_frame = (im, im0, ts)
            self._frame_event.set()

    def run(self, capture):
        self._load_model()
        self._should_stop.clear()
        self._latest_frame = None
        self._frame_event.clear()
        log.info('detection engine started')

        # 启动独立捕获线程 → CPU grab 与 GPU infer 流水线并行
        self._capture_thread = threading.Thread(
            target=self._capture_worker, args=(capture,), daemon=True,
        )
        self._capture_thread.start()
        log.info('capture pipeline thread started')

        if self._config.view_img:
            self._overlay.start()

        while not self._should_stop.is_set():
            if not self._frame_event.wait(timeout=0.1):
                continue
            self._frame_event.clear()

            with self._frame_lock:
                if self._latest_frame is None:
                    continue
                im, im0, capture_ts = self._latest_frame
                self._latest_frame = None

            if self._should_stop.is_set():
                break

            detections = self.update_detections(im, capture_ts)
            if self._config.view_img:
                self._overlay.set_region_offset(capture.capture_left, capture.capture_top)
                self._overlay.show(detections)

        self._overlay.stop()
        log.info('detection engine stopped')

    def stop(self):
        self._should_stop.set()
        self._frame_event.set()  # 唤醒等待中的推理循环
