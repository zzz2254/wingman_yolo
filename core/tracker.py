"""轻量级 IoU 跨帧跟踪器,提供检测框时序平滑。

对相邻帧的检测结果做 IoU 匹配 + EMA 平滑,
消除单帧独立推理引起的检测框闪烁。
"""

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """两个 xyxy 框的 IoU。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DetectionTracker:
    """IoU 匹配 + EMA 平滑的轻量跟踪器。

    不替代完整的多目标跟踪器 (如 ByteTrack),
    仅用于消除逐帧独立推理产生的 bbox 抖动。
    """

    def __init__(self, alpha: float = 0.4, iou_threshold: float = 0.3, max_age: int = 5):
        self._alpha = alpha
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._tracks: list[dict] = []  # [{bbox, id, age, cls, conf}]
        self._next_id: int = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """对输入的检测结果做匹配+平滑,返回同格式的 np.ndarray。"""
        if detections.shape[0] == 0:
            # 无检测: 老化所有 track
            for t in self._tracks:
                t['age'] += 1
            self._tracks = [t for t in self._tracks if t['age'] <= self._max_age]
            return detections

        matched_track_indices: set = set()
        matched_det_indices: set = set()

        # 对每个新检测,找当前 track 中的最佳 IoU 匹配
        for di, det in enumerate(detections):
            best_iou = 0.0
            best_ti = -1
            for ti, track in enumerate(self._tracks):
                if ti in matched_track_indices:
                    continue
                iou = _box_iou(det[:4], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_ti = ti

            if best_iou >= self._iou_threshold and best_ti >= 0:
                matched_track_indices.add(best_ti)
                matched_det_indices.add(di)

                track = self._tracks[best_ti]
                # EMA 平滑 bbox
                a = self._alpha
                track['bbox'] = (
                    a * det[0] + (1 - a) * track['bbox'][0],
                    a * det[1] + (1 - a) * track['bbox'][1],
                    a * det[2] + (1 - a) * track['bbox'][2],
                    a * det[3] + (1 - a) * track['bbox'][3],
                )
                track['conf'] = det[4]
                track['cls'] = int(det[5])
                track['age'] = 0

        # 未匹配的 track: 老化
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_indices:
                track['age'] += 1

        # 删除过期 track
        self._tracks = [t for t in self._tracks if t['age'] <= self._max_age]

        # 未匹配的检测: 创建新 track
        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                self._tracks.append({
                    'bbox': (float(det[0]), float(det[1]), float(det[2]), float(det[3])),
                    'id': self._next_id,
                    'age': 0,
                    'cls': int(det[5]),
                    'conf': float(det[4]),
                })
                self._next_id += 1

        # 构建输出: 所有存活 track → np.ndarray
        alive = [t for t in self._tracks if t['age'] == 0]
        if not alive:
            return np.empty((0, 6), dtype=np.float32)

        result = np.array([
            [t['bbox'][0], t['bbox'][1], t['bbox'][2], t['bbox'][3], t['conf'], t['cls']]
            for t in alive
        ], dtype=np.float32)

        log.debug('tracker: %d dets → %d tracks (total tracks: %d)',
                  len(detections), len(alive), len(self._tracks))
        return result

    def reset(self):
        self._tracks.clear()
        self._next_id = 0
