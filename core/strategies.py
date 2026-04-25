from abc import ABC, abstractmethod
from math import atan2
from typing import Callable, Optional

import numpy as np


class TargetSelector(ABC):

    @abstractmethod
    def select(self, detections: np.ndarray, imgsz: tuple[int, int], enemy_label: int) -> Optional[np.ndarray]:
        ...


class NearestEnemySelector(TargetSelector):

    def select(self, detections: np.ndarray, imgsz: tuple[int, int], enemy_label: int) -> Optional[np.ndarray]:
        if len(detections) == 0:
            return None
        center_x, center_y = imgsz[0] / 2, imgsz[1] / 2
        best = None
        best_dist = float('inf')
        for det in detections:
            if int(det[5]) != enemy_label:
                continue
            bx_cx = (det[0] + det[2]) / 2
            bx_cy = (det[1] + det[3]) / 2
            dist = (bx_cx - center_x) ** 2 + (bx_cy - center_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best = det
        return best


class MouseController(ABC):

    @abstractmethod
    def move(self, target_bbox: np.ndarray, imgsz: tuple[int, int], smooth_factor: float):
        ...


class SmoothAtan2Controller(MouseController):

    def __init__(self, mouse_move_fn: Callable[[float, float], None]):
        self._mouse_move = mouse_move_fn

    def move(self, target_bbox: np.ndarray, imgsz: tuple[int, int], smooth_factor: float):
        size = imgsz[0]
        rel_x = (target_bbox[0] + target_bbox[2] - size) / 2 * smooth_factor
        rel_y = (target_bbox[1] + target_bbox[3] - size) / 2 * smooth_factor
        move_x = atan2(rel_x, size) * size
        move_y = atan2(rel_y, size) * size
        self._mouse_move(move_x, move_y)
