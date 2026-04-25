import time
from abc import ABC, abstractmethod
from math import atan2
from typing import Callable, Optional

import numpy as np

import pyautogui


class TargetSelector(ABC):

    @abstractmethod
    def select(self, detections: np.ndarray, imgsz: tuple[int, int], enemy_label: int) -> Optional[np.ndarray]:
        ...


class NearestEnemySelector(TargetSelector):
    """离屏幕中心最近的敌人（默认）。"""

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


class LargestEnemySelector(TargetSelector):
    """检测框面积最大的敌人（≈离玩家最近、威胁最大）。"""

    def select(self, detections: np.ndarray, imgsz: tuple[int, int], enemy_label: int) -> Optional[np.ndarray]:
        if len(detections) == 0:
            return None
        best = None
        best_area = 0.0
        for det in detections:
            if int(det[5]) != enemy_label:
                continue
            area = (det[2] - det[0]) * (det[3] - det[1])
            if area > best_area:
                best_area = area
                best = det
        return best


class CrosshairProximitySelector(TargetSelector):
    """离鼠标光标实际位置最近的敌人（与准星可能不同,适用于跟枪场景）。"""

    def select(self, detections: np.ndarray, imgsz: tuple[int, int], enemy_label: int) -> Optional[np.ndarray]:
        if len(detections) == 0:
            return None
        cursor_x, cursor_y = pyautogui.position()
        # 将光标位置映射到截屏区域的坐标系
        screen_w, screen_h = pyautogui.size()
        half_size = imgsz[0] / 2
        center_x, center_y = screen_w / 2, screen_h / 2
        rel_cx = (cursor_x - center_x) + half_size
        rel_cy = (cursor_y - center_y) + half_size

        best = None
        best_dist = float('inf')
        for det in detections:
            if int(det[5]) != enemy_label:
                continue
            bx_cx = (det[0] + det[2]) / 2
            bx_cy = (det[1] + det[3]) / 2
            dist = (bx_cx - rel_cx) ** 2 + (bx_cy - rel_cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best = det
        return best


class StrategyRouter(TargetSelector):
    """根据配置路由到具体的目标选择策略,支持运行时切换。"""

    STRATEGIES = {
        'nearest': NearestEnemySelector(),
        'largest': LargestEnemySelector(),
        'crosshair': CrosshairProximitySelector(),
    }

    def __init__(self, config):
        self._config = config
        self._current: Optional[TargetSelector] = None

    def select(self, detections: np.ndarray, imgsz: tuple[int, int], enemy_label: int) -> Optional[np.ndarray]:
        name = self._config.target_strategy
        selector = self.STRATEGIES.get(name)
        if selector is None:
            selector = self.STRATEGIES['nearest']
        return selector.select(detections, imgsz, enemy_label)


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


class PidMouseController(MouseController):
    """PID 闭环鼠标控制器,比开环 atan2 更平滑,减少 overshoot 和抖动。

    所有增益参数从 AppConfig 热加载,运行时修改配置立即生效。
    """

    def __init__(
        self,
        mouse_move_fn: Callable[[float, float], None],
        kp: float = 0.35,
        ki: float = 0.02,
        kd: float = 0.08,
        max_integral: float = 30.0,
        deadband: float = 2.0,
    ):
        self._mouse_move = mouse_move_fn
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.deadband = deadband

        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._last_time: Optional[float] = None

    def update_gains(self, kp: float, ki: float, kd: float, max_integral: float, deadband: float):
        """运行时更新 PID 参数（热加载入口）。"""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.deadband = deadband

    def reset(self):
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._last_time = None

    def move(self, target_bbox: np.ndarray, imgsz: tuple[int, int], smooth_factor: float):
        _ = smooth_factor  # unused by PID (uses its own tuned gains)

        now = time.perf_counter()
        dt = now - self._last_time if self._last_time is not None else 1 / 120
        dt = max(dt, 1 / 1000)
        self._last_time = now

        size = imgsz[0]
        target_cx = (target_bbox[0] + target_bbox[2]) / 2
        target_cy = (target_bbox[1] + target_bbox[3]) / 2
        error_x = target_cx - size / 2
        error_y = target_cy - size / 2

        # Deadband: 准星已在目标附近时不移动
        distance = (error_x ** 2 + error_y ** 2) ** 0.5
        if distance < self.deadband:
            self._integral_x = 0.0
            self._integral_y = 0.0
            self._prev_error_x = error_x
            self._prev_error_y = error_y
            return

        # Integral (with windup clamp)
        self._integral_x = np.clip(
            self._integral_x + error_x * dt * self.ki,
            -self.max_integral, self.max_integral,
        )
        self._integral_y = np.clip(
            self._integral_y + error_y * dt * self.ki,
            -self.max_integral, self.max_integral,
        )

        # Derivative
        derivative_x = (error_x - self._prev_error_x) / dt
        derivative_y = (error_y - self._prev_error_y) / dt

        # PID output
        output_x = self.kp * error_x + self._integral_x + self.kd * derivative_x
        output_y = self.kp * error_y + self._integral_y + self.kd * derivative_y

        self._prev_error_x = error_x
        self._prev_error_y = error_y

        self._mouse_move(output_x, output_y)
