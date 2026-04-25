import logging
import random
import time
from abc import ABC, abstractmethod
from ctypes import byref, windll, wintypes
from math import atan2
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


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
        best_fallback = None
        best_dist = float('inf')
        best_fallback_dist = float('inf')
        for det in detections:
            bx_cx = (det[0] + det[2]) / 2
            bx_cy = (det[1] + det[3]) / 2
            dist = (bx_cx - center_x) ** 2 + (bx_cy - center_y) ** 2
            if int(det[5]) == enemy_label:
                if dist < best_dist:
                    best_dist = dist
                    best = det
            else:
                if dist < best_fallback_dist:
                    best_fallback_dist = dist
                    best_fallback = det
        if best is not None:
            return best
        # fallback: 没有 enemy_label 目标时,取离中心最近的其他目标
        if best_fallback is not None:
            return best_fallback
        return None


class LargestEnemySelector(TargetSelector):
    """检测框面积最大的敌人（≈离玩家最近、威胁最大）。"""

    def select(self, detections: np.ndarray, imgsz: tuple[int, int], enemy_label: int) -> Optional[np.ndarray]:
        if len(detections) == 0:
            return None
        best = None
        best_fallback = None
        best_area = 0.0
        best_fallback_area = 0.0
        for det in detections:
            area = (det[2] - det[0]) * (det[3] - det[1])
            if int(det[5]) == enemy_label:
                if area > best_area:
                    best_area = area
                    best = det
            else:
                if area > best_fallback_area:
                    best_fallback_area = area
                    best_fallback = det
        if best is not None:
            return best
        if best_fallback is not None:
            return best_fallback
        return None


class CrosshairProximitySelector(TargetSelector):
    """离鼠标光标实际位置最近的敌人（与准星可能不同,适用于跟枪场景）。"""

    def select(self, detections: np.ndarray, imgsz: tuple[int, int], enemy_label: int) -> Optional[np.ndarray]:
        if len(detections) == 0:
            return None
        cursor = wintypes.POINT()
        windll.user32.GetCursorPos(byref(cursor))
        cursor_x, cursor_y = cursor.x, cursor.y
        screen_w = windll.user32.GetSystemMetrics(0)
        screen_h = windll.user32.GetSystemMetrics(1)
        half_size = imgsz[0] / 2
        center_x, center_y = screen_w / 2, screen_h / 2
        rel_cx = (cursor_x - center_x) + half_size
        rel_cy = (cursor_y - center_y) + half_size

        best = None
        best_fallback = None
        best_dist = float('inf')
        best_fallback_dist = float('inf')
        for det in detections:
            bx_cx = (det[0] + det[2]) / 2
            bx_cy = (det[1] + det[3]) / 2
            dist = (bx_cx - rel_cx) ** 2 + (bx_cy - rel_cy) ** 2
            if int(det[5]) == enemy_label:
                if dist < best_dist:
                    best_dist = dist
                    best = det
            else:
                if dist < best_fallback_dist:
                    best_fallback_dist = dist
                    best_fallback = det
        if best is not None:
            return best
        if best_fallback is not None:
            return best_fallback
        return None


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

    支持速度前馈、分段增益调度、人类化噪声注入。
    所有增益参数从 AppConfig 热加载,运行时修改配置立即生效。
    """

    # ── 分段增益调度阈值与倍率 ──
    _FAR_DIST: float = 200.0
    _NEAR_DIST: float = 50.0
    _FAR_KP_MUL: float = 1.4
    _FAR_KI_MUL: float = 0.5
    _FAR_KD_MUL: float = 0.6
    _NEAR_KP_MUL: float = 0.6
    _NEAR_KI_MUL: float = 1.5
    _NEAR_KD_MUL: float = 1.5

    # ── 噪声注入阈值（距离小于此值时注入,大角度甩枪不叠加） ──
    _NOISE_MAX_DIST: float = 80.0

    def __init__(
        self,
        mouse_move_fn: Callable[[float, float], None],
        kp: float = 0.35,
        ki: float = 0.02,
        kd: float = 0.08,
        max_integral: float = 30.0,
        deadband: float = 2.0,
        kff: float = 0.0,
        noise_amplitude: float = 0.5,
        sensitivity: float = 1.0,
    ):
        self._mouse_move = mouse_move_fn
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.deadband = deadband
        self.kff = kff
        self.noise_amplitude = noise_amplitude
        self._sensitivity = sensitivity

        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._last_time: Optional[float] = None
        self._target_vx: float = 0.0
        self._target_vy: float = 0.0

    def set_target_velocity(self, vx: float, vy: float):
        """由 AimController 每帧调用,传入目标速度用于前馈补偿。"""
        self._target_vx = vx
        self._target_vy = vy

    def update_gains(
        self, kp: float, ki: float, kd: float, max_integral: float,
        deadband: float, kff: Optional[float] = None,
        noise_amplitude: Optional[float] = None,
        sensitivity: Optional[float] = None,
    ):
        """运行时更新 PID 参数（热加载入口）。"""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.deadband = deadband
        if kff is not None:
            self.kff = kff
        if noise_amplitude is not None:
            self.noise_amplitude = noise_amplitude
        if sensitivity is not None:
            self._sensitivity = sensitivity

    def reset(self):
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._last_time = None
        self._target_vx = 0.0
        self._target_vy = 0.0

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
            log.debug('pid: deadband skip — dist=%.1fpx < deadband=%.1fpx', distance, self.deadband)
            self._integral_x = 0.0
            self._integral_y = 0.0
            self._prev_error_x = error_x
            self._prev_error_y = error_y
            return

        # ── 分段增益调度（根据距离调整 PID 参数） ──
        kp = self.kp
        ki = self.ki
        kd = self.kd
        if distance > self._FAR_DIST:
            kp *= self._FAR_KP_MUL
            ki *= self._FAR_KI_MUL
            kd *= self._FAR_KD_MUL
        elif distance < self._NEAR_DIST:
            kp *= self._NEAR_KP_MUL
            ki *= self._NEAR_KI_MUL
            kd *= self._NEAR_KD_MUL

        # Integral (with windup clamp)
        self._integral_x = np.clip(
            self._integral_x + error_x * dt * ki,
            -self.max_integral, self.max_integral,
        )
        self._integral_y = np.clip(
            self._integral_y + error_y * dt * ki,
            -self.max_integral, self.max_integral,
        )

        # Derivative
        derivative_x = (error_x - self._prev_error_x) / dt
        derivative_y = (error_y - self._prev_error_y) / dt

        # PID output
        output_x = kp * error_x + self._integral_x + kd * derivative_x
        output_y = kp * error_y + self._integral_y + kd * derivative_y

        # ── 速度前馈（velocity feedforward） ──
        output_x += self.kff * self._target_vx
        output_y += self.kff * self._target_vy

        self._prev_error_x = error_x
        self._prev_error_y = error_y

        # ── 人类化噪声注入（仅在跟枪时叠加,甩枪时跳过） ──
        if distance < self._NOISE_MAX_DIST and self.noise_amplitude > 0:
            sigma = self.noise_amplitude / 3  # 3σ ≈ ±amplitude
            output_x += random.gauss(0, sigma)
            output_y += random.gauss(0, sigma)

        # ── 灵敏度倍率 ──
        output_x *= self._sensitivity
        output_y *= self._sensitivity

        log.debug('pid: move — err=(%+.0f,%+.0f) dist=%.0fpx out=(%+.1f,%+.1f) dt=%.1fms',
                   error_x, error_y, distance, output_x, output_y, dt * 1000)
        self._mouse_move(output_x, output_y)
