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

    def move_raw(self, dx: float, dy: float):
        """直接移动鼠标(dx,dy)像素,不经过任何计算。子类可选覆写。"""


class SmoothAtan2Controller(MouseController):

    def __init__(self, mouse_move_fn: Callable[[float, float], None]):
        self._mouse_move = mouse_move_fn

    def move_raw(self, dx: float, dy: float):
        self._mouse_move(dx, dy)

    def move(self, target_bbox: np.ndarray, imgsz: tuple[int, int], smooth_factor: float):
        size = imgsz[0]
        rel_x = (target_bbox[0] + target_bbox[2] - size) / 2 * smooth_factor
        rel_y = (target_bbox[1] + target_bbox[3] - size) / 2 * smooth_factor
        move_x = atan2(rel_x, size) * size
        move_y = atan2(rel_y, size) * size
        self._mouse_move(move_x, move_y)
        return move_x, move_y


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
        self._px_per_unit: float = 1.0  # 像素/输入单位 比率,由 calibrate() 测量
        self._move_counter: int = 0

    def set_target_velocity(self, vx: float, vy: float):
        """由 AimController 每帧调用,传入目标速度用于前馈补偿。"""
        self._target_vx = vx
        self._target_vy = vy

    def move_raw(self, dx: float, dy: float):
        """直接移动鼠标,绕过 PID 计算。"""
        self._mouse_move(dx, dy)

    def calibrate(self):
        """测量鼠标输入单位到屏幕像素的比率。

        发送已知移动量,测量光标实际位移,计算 px_per_unit。
        如果光标未移动 (如游戏中光标锁定),保留默认值 1.0。
        """
        TEST_STEPS = 80  # 测试移动量 (输入单位)

        before = wintypes.POINT()
        windll.user32.GetCursorPos(byref(before))

        self._mouse_move(TEST_STEPS, 0)
        time.sleep(0.06)

        after = wintypes.POINT()
        windll.user32.GetCursorPos(byref(after))

        actual_dx = after.x - before.x  # 带符号: 正=右移
        if abs(actual_dx) > 0:
            self._px_per_unit = abs(actual_dx) / TEST_STEPS
            log.info('calibration: %d input units → %d px, ratio=%.4f px/unit',
                     TEST_STEPS, actual_dx, self._px_per_unit)
            # 光标归位: 发送相反方向同等屏幕像素量的移动
            back_units = -actual_dx / self._px_per_unit
            self._mouse_move(back_units, 0)
            time.sleep(0.03)
            # 微调修正残余偏差
            after2 = wintypes.POINT()
            windll.user32.GetCursorPos(byref(after2))
            residual = before.x - after2.x
            if abs(residual) > 0:
                self._mouse_move(residual / self._px_per_unit, 0)
        else:
            self._px_per_unit = 1.0
            log.info('calibration: cursor did not move (locked?), using ratio=1.0')

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

        # ── 目标切换检测：误差突变 → 重置积分和微分 ──
        _TARGET_SWITCH_THRESHOLD = 200.0  # 误差变化超过此值视为目标切换
        if abs(error_x - self._prev_error_x) > _TARGET_SWITCH_THRESHOLD or \
           abs(error_y - self._prev_error_y) > _TARGET_SWITCH_THRESHOLD:
            self._integral_x = 0.0
            self._integral_y = 0.0
            self._prev_error_x = error_x
            self._prev_error_y = error_y
            log.info('pid: target switch detected — integrator+derivative reset')

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

        # ── 像素→输入单位转换 ──
        output_x /= self._px_per_unit
        output_y /= self._px_per_unit

        # ── 输出限幅：防止目标切换时微分爆炸导致输出失控 ──
        _MAX_OUTPUT = 100.0
        output_x = float(np.clip(output_x, -_MAX_OUTPUT, _MAX_OUTPUT))
        output_y = float(np.clip(output_y, -_MAX_OUTPUT, _MAX_OUTPUT))

        if self._move_counter % 10 == 0:
            log.info('pid: move — err=(%+.0f,%+.0f) dist=%.0fpx P=(%+.1f,%+.1f) I=(%+.1f,%+.1f) D=(%+.1f,%+.1f) out=(%+.1f,%+.1f) kp=%.3f zone=%s',
                       error_x, error_y, distance,
                       kp * error_x, kp * error_y,
                       self._integral_x, self._integral_y,
                       kd * derivative_x, kd * derivative_y,
                       output_x, output_y, kp,
                       'far' if distance > self._FAR_DIST else 'near' if distance < self._NEAR_DIST else 'mid')
        self._move_counter += 1
        self._mouse_move(output_x, output_y)
        return output_x, output_y
