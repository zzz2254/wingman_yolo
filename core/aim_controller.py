import logging
import queue
import threading
import time
from ctypes import byref, windll, wintypes
from enum import Enum, auto
from math import hypot
from typing import Optional

import numpy as np

from core.config import AppConfig
from core.event_bus import EventBus
from core.state_machine import AppState, StateMachine
from core.strategies import MouseController, TargetSelector

log = logging.getLogger(__name__)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class AssistState(Enum):
    FREE = auto()       # 未吸附,等待准星靠近目标
    ENGAGED = auto()    # 已吸附,磁力拉向锁定目标
    OVERRIDE = auto()   # 手动接管,停止辅助


class AimController:

    def __init__(
        self,
        config: AppConfig,
        event_bus: EventBus,
        state_machine: StateMachine,
        target_selector: TargetSelector,
        mouse_controller: MouseController,
    ):
        self._config = config
        self._event_bus = event_bus
        self._state_machine = state_machine
        self._target_selector = target_selector
        self._mouse_controller = mouse_controller

        self._aim_disabled = False
        self._should_stop = threading.Event()

        # ── 捕获区域 (由外部 set_capture_region 设置) ──
        self._capture_left: int = 0
        self._capture_top: int = 0
        self._capture_size: int = 640

        # ── 吸附状态机 ──
        self._assist_state: AssistState = AssistState.FREE
        self._locked_target: Optional[np.ndarray] = None  # 当前锁定的目标 bbox
        self._prev_locked_target: Optional[np.ndarray] = None  # override 前的目标 (防立即重吸)

        # ── 人控检测 ──
        self._last_cursor_x: float = 0.0
        self._last_cursor_y: float = 0.0
        self._cursor_initialized: bool = False

        # ── 检测结果队列 ──
        self._detection_queue: queue.Queue = queue.Queue(maxsize=1)
        self._detection_seq: int = 0

        # ── 运动预测 ──
        self._last_target_center: Optional[tuple] = None
        self._last_target_ts: float = 0.0
        self._target_velocity: tuple = (0.0, 0.0)

        self._event_bus.subscribe('detect.result', self._on_detection)
        self._event_bus.subscribe('state.changed', self._on_state_changed)
        self._event_bus.subscribe('config.reloaded', self._on_config_reloaded)
        self._event_bus.subscribe('cmd.aim_on', lambda: setattr(self, '_aim_disabled', False))
        self._event_bus.subscribe('cmd.aim_off', lambda: setattr(self, '_aim_disabled', True))

    def set_capture_region(self, left: int, top: int, size: int):
        self._capture_left = left
        self._capture_top = top
        self._capture_size = size

    # ── 事件处理 ───────────────────────────────────────────

    def _on_detection(self, detections: np.ndarray, **kwargs):
        item = (detections.copy(), kwargs.get('capture_timestamp', 0.0))
        try:
            self._detection_queue.put_nowait(item)
        except queue.Full:
            try:
                self._detection_queue.get_nowait()
            except queue.Empty:
                pass
            self._detection_queue.put_nowait(item)

    def _on_state_changed(self, old_state: AppState, new_state: AppState):
        log.info('aim state: %s -> %s', old_state.name, new_state.name)
        if new_state == AppState.AIMING:
            self._mouse_controller.reset()
            self._assist_state = AssistState.FREE
            self._locked_target = None

    def _on_config_reloaded(self):
        pid = getattr(self._mouse_controller, 'update_gains', None)
        if pid:
            pid(
                kp=self._config.pid_kp,
                ki=self._config.pid_ki,
                kd=self._config.pid_kd,
                max_integral=self._config.pid_max_integral,
                deadband=self._config.pid_deadband,
                kff=self._config.pid_kff,
                noise_amplitude=self._config.noise_amplitude,
                sensitivity=self._config.mouse_sensitivity,
            )

    # ── 坐标转换 ───────────────────────────────────────────

    def _cursor_in_model_space(self) -> tuple[float, float]:
        """将屏幕光标位置转换为模型坐标空间。"""
        pt = wintypes.POINT()
        windll.user32.GetCursorPos(byref(pt))
        scale = self._config.imgsz[0] / max(self._capture_size, 1)
        cx = (pt.x - self._capture_left) * scale
        cy = (pt.y - self._capture_top) * scale
        return cx, cy

    def _get_human_delta(self) -> tuple[float, float]:
        """读取光标位移 (人控鼠标移动量)。"""
        cx, cy = self._cursor_in_model_space()
        if not self._cursor_initialized:
            self._last_cursor_x, self._last_cursor_y = cx, cy
            self._cursor_initialized = True
            return 0.0, 0.0
        dx = cx - self._last_cursor_x
        dy = cy - self._last_cursor_y
        self._last_cursor_x, self._last_cursor_y = cx, cy
        return dx, dy

    # ── 运动预测 ───────────────────────────────────────────

    def _update_motion_model(self, target: np.ndarray, capture_ts: float):
        cx = (target[0] + target[2]) / 2
        cy = (target[1] + target[3]) / 2
        if self._last_target_center is not None and capture_ts > self._last_target_ts:
            dt = capture_ts - self._last_target_ts
            if dt > 0.001:
                vx = (cx - self._last_target_center[0]) / dt
                vy = (cy - self._last_target_center[1]) / dt
                alpha = 0.3
                self._target_velocity = (
                    self._target_velocity[0] * (1 - alpha) + vx * alpha,
                    self._target_velocity[1] * (1 - alpha) + vy * alpha,
                )
        self._last_target_center = (cx, cy)
        self._last_target_ts = capture_ts

    # ── 目标匹配 ───────────────────────────────────────────

    def _find_matching_detection(self, target: np.ndarray, detections: np.ndarray) -> Optional[np.ndarray]:
        """在检测列表中查找与锁定目标 IoU 最高的匹配。"""
        if len(detections) == 0:
            return None
        best_iou = 0.0
        best_det = None
        for det in detections:
            iou = _box_iou(target[:4], det[:4])
            if iou > best_iou:
                best_iou = iou
                best_det = det
        return best_det if best_iou >= 0.3 else None

    def _find_nearest_to_cursor(self, detections: np.ndarray) -> Optional[np.ndarray]:
        """找离光标最近的检测框。"""
        if len(detections) == 0:
            return None
        cx, cy = self._cursor_in_model_space()
        best = None
        best_dist = float('inf')
        for det in detections:
            bx = (det[0] + det[2]) / 2
            by = (det[1] + det[3]) / 2
            dist = hypot(bx - cx, by - cy)
            if det[5] == self._config.enemy_label:
                if dist < best_dist:
                    best_dist = dist
                    best = det
            else:
                # fallback: non-enemy
                if best is None and dist < best_dist:
                    best_dist = dist
                    best = det
        return best

    def _distance_to_bbox_center(self, bbox: np.ndarray) -> float:
        """光标到 bbox 中心的距离 (模型坐标空间)。"""
        cx, cy = self._cursor_in_model_space()
        bx = (bbox[0] + bbox[2]) / 2
        by = (bbox[1] + bbox[3]) / 2
        return hypot(bx - cx, by - cy)

    # ── 磁力吸附 ───────────────────────────────────────────

    def _apply_magnetic_pull(self, target: np.ndarray, human_dx: float, human_dy: float):
        """对锁定目标施加磁力吸附。

        在 ENGAGED 状态下每帧调用。计算光标准星到目标中心的误差,
        输出恒定强度的磁力拉向目标。人控输入存在时,磁力叠加在人控之上。
        """
        magnet = self._config.aim_magnet_strength
        base_speed = self._config.aim_base_speed
        deadband = self._config.pid_deadband

        cx, cy = self._cursor_in_model_space()
        bx = (target[0] + target[2]) / 2
        by = (target[1] + target[3]) / 2
        error_x = bx - cx
        error_y = by - cy
        distance = hypot(error_x, error_y)

        if distance < deadband:
            return  # 已在目标上

        # 磁力强度: 越近越强 (模拟手柄吸附手感)
        # 在 engage_range 边缘: 弱; 在目标附近: 强
        engage_range = self._config.aim_engage_range
        t = 1.0 - min(distance / engage_range, 1.0)  # 0(远)→1(近)
        pull = magnet * (0.3 + 0.7 * t)  # 最低 30% 磁力,最高 100%
        speed = pull * base_speed

        # 方向: 朝向目标中心
        dir_x = error_x / distance
        dir_y = error_y / distance

        dx = dir_x * speed
        dy = dir_y * speed

        # 如果有人控,磁力叠加在人控之上 (人控仍占主导)
        # 人控往目标方向 → 加速, 人控背离目标 → 减弱
        if abs(human_dx) > 0.5 or abs(human_dy) > 0.5:
            human_dir_x = human_dx / max(hypot(human_dx, human_dy), 0.001)
            human_dir_y = human_dy / max(hypot(human_dx, human_dy), 0.001)
            # 人控背离目标时减弱磁力
            alignment = human_dir_x * dir_x + human_dir_y * dir_y  # [-1, 1]
            if alignment < 0:
                dx *= 0.3  # 人控背离,磁力减到 30%
                dy *= 0.3

        self._mouse_controller.move_raw(dx, dy)

    def _is_human_overriding(self, human_dx: float, human_dy: float,
                              target: np.ndarray) -> bool:
        """判断人控是否在主动脱离吸附。"""
        human_dist = hypot(human_dx, human_dy)
        threshold = self._config.aim_override_threshold
        if human_dist < threshold:
            return False

        # 人控方向背离目标 → override
        cx, cy = self._cursor_in_model_space()
        bx = (target[0] + target[2]) / 2
        by = (target[1] + target[3]) / 2
        target_dir_x = bx - cx
        target_dir_y = by - cy
        target_dist = hypot(target_dir_x, target_dir_y)
        if target_dist < 1:
            return False
        target_dir_x /= target_dist
        target_dir_y /= target_dist

        human_dir_x = human_dx / human_dist
        human_dir_y = human_dy / human_dist
        dot = human_dir_x * target_dir_x + human_dir_y * target_dir_y
        return dot < -0.3  # 人控方向与目标方向夹角 > ~108°

    # ── 主循环 ─────────────────────────────────────────────

    def _get_latest_detections(self) -> tuple[Optional[np.ndarray], float]:
        try:
            detections, capture_ts = self._detection_queue.get_nowait()
            self._detection_seq += 1
            return detections, capture_ts
        except queue.Empty:
            return None, 0.0

    def run(self):
        self._should_stop.clear()
        aim_hz = self._config.aim_loop_hz
        interval = 1.0 / aim_hz
        next_frame = time.perf_counter()
        log.info('aim controller (sticky assist) started at %d Hz', aim_hz)

        tune_counter = 0
        tune_window = aim_hz
        detection_count_base = 0
        _diag_interval = max(1, aim_hz)
        _diag_counter = 0
        _move_count = 0

        while not self._should_stop.is_set():
            aim_hz = self._config.aim_loop_hz
            interval = 1.0 / aim_hz

            if self._state_machine.is_aiming() and not self._aim_disabled:
                # ── 初始化光标位置 ──
                human_dx, human_dy = self._get_human_delta()

                # ── 获取最新检测 ──
                detections, capture_ts = self._get_latest_detections()

                # ── 状态机 ──
                if self._assist_state == AssistState.FREE:
                    self._step_free(detections, human_dx, human_dy, capture_ts)

                elif self._assist_state == AssistState.ENGAGED:
                    moved = self._step_engaged(detections, human_dx, human_dy, capture_ts)
                    if moved:
                        _move_count += 1

                elif self._assist_state == AssistState.OVERRIDE:
                    self._step_override(detections)
            else:
                if _diag_counter == 0:
                    if not self._state_machine.is_aiming():
                        log.info('aim: NOT active — state=%s', self._state_machine.state.name)
                    elif self._aim_disabled:
                        log.info('aim: NOT active — disabled by user')

            # ── 动态频率 ──
            tune_counter += 1
            _diag_counter += 1
            if tune_counter >= tune_window:
                detections_in_window = self._detection_seq - detection_count_base
                if detections_in_window > 0:
                    target_hz = max(30, min(240, int(detections_in_window * 1.5)))
                    if target_hz != aim_hz and target_hz != self._config.aim_loop_hz:
                        self._config.aim_loop_hz = target_hz
                        _diag_interval = max(1, target_hz)
                        log.info('aim loop auto-tuned: %d → %d Hz (detection: %d FPS)',
                                 aim_hz, target_hz, detections_in_window)
                detection_count_base = self._detection_seq
                tune_counter = 0
                tune_window = self._config.aim_loop_hz

            # ── 诊断 ──
            if _diag_counter >= _diag_interval:
                active = self._state_machine.is_aiming() and not self._aim_disabled
                if active and _move_count == 0:
                    log.info('aim: %s — %d moves/s (locked=%s)',
                             self._assist_state.name, _move_count,
                             self._locked_target is not None)
                elif active:
                    log.info('aim: %s — %d moves/s (locked=%s)',
                             self._assist_state.name, _move_count,
                             self._locked_target is not None)
                _diag_counter = 0
                _move_count = 0

            # ── 高精度等待 ──
            next_frame += interval
            now = time.perf_counter()
            remaining = next_frame - now
            if remaining > 0.003:
                self._should_stop.wait(max(0, remaining - 0.002))
            if remaining > 0 and not self._should_stop.is_set():
                while time.perf_counter() < next_frame:
                    pass
            elif remaining < -interval:
                next_frame = time.perf_counter()

        log.info('aim controller stopped')

    # ── 状态机步骤 ──────────────────────────────────────────

    def _step_free(self, detections: Optional[np.ndarray],
                   human_dx: float, human_dy: float, capture_ts: float):
        """FREE 状态: 扫描是否有目标进入吸附范围。"""
        if detections is None or len(detections) == 0:
            return

        nearest = self._find_nearest_to_cursor(detections)
        if nearest is None:
            return

        dist = self._distance_to_bbox_center(nearest)
        if dist < self._config.aim_engage_range:
            # 如果之前 override 解锁的同一个目标,不加额外冷却
            if self._prev_locked_target is not None:
                prev_iou = _box_iou(self._prev_locked_target[:4], nearest[:4])
                if prev_iou > 0.5:
                    # 同一个目标,需要冷却: 必须离开再进入
                    return

            self._locked_target = nearest.copy()
            self._assist_state = AssistState.ENGAGED
            self._update_motion_model(nearest, capture_ts)
            log.info('aim: ENGAGED — locked target at dist=%.0fpx', dist)

    def _step_engaged(self, detections: Optional[np.ndarray],
                      human_dx: float, human_dy: float, capture_ts: float) -> bool:
        """ENGAGED 状态: 维持吸附,或检测脱离条件。"""
        # 条件1: 目标消失
        target_gone = True
        if detections is not None and len(detections) > 0 and self._locked_target is not None:
            matched = self._find_matching_detection(self._locked_target, detections)
            if matched is not None:
                target_gone = False
                # 更新锁定位置 (跟随目标移动)
                self._locked_target = matched.copy()
                self._update_motion_model(matched, capture_ts)

        if target_gone:
            log.info('aim: FREE — target lost')
            self._assist_state = AssistState.FREE
            self._prev_locked_target = self._locked_target
            self._locked_target = None
            return False

        # 条件2: 人控脱离
        if self._is_human_overriding(human_dx, human_dy, self._locked_target):
            log.info('aim: OVERRIDE — human took control (delta=%.0fpx)',
                     hypot(human_dx, human_dy))
            self._assist_state = AssistState.OVERRIDE
            self._prev_locked_target = self._locked_target.copy()
            self._locked_target = None
            return False

        # 维持吸附
        self._apply_magnetic_pull(self._locked_target, human_dx, human_dy)
        return True

    def _step_override(self, detections: Optional[np.ndarray]):
        """OVERRIDE 状态: 等人控靠近新目标。"""
        if detections is None or len(detections) == 0:
            return

        nearest = self._find_nearest_to_cursor(detections)
        if nearest is None:
            return

        dist = self._distance_to_bbox_center(nearest)
        if dist < self._config.aim_engage_range:
            # 检查是否是新目标 (不是刚脱离的那个)
            is_new = True
            if self._prev_locked_target is not None:
                iou = _box_iou(self._prev_locked_target[:4], nearest[:4])
                if iou > 0.5:
                    is_new = False

            if is_new:
                log.info('aim: ENGAGED — re-engaged new target at dist=%.0fpx', dist)
                self._locked_target = nearest.copy()
                self._assist_state = AssistState.ENGAGED
                self._prev_locked_target = None

    def stop(self):
        self._should_stop.set()
