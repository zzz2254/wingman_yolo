import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

from core.config import AppConfig
from core.event_bus import EventBus
from core.state_machine import AppState, StateMachine
from core.strategies import MouseController, TargetSelector

log = logging.getLogger(__name__)


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

        # ── 检测结果追踪 ──
        self._detection_queue: queue.Queue = queue.Queue(maxsize=1)
        self._detection_seq: int = 0       # aim 线程独有,无锁安全
        self._cached_target: Optional[np.ndarray] = None  # 缓存的原始目标框
        self._cached_capture_ts: float = 0.0

        # ── 运动预测（瓶颈3: target leading） ──
        self._last_target_center: Optional[tuple] = None  # (cx, cy)
        self._last_target_ts: float = 0.0
        self._target_velocity: tuple = (0.0, 0.0)  # (vx, vy) px/s

        self._event_bus.subscribe('detect.result', self._on_detection)
        self._event_bus.subscribe('state.changed', self._on_state_changed)
        self._event_bus.subscribe('config.reloaded', self._on_config_reloaded)
        self._event_bus.subscribe('cmd.aim_on', lambda: setattr(self, '_aim_disabled', False))
        self._event_bus.subscribe('cmd.aim_off', lambda: setattr(self, '_aim_disabled', True))

    def _on_detection(self, detections: np.ndarray, **kwargs):
        capture_ts = kwargs.get('capture_timestamp', 0.0)

        # 不管鼠标是否按下，只要找到有效目标就更新缓存
        target = self._target_selector.select(
            detections, self._config.imgsz, self._config.enemy_label,
        )
        if target is not None:
            self._cached_target = target.copy()
            self._cached_capture_ts = capture_ts

        item = (detections.copy(), capture_ts)
        try:
            self._detection_queue.put_nowait(item)
        except queue.Full:
            # 队列满 → 丢弃旧帧保留最新
            try:
                self._detection_queue.get_nowait()
            except queue.Empty:
                pass
            self._detection_queue.put_nowait(item)

    def _on_state_changed(self, old_state: AppState, new_state: AppState):
        log.info('aim state: %s -> %s', old_state.name, new_state.name)
        if new_state == AppState.AIMING:
            self._mouse_controller.reset()

    def _on_config_reloaded(self):
        """配置热加载后，同步 PID 增益到 mouse_controller。"""
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
        # 目标选择策略由 StrategyRouter 每次读取 self._config.target_strategy,无需额外操作

    # ── 运动预测 ─────────────────────────────────────────────

    def _update_motion_model(self, target: np.ndarray, capture_ts: float):
        """根据目标位移更新速度模型（低通滤波）。"""
        cx = (target[0] + target[2]) / 2
        cy = (target[1] + target[3]) / 2

        if self._last_target_center is not None and capture_ts > self._last_target_ts:
            dt = capture_ts - self._last_target_ts
            if dt > 0.001:
                vx = (cx - self._last_target_center[0]) / dt
                vy = (cy - self._last_target_center[1]) / dt
                # 低通滤波: 平滑速度估计,防止单帧抖动
                alpha = 0.3
                self._target_velocity = (
                    self._target_velocity[0] * (1 - alpha) + vx * alpha,
                    self._target_velocity[1] * (1 - alpha) + vy * alpha,
                )

        self._last_target_center = (cx, cy)
        self._last_target_ts = capture_ts

    def _predict_target(self, target: np.ndarray, capture_ts: float) -> np.ndarray:
        """根据数据年龄外推目标当前位置（target leading）。"""
        if capture_ts <= 0 or all(v == 0.0 for v in self._target_velocity):
            return target

        dt = time.perf_counter() - capture_ts
        if dt < 0.002:
            return target  # 足够新鲜,无需预测

        dt = min(dt, 0.1)  # 最多外推 100ms

        cx = (target[0] + target[2]) / 2
        cy = (target[1] + target[3]) / 2
        half_w = (target[2] - target[0]) / 2
        half_h = (target[3] - target[1]) / 2

        pred_cx = cx + self._target_velocity[0] * dt
        pred_cy = cy + self._target_velocity[1] * dt

        return np.array([
            pred_cx - half_w, pred_cy - half_h,
            pred_cx + half_w, pred_cy + half_h,
            target[4], target[5],
        ], dtype=target.dtype)

    def _acquire_target(self) -> Optional[np.ndarray]:
        try:
            detections, capture_ts = self._detection_queue.get_nowait()
            self._detection_seq += 1
        except queue.Empty:
            if self._cached_target is not None:
                predicted = self._predict_target(self._cached_target, self._cached_capture_ts)
                log.debug('aim: using cached target (no new detection)')
                return predicted
            log.debug('aim: no target (queue empty, no cache)')
            return None

        # 新检测 → 选择新目标
        n_raw = len(detections)
        target = self._target_selector.select(
            detections, self._config.imgsz, self._config.enemy_label,
        )

        if target is not None:
            self._update_motion_model(target, capture_ts)
            predicted = self._predict_target(target, capture_ts)
            self._cached_target = target.copy()
            self._cached_capture_ts = capture_ts
            log.debug('aim: acquired target pos=(%.0f,%.0f) cls=%d conf=%.2f among %d dets',
                       (target[0] + target[2]) / 2, (target[1] + target[3]) / 2,
                       int(target[5]), target[4], n_raw)
            return predicted

        # 新帧无有效目标 → 回退到缓存（不超过 500ms 老化）
        if self._cached_target is not None:
            age = time.perf_counter() - self._cached_capture_ts
            if age < 0.5:
                predicted = self._predict_target(self._cached_target, self._cached_capture_ts)
                log.debug('aim: using cached target (no valid det in frame, age=%.0fms)', age * 1000)
                return predicted
            else:
                self._cached_target = None
                log.debug('aim: cache expired (age=%.0fms > 500ms)', age * 1000)

        log.debug('aim: no enemy in %d detections (label=%d)', n_raw, self._config.enemy_label)
        self._last_target_center = None
        return None

    def run(self):
        self._should_stop.clear()
        aim_hz = self._config.aim_loop_hz
        interval = 1.0 / aim_hz
        next_frame = time.perf_counter()
        log.info('aim controller started at %d Hz', aim_hz)

        # 动态频率追踪
        tune_counter = 0
        tune_window = aim_hz  # 每秒统计一次
        detection_count_base = 0

        # 诊断计数器
        _diag_interval = max(1, aim_hz)  # 每秒输出一次诊断
        _diag_counter = 0
        _move_count = 0  # 本周期内鼠标移动次数

        while not self._should_stop.is_set():
            # 支持热加载
            aim_hz = self._config.aim_loop_hz
            interval = 1.0 / aim_hz

            if self._state_machine.is_aiming() and not self._aim_disabled:
                target = self._acquire_target()
                if target is not None:
                    if hasattr(self._mouse_controller, 'set_target_velocity'):
                        self._mouse_controller.set_target_velocity(
                            self._target_velocity[0], self._target_velocity[1],
                        )
                    self._mouse_controller.move(
                        target, self._config.imgsz, self._config.smooth_factor,
                    )
                    _move_count += 1
            else:
                # 诊断: 为什么没瞄准（每秒输出一次）
                if _diag_counter == 0:
                    if not self._state_machine.is_aiming():
                        log.info('aim: NOT active — state=%s (need AIMING)',
                                  self._state_machine.state.name)
                    elif self._aim_disabled:
                        log.info('aim: NOT active — aim disabled by user (state=AIMING)')

            # ── 动态频率适配（瓶颈2） ────────────────────
            tune_counter += 1
            _diag_counter += 1
            if tune_counter >= tune_window:
                detections_in_window = self._detection_seq - detection_count_base
                if detections_in_window > 0:
                    # 目标频率 = 检测帧率 × 1.5, 保证有足够新鲜数据供 PID 消费
                    target_hz = max(30, min(240, int(detections_in_window * 1.5)))
                    if target_hz != aim_hz and target_hz != self._config.aim_loop_hz:
                        self._config.aim_loop_hz = target_hz
                        _diag_interval = max(1, target_hz)
                        log.info(
                            'aim loop auto-tuned: %d → %d Hz (detection: %d FPS)',
                            aim_hz, target_hz, detections_in_window,
                        )
                detection_count_base = self._detection_seq
                tune_counter = 0
                tune_window = self._config.aim_loop_hz

            # ── 诊断：每秒输出瞄准统计 ──────────────────
            if _diag_counter >= _diag_interval:
                is_aiming = self._state_machine.is_aiming()
                has_cache = self._cached_target is not None
                active = is_aiming and not self._aim_disabled
                if active and _move_count == 0:
                    log.info('aim: ACTIVE but 0 moves/s (is_aiming=%s cache=%s)',
                              is_aiming, has_cache)
                elif active:
                    log.info('aim: OK — %d moves/s (cache=%s vel=(%.0f,%.0f))',
                              _move_count, has_cache,
                              self._target_velocity[0], self._target_velocity[1])
                _diag_counter = 0
                _move_count = 0

            # ── 高精度等待 ──────────────────────────────
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

    def stop(self):
        self._should_stop.set()
