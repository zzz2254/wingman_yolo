import logging
import threading
import time
from typing import Optional

import numpy as np
from pynput.mouse import Button, Listener

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

        self._latest_detections: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._mouse_pressed = False
        self._should_stop = threading.Event()

        self._mouse_listener = Listener(on_click=self._on_click)
        self._mouse_listener.start()

        self._event_bus.subscribe('detect.result', self._on_detection)
        self._event_bus.subscribe('state.changed', self._on_state_changed)

    def _on_detection(self, detections: np.ndarray):
        with self._lock:
            self._latest_detections = detections

    def _on_state_changed(self, old_state: AppState, new_state: AppState):
        log.debug('state: %s -> %s', old_state.name, new_state.name)

    def _on_click(self, x, y, button, pressed):
        if not self._state_machine.is_aiming():
            return
        if button in (Button.left, Button.right):
            self._mouse_pressed = pressed
            log.info('mouse lock %s', 'enabled' if pressed else 'disabled')

    def _acquire_target(self) -> Optional[np.ndarray]:
        detections = self._latest_detections
        if detections is None or len(detections) == 0:
            return None
        return self._target_selector.select(
            detections, self._config.imgsz, self._config.enemy_label,
        )

    def run(self):
        self._should_stop.clear()
        interval = 1.0 / self._config.aim_loop_hz
        log.info('aim controller started at %d Hz', self._config.aim_loop_hz)

        while not self._should_stop.is_set():
            if self._state_machine.is_aiming() and self._mouse_pressed:
                target = self._acquire_target()
                if target is not None and int(target[5]) == self._config.enemy_label:
                    self._mouse_controller.move(
                        target, self._config.imgsz, self._config.smooth_factor,
                    )
            time.sleep(interval)

        log.info('aim controller stopped')

    def stop(self):
        self._should_stop.set()
