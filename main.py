import logging
import threading
import sys

from mouse_driver.MouseMove import ghub_mouse_move

from core.capture import ScreenCapture
from core.config import AppConfig
from core.detector import DetectionEngine
from core.event_bus import EventBus
from core.state_machine import AppState, StateMachine
from core.strategies import (
    NearestEnemySelector,
    SmoothAtan2Controller,
)
from core.aim_controller import AimController
from ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def main():
    config = AppConfig(
        weights='./weights/best.pt',
        data='./configs/AL_data.yaml',
        device='0',
    )

    event_bus = EventBus()
    state_machine = StateMachine(event_bus)

    capture = ScreenCapture(capture_size=config.capture_size)
    engine = DetectionEngine(config, event_bus)

    target_selector = NearestEnemySelector()
    mouse_controller = SmoothAtan2Controller(ghub_mouse_move)
    aim_ctl = AimController(
        config, event_bus, state_machine, target_selector, mouse_controller,
    )

    window = MainWindow(config, event_bus, state_machine)

    def _on_start_detection():
        engine_thread = threading.Thread(
            target=engine.run, args=(capture,), daemon=True,
        )
        engine_thread.start()
        aim_thread = threading.Thread(
            target=aim_ctl.run, daemon=True,
        )
        aim_thread.start()
        log.info('detection and aim control started')

    def _on_stop_detection():
        engine.stop()
        aim_ctl.stop()
        log.info('detection and aim control stopped')

    event_bus.subscribe('cmd.start_detection', _on_start_detection)
    event_bus.subscribe('cmd.stop_detection', _on_stop_detection)

    def _on_shutdown():
        engine.stop()
        aim_ctl.stop()
        log.info('application shutting down')

    event_bus.subscribe('cmd.shutdown', _on_shutdown)

    event_bus.subscribe('state.changed', lambda **kw: print(
        f'[状态] {kw["old_state"].name} -> {kw["new_state"].name}',
    ))

    log.info('AL_Yolo starting')
    window.run()


if __name__ == '__main__':
    main()
