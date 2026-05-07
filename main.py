import logging
import logging.handlers
import threading
from pathlib import Path

import yaml

from mouse_driver.MouseMove import ghub_mouse_move

from core.capture import ScreenCapture
from core.config import AppConfig
from core.detector import DetectionEngine
from core.event_bus import EventBus
from core.state_machine import AppState, StateMachine
from core.strategies import (
    PidMouseController,
    StrategyRouter,
)
from core.aim_controller import AimController
from ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
_log_root = logging.getLogger()
_log_dir = Path(__file__).resolve().parent / 'logs'
_log_dir.mkdir(exist_ok=True)
_file_handler = logging.handlers.RotatingFileHandler(
    _log_dir / 'wingman_yolo.log', maxBytes=5 * 1024 * 1024, backupCount=3,
    encoding='utf-8',
)
_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
))
_log_root.addHandler(_file_handler)

log = logging.getLogger(__name__)
log.info('log file: %s', _log_dir / 'wingman_yolo.log')


def main():
    config = AppConfig(
        weights='./runs/detect/train-4/weights/best.pt',
        data='./configs/data.yaml',
        device='0',
    )
    # 从 YAML 加载热加载配置（覆盖上述默认值）
    cfg_yaml = './configs/config.yaml'
    if Path(cfg_yaml).exists():
        try:
            raw = yaml.safe_load(Path(cfg_yaml).read_text(encoding='utf-8'))
            for k, v in raw.items():
                if k in AppConfig.__dataclass_fields__:
                    setattr(config, k, tuple(v) if k == 'imgsz' else v)
            config._config_path = cfg_yaml
            log.info('loaded config from %s', cfg_yaml)
        except Exception:
            log.exception('failed to load config yaml')

    event_bus = EventBus()
    state_machine = StateMachine(event_bus)

    capture = ScreenCapture(capture_size=config.capture_size)
    engine = DetectionEngine(config, event_bus)

    target_selector = StrategyRouter(config)
    mouse_controller = PidMouseController(
        ghub_mouse_move,
        kp=config.pid_kp,
        ki=config.pid_ki,
        kd=config.pid_kd,
        max_integral=config.pid_max_integral,
        deadband=config.pid_deadband,
        kff=config.pid_kff,
        noise_amplitude=config.noise_amplitude,
        sensitivity=config.mouse_sensitivity,
    )
    aim_ctl = AimController(
        config, event_bus, state_machine, target_selector, mouse_controller,
    )
    aim_ctl.set_capture_region(capture.capture_left, capture.capture_top, config.capture_size)

    # 标定鼠标输入单位→像素比率
    mouse_controller.calibrate()

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

    def _on_resize_capture(**kw):
        new_size = kw.get('size', 640)
        if state_machine.state != AppState.IDLE:
            log.warning('cannot resize capture while detection is running')
            return
        capture.resize(new_size)

    event_bus.subscribe('cmd.resize_capture', _on_resize_capture)

    def _on_shutdown():
        engine.stop()
        aim_ctl.stop()
        log.info('application shutting down')

    event_bus.subscribe('cmd.shutdown', _on_shutdown)

    event_bus.subscribe('state.changed', lambda **kw: print(
        f'[状态] {kw["old_state"].name} -> {kw["new_state"].name}',
    ))

    log.info('Wingman_Yolo starting')
    window.run()


if __name__ == '__main__':
    main()
