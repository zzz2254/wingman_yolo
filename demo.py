"""Demo mode: screen-capture YOLO detection + PID auto-aim, no GUI.

Usage:
    1. Open a video in PotPlayer / browser, position the window near screen center.
    2. Run as admin:
       python demo.py
    3. Press Ctrl+C to exit.

The demo reuses the full production pipeline (ScreenCapture, DetectionEngine,
AimController, PID, DetectionOverlay) but replaces the GUI state machine
control with a direct IDLE→SCANNING→AIMING transition on startup.
"""

import logging
import threading
from pathlib import Path

import yaml

from core.capture import ScreenCapture
from core.config import AppConfig
from core.detector import DetectionEngine
from core.event_bus import EventBus
from core.state_machine import AppState, StateMachine
from core.strategies import PidMouseController, StrategyRouter
from core.aim_controller import AimController
from mouse_driver.MouseMove import ghub_mouse_move

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    config = AppConfig(
        weights="./runs/detect/train-4/weights/best.pt",
        data="./configs/data.yaml",
        device="0",
    )
    cfg_yaml = "./configs/config.yaml"
    if Path(cfg_yaml).exists():
        try:
            raw = yaml.safe_load(Path(cfg_yaml).read_text(encoding="utf-8"))
            for k, v in raw.items():
                if k in AppConfig.__dataclass_fields__:
                    setattr(config, k, tuple(v) if k == "imgsz" else v)
            config._config_path = cfg_yaml
            log.info("loaded config from %s", cfg_yaml)
        except Exception:
            log.exception("failed to load config yaml")

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

    # ── Start detection + aim threads (same wiring as main.py) ──
    def _on_start_detection():
        engine_thread = threading.Thread(
            target=engine.run, args=(capture,), daemon=True,
        )
        engine_thread.start()
        aim_thread = threading.Thread(
            target=aim_ctl.run, daemon=True,
        )
        aim_thread.start()
        log.info("detection and aim control started")

    def _on_stop_detection():
        engine.stop()
        aim_ctl.stop()
        log.info("detection and aim control stopped")

    event_bus.subscribe("cmd.start_detection", _on_start_detection)
    event_bus.subscribe("cmd.stop_detection", _on_stop_detection)

    event_bus.subscribe("state.changed", lambda **kw: print(
        f"[State] {kw['old_state'].name} -> {kw['new_state'].name}",
    ))

    # ── Enter AIMING directly ──
    state_machine.transition(AppState.SCANNING)
    event_bus.publish("cmd.start_detection")
    state_machine.transition(AppState.AIMING)
    event_bus.publish("cmd.aim_on")

    log.info("Demo running — Ctrl+C to exit")

    # ── Shutdown handling ──
    import sys
    shutdown_event = threading.Event()
    _shutting_down = False
    _shutdown_lock = threading.Lock()

    def _do_shutdown():
        nonlocal _shutting_down
        with _shutdown_lock:
            if _shutting_down:
                return
            _shutting_down = True
        log.info("shutting down...")
        event_bus.publish("cmd.stop_detection")
        state_machine.transition(AppState.IDLE)
        state_machine.transition(AppState.SHUTDOWN)
        shutdown_event.set()

    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=0.5)
    except KeyboardInterrupt:
        _do_shutdown()

    log.info("demo exited")
    sys.exit(0)


if __name__ == "__main__":
    main()
