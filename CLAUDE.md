# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

wingman_yolo — Apex Legends AI aim-assist tool. YOLO-based object detection → PID-controlled mouse movement. Python 3.10, CUDA 11, torch >= 2.0, ultralytics YOLO + customtkinter GUI.

## Workflow Rule

**Each time a task is assigned, one must first analyze the requirements, provide an implementation plan and key design decisions. After the user agrees to the plan, the coding can be started; direct coding is not allowed.**

## Key Commands

```bash
# Run (must be admin for Dxshot + mouse input)
python main.py

# Train model
python train.py --model yolo11s.pt --epochs 100 --batch 16

# Dependencies
pip install -r requirements.txt
```

## Architecture

### Three parallel threads

```
Capture Thread ──→ Inference Thread ──→ EventBus ──→ Aim Thread (auto-tuned Hz)
(CPU grab)         (GPU YOLO + NMS)     (async)      (PID mouse)
```

- **Capture thread**: `DetectionEngine._capture_loop()` — internal worker that calls `ScreenCapture.grab()`, stamps `time.perf_counter()`, feeds frames to inference via `queue.Queue(maxsize=1)` (latest-only, drops stale frames)
- **Inference thread**: `DetectionEngine.run()` — preprocesses with CUDA Stream (non_blocking H2D), runs YOLO via `AutoBackend`, does GPU NMS via `torchvision.ops.nms`, publishes results via EventBus
- **Aim thread**: `AimController.run()` at variable Hz (auto-tuned to detection_FPS × 1.5, clamped [30, 240]), reads latest detections via `queue.Queue(maxsize=1)`, runs strategy selector, applies motion prediction (velocity extrapolation from capture timestamp with low-pass filter α=0.3), moves mouse via PID controller

### Core modules (`core/`)

| Module | Role |
|--------|------|
| `event_bus.py` | Publish/subscribe pattern — modules never import each other directly |
| `state_machine.py` | States: IDLE → SCANNING → AIMING → SHUTDOWN |
| `config.py` | `AppConfig` dataclass (~30 fields), YAML load/save/reload, hot-reload support |
| `capture.py` | `ScreenCapture` — Dxshot (~5ms) → mss (~12ms) fallback chain with exponential backoff |
| `detector.py` | `DetectionEngine` — model loading (AutoBackend + auto TensorRT export), CUDA Stream pipeline, GPU NMS, internal capture worker thread |
| `strategies.py` | `TargetSelector` ABC + 3 implementations (nearest/largest/crosshair), `MouseController` ABC + `PidMouseController` (PID + feedforward + gain scheduling + noise) + legacy `SmoothAtan2Controller` |
| `aim_controller.py` | `AimController` — target acquisition, motion prediction, dynamic frequency auto-tuning, high-precision timer |
| `overlay.py` | `DetectionOverlay` — Win32 Layered Window (WS_EX_LAYERED\|TRANSPARENT\|TOPMOST\|NOACTIVATE), 32-bit BGRA DIB via numpy, independent render thread with `Queue(maxsize=2)` |
| `display.py` | **[Deprecated]** OpenCV cv2.imshow preview window — replaced by `DetectionOverlay` |

### UI (`ui/`)

| Module | Role |
|--------|------|
| `main_window.py` | customtkinter GUI — status indicator, detection/aim/FPS toggles, strategy selector, capture resolution, reload config, quit. Keyboard listener via pynput (PgUp/PgDn). |

### Mouse driver (`mouse_driver/`)

| Module | Role |
|--------|------|
| `ghub_mouse.py` | G Hub driver (`ghub_mouse.dll`) via ctypes → Win32 `SendInput` fallback |
| `MouseMove.py` | Thin wrapper — `ghub_mouse_move()` and `pygui_mouse_move()` |

### Key patterns

- **EventBus**: Modules talk through events — never direct imports between engines
- **Strategy pattern**: `TargetSelector` base with three implementations, swapped at runtime via config
- **Hot-reload**: Edit `configs/config.yaml` → click "重载配置" → `config.reloaded` event → all modules sync

### EventBus events

| Event | Publisher | Subscribers | Purpose |
|-------|-----------|-------------|---------|
| `cmd.start_detection` | GUI → main | main | Start detection + aim threads |
| `cmd.stop_detection` | GUI → main | main | Stop both threads |
| `cmd.aim_on` | GUI | AimController | Enable aiming |
| `cmd.aim_off` | GUI | AimController | Disable aiming |
| `cmd.shutdown` | GUI → main | main | Clean shutdown |
| `cmd.resize_capture` | GUI → main | main | Resize capture region (IDLE only) |
| `detect.result` | DetectionEngine | AimController | Pass detections with capture timestamp |
| `state.changed` | StateMachine | GUI, AimController | State transition notification |
| `config.reloaded` | GUI (reload btn) | DetectionEngine, AimController | Hot-reload sync |

### Capture fallback chain

```
Dxshot (DXGI Desktop Duplication, ~5ms) → mss (BitBlt C-ext, ~12ms)
```

Dxshot failures trigger exponential backoff (1s → 30s). After repeated failures, permanently falls back to mss silently.

### Configuration

`configs/config.yaml` — most fields hot-reloadable (click "重载配置" in GUI). Fields requiring restart: `weights`, `device`, `imgsz`, `half`, `dnn`, `use_tensorrt`.

`configs/data.yaml` — dataset config: 2 classes (0=person/enemy, 1=else), path `C:/document/datasets/AL_data`.

### Mouse control

- Priority: G Hub driver (`ghub_mouse.dll`) → Win32 `SendInput`
- PID controller defaults: kp=0.35, ki=0.02, kd=0.08, deadband=2px
- Velocity feedforward (`pid_kff`, default 0), gain scheduling by distance zone (far/mid/near), human-like Gaussian noise injection (tracking only, σ=amplitude/3, default amplitude=0.5)
- Legacy atan2 open-loop controller retained

### EAC verification (2026-04-25)

Test results — both mouse paths work on desktop (admin mode required) but are blocked inside Apex Legends:

| Path | Desktop (admin) | In-game | Verdict |
|------|----------------|---------|---------|
| G Hub `gm.moveR()` | ✅ Moves | ❌ Blocked | EAC intercepts G Hub driver calls |
| Win32 `SendInput` | ✅ Moves | ❌ Blocked | EAC intercepts user-mode input APIs |

All user-mode mouse simulation APIs are intercepted by EAC in the game process. A hardware HID device (Arduino Leonardo / RPi Pico) would be required to bypass this.

### Overlay rendering

`DetectionOverlay` creates a fullscreen transparent window that renders detection boxes directly on top of the game via `UpdateLayeredWindow`. Enemies (class 0) drawn in green, non-enemies (class 1) in red. Requires borderless windowed mode — exclusive fullscreen blocks desktop windows.

## Performance bottlenecks (current state)

| Status | Bottleneck |
|--------|------------|
| ✅ Fixed | Serial capture-inference pipeline (now dual-buffered with internal capture thread) |
| ✅ Fixed | CPU NMS (now GPU via torchvision.ops.nms) |
| ✅ Fixed | Missing CUDA Stream (non_blocking H2D + separate stream) |
| ✅ Fixed | Missing motion prediction (velocity extrapolation from timestamps, low-pass filter α=0.3) |
| ✅ Fixed | Frame rate mismatch (auto-tune aim_loop_hz to detection_FPS × 1.5) |
| ✅ Fixed | pyautogui fallback (replaced with mss, ~12ms vs ~50ms) |
| ✅ Fixed | DetectionEngine data race (replaced Lock with queue.Queue) |

## Known limitations

- **EAC blocks all user-mode mouse APIs** — hardware HID device required for in-game use
- **2-class dataset only** — cannot distinguish enemy vs teammate players
- **Exclusive fullscreen incompatible** — overlay requires borderless windowed mode
- **Python 3.10 only** — constrained by CUDA 11 and specific package versions
- **Admin mode required** — mouse input and Dxshot capture need administrator privileges
