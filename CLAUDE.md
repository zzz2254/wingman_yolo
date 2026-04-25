# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

wingman_yolo — Apex Legends AI aim-assist tool. YOLO-based object detection → PID-controlled mouse movement. Python project using ultralytics YOLO + customtkinter GUI.

## Workflow Rule

**每次接受任务时，必须先分析需求，提供实现方案和关键设计决策，在用户同意方案后再动手写代码，不允许直接编写代码。** 

## Key Commands

```bash
# Run (must be admin for Dxshot + mouse input)
python main.py

# Train model
python train.py --model yolo11s.pt --epochs 100 --batch 16

# Dependencies
pip install -r requirements.txt
```

Python 3.10 only, CUDA 11, torch >= 2.0.

## Architecture

### Three parallel threads

```
Capture Thread ──→ Ring buffer ──→ Inference Thread ──→ EventBus ──→ Aim Thread (120Hz)
(CPU grab)        (latest frame)    (GPU + NMS)         (async)      (PID mouse)
```

- **Capture thread**: `ScreenCapture` iterates, stamps `time.perf_counter()`, pushes to detection engine
- **Inference thread**: `DetectionEngine.run()` waits for frames, preprocesses with CUDA Stream, runs YOLO (GPU), does GPU NMS with `torchvision.ops.nms`, publishes results via EventBus
- **Aim thread**: `AimController.run()` at variable Hz (auto-tuned to detection_FPS × 1.5), reads latest detections, runs strategy selector, applies motion prediction (velocity extrapolation from capture timestamp), moves mouse via PID controller

### Core modules (`core/`)

| Module | Role |
|--------|------|
| `event_bus.py` | Publish/subscribe pattern — modules never import each other directly |
| `state_machine.py` | States: IDLE → SCANNING → AIMING → SHUTDOWN |
| `config.py` | `AppConfig` dataclass, supports YAML hot-reload |
| `capture.py` | `ScreenCapture` — Dxshot → mss fallback chain |
| `detector.py` | `DetectionEngine` — model loading, CUDA Stream pipeline, GPU NMS |
| `strategies.py` | `StrategyRouter` (nearest/largest/crosshair) + `PidMouseController` |
| `aim_controller.py` | `AimController` — target selection, motion prediction, dynamic frequency |
| `display.py` | `Display` — independent thread cv2.imshow + FPS overlay |

### Key patterns

- **EventBus**: Modules talk through events (`detect.result`, `cmd.start_detection`, `config.reloaded`, etc.) — never direct imports between engines
- **Strategy pattern**: `TargetSelector` base with three implementations, swapped at runtime via config
- **Hot-reload**: Edit `configs/config.yaml` → click "重载配置" → `config.reloaded` event → all modules sync

### Capture fallback chain

```
Dxshot (DXGI, ~3ms) → mss (BitBlt C-ext, ~12ms)
```

### Configuration

`configs/config.yaml` — most fields hot-reloadable. Fields requiring restart: `weights`, `device`, `imgsz`, `half`, `dnn`, `use_tensorrt`.

### Mouse control

- Priority: G Hub driver (`ghub_mouse.dll`) → Win32 `SendInput`
- PID controller (default: kp=0.35, ki=0.02, kd=0.08, deadband=2px)

### EAC verification (2026-04-25)

Test results — both mouse paths work on desktop (admin mode required) but are blocked inside Apex Legends:

| Path | Desktop (admin) | In-game | Verdict |
|------|----------------|---------|---------|
| G Hub `gm.moveR()` | ✅ Moves | ❌ Blocked | EAC intercepts G Hub driver calls |
| Win32 `SendInput` | ✅ Moves | ❌ Blocked | EAC intercepts user-mode input APIs |

All user-mode mouse simulation APIs are intercepted by EAC in the game process. A hardware HID device (Arduino Leonardo / RPi Pico) would be required to bypass this.

## Performance bottlenecks (current state)

| Status | Bottleneck |
|--------|------------|
| ✅ Fixed | Serial capture-inference pipeline (double-buffered) |
| ✅ Fixed | CPU NMS (now GPU via torchvision.ops.nms) |
| ✅ Fixed | Missing CUDA Stream (non_blocking H2D + separate stream) |
| ✅ Fixed | Missing motion prediction (velocity extrapolation from timestamps) |
| ✅ Fixed | Frame rate mismatch (auto-tune aim_loop_hz to detection_FPS × 1.5) |
| ✅ Fixed | pyautogui fallback (replaced with mss, ~12ms vs ~50ms) |
| ✅ Fixed | DetectionEngine data race (replaced Lock with queue.Queue) |
