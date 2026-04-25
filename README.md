# AL_Yolo

基于 YOLO 的 Apex Legends AI 辅瞄工具。

## 环境依赖

- 罗技驱动（版本不超过 21.9，可选）
- Python >= 3.10 && < 3.11
- CUDA 11
- torch >= 2.0
- 更多依赖见 `requirements.txt`

## 快速开始

> 默认在 Windows 系统下

#### 配置 scoop（Windows 包管理器）

打开 PowerShell（version 5.1 or later）

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
```

#### 配置 anaconda

```
scoop bucket add extras
scoop install anaconda3
```

#### 创建 conda 环境并安装依赖

```
conda create -n apex python=3.10
conda activate apex
pip install -r requirements.txt
```

#### 配置 CUDA、cuDNN、PyTorch

GPU 推理所必须，请自行搜索配置教程。

#### 安装罗技驱动（可选）

项目提供一份 2021.9 版本（`lghub_installer*`），仅供学习。不安装时会自动回退到 Win32 `SendInput`。

#### 运行

管理员模式打开终端：

```
python main.py
```

## 项目架构

```
main.py                  # 入口：组装模块，启动检测 + 瞄准双线程
├── core/
│   ├── config.py        # AppConfig 配置数据类（支持 YAML 热加载）
│   ├── event_bus.py     # 发布/订阅事件总线（模块解耦）
│   ├── state_machine.py # 有限状态机（IDLE → SCANNING → AIMING → SHUTDOWN）
│   ├── capture.py       # 屏幕捕获（Dxshot + pyautogui 回退）
│   ├── detector.py      # YOLO 检测引擎（AutoBackend + TensorRT 自动导出）
│   ├── strategies.py    # 目标选择策略 + PID/atan2 鼠标控制
│   ├── aim_controller.py# 瞄准控制主循环（120Hz 高精度定时器）
│   └── display.py       # 独立线程的 OpenCV 预览窗口
├── ui/
│   └── main_window.py   # Apple 风格 GUI（customtkinter）
├── mouse_driver/
│   ├── ghub_mouse.py    # 罗技 G Hub 驱动 / Win32 SendInput 回退
│   └── MouseMove.py     # 鼠标移动封装层
├── train.py             # YOLO 模型训练
├── weights/             # 模型权重文件
├── configs/             # YAML 配置文件
├── logs/                # 日志输出（自动轮转，单文件 5MB，保留 3 份）
└── runs/                # 训练输出
```

## 全流程管线

### 状态机

```
IDLE ──[GUI 开启检测]──→ SCANNING ──[PgUp / GUI 开启瞄准]──→ AIMING
                           ↑                                       │
                           └────────────[PgDn / GUI 关闭瞄准]───────┘
任何状态 ──[关闭程序]──→ SHUTDOWN
```

- **IDLE**: 就绪状态，可修改截屏分辨率
- **SCANNING**: 检测引擎运行中，显示检测框但不移动鼠标
- **AIMING**: 检测 + 自动瞄准均激活，按住鼠标键时移动准星
- **SHUTDOWN**: 停止所有线程，清理资源

### 数据流管线（逐帧循环）

```
┌─────────────────────────────────────────────────────────────────────┐
│                     检测线程 (DetectionEngine)                       │
│                                                                     │
│  ScreenCapture.grab()                                               │
│    ├─ 优先 Dxshot (DXGI Desktop Duplication, ~5ms 延迟)             │
│    └─ 失败回退 pyautogui.screenshot()                               │
│         ↓                                                           │
│  截取屏幕中央 640×640 区域，输出 (CHW tensor, HWC ndarray)          │
│         ↓                                                           │
│  DetectionEngine.infer()                                            │
│    ├─ TensorRT 自动导出 (.pt → .engine, FP16 加速)                  │
│    ├─ 预处理: uint8 → float, /255, unsqueeze batch dim              │
│    ├─ AutoBackend 推理 (ultralytics)                                │
│    └─ NMS 后处理 (conf_thres=0.45, iou_thres=0.45)                 │
│         ↓                                                           │
│  publish('detect.result', detections=...)                           │
│         ↓                                                           │
│  Display 独立线程绘制检测框 (绿=敌人, 蓝=友军, ESC 退出)            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (EventBus 异步传递引用)
┌─────────────────────────────────────────────────────────────────────┐
│                     瞄准线程 (AimController, 120Hz)                  │
│                                                                     │
│  读取最新检测结果                                                     │
│         ↓                                                           │
│  if state == AIMING && mouse_pressed:                               │
│         ↓                                                           │
│  StrategyRouter.select(target_strategy 可运行时切换)                  │
│    ├─ 'nearest'   → NearestEnemySelector   (离屏幕中心最近)          │
│    ├─ 'largest'   → LargestEnemySelector    (框面积最大≈威胁最大)   │
│    └─ 'crosshair' → CrosshairProximitySelector (离鼠标光标最近)      │
│         ↓                                                           │
│  PidMouseController.move()                                          │
│    ├─ 计算误差: 目标中心 - 画面中心                                   │
│    ├─ Deadband 检查: 误差 < 2px → 跳过 (防像素级抖动)                │
│    ├─ PID 计算: P(比例) + I(积分, 限幅30) + D(微分)                 │
│    └─ 输出 (dx, dy) → ghub_mouse.moveR() / SendInput                │
│                                                                     │
│  定时器: sleep 大部分时间 + 2ms 自旋校准 (跑满设定 Hz)               │
└─────────────────────────────────────────────────────────────────────┘
```

### 事件总线 (EventBus)

模块间通过事件完全解耦：

| 事件 | 发布者 | 订阅者 | 说明 |
|------|--------|--------|------|
| `cmd.start_detection` | GUI → main | main | 启动检测+瞄准线程 |
| `cmd.stop_detection` | GUI → main | main | 停止双线程 |
| `cmd.aim_on / aim_off` | GUI | — | 瞄准开关（保留） |
| `cmd.shutdown` | GUI → main | main | 清理资源 |
| `cmd.resize_capture` | GUI → main | main | 重置截屏尺寸 |
| `detect.result` | DetectionEngine | AimController | 传递检测结果 |
| `state.changed` | StateMachine | GUI / AimController | 状态刷新 |
| `config.reloaded` | GUI | Engine / AimController | 热加载同步 |

## 配置热加载

编辑 `configs/config.yaml` 后点击 GUI 中的 **重载配置** 按钮即时生效，无需重启。

**支持热加载：**
`conf_thres` / `iou_thres` / `max_det` / `agnostic_nms` / `enemy_label` /
`target_strategy` / `capture_size` / `pid_kp` / `pid_ki` / `pid_kd` / `pid_max_integral` / `pid_deadband` /
`view_img` / `show_fps` / `aim_loop_hz`

**不支持热加载（需重启程序）：**
`weights` — 模型已加载到显存
`device` — GPU 设备切换
`data` — 数据集配置仅在模型加载时读取
`imgsz` — 模型输入 shape 已固定
`half` — FP16/FP32 需重建推理引擎
`dnn` / `use_tensorrt` — 推理后端切换

## 模型训练

#### 数据集

[https://github.com/goldjee/AL-YOLO-dataset](https://github.com/goldjee/AL-YOLO-dataset)

#### 训练

```
python train.py --model yolo11s.pt --epochs 100 --batch 16
```

支持 yolo11n/s/m/l/x 等 ultralytics 系列模型。

## 性能瓶颈分析

### 当前管线全景

系统分为两个并行线程：**检测线程**（捕获 → 推理 → 显示）和**瞄准线程**（120Hz 目标选择 → PID 控制）。检测线程是吞吐瓶颈的决定者，瞄准线程的 120Hz 主要靠 PID 的轻量计算维持。

### 瓶颈排序（由重到轻）

#### 1. 串行的捕获-推理管线（首要瓶颈）

检测线程内部是**严格串行**的：`grab()` → `infer()` → `show()` → `grab()` → ...。GPU 推理期间 CPU 完全空闲等待，CPU 捕获期间 GPU 完全空闲。现代 GPU 推理（TensorRT FP16）可能仅需 5-8ms，但加上捕获、预处理、NMS、显示，单帧总延迟约 12-20ms，对应 **50-80 FPS**。这一帧率直接决定了瞄准系统可用的检测数据新鲜度。

**改进方向**: 引入双缓冲/流水线并行——GPU 推理第 N 帧的同时，CPU 捕获第 N+1 帧。可将有效 FPS 提升接近纯推理速度。

#### 2. 推理帧率与瞄准帧率不匹配

AimController 以 120Hz 运行，但检测结果更新速度可能只有 50-80 FPS。这意味着每帧检测结果被重复消费 1.5-2.4 次，PID 控制器在大部分周期内处理的是与上一周期相同的数据。虽然不是功能性缺陷，但 120Hz 的循环频率有一半以上的周期在做"无效"迭代。

**改进方向**: 降低 `aim_loop_hz` 到检测帧率的 1.5 倍左右（如 90Hz），或让 AimController 根据检测到达时间动态调整。

#### 3. 检测结果缺乏时间戳

检测结果仅传递裸 ndarray，不附带捕获时间戳。AimController 无法判断当前检测数据有多旧（从捕获到被消费之间可能累积 1-2 帧的延迟），也无法做**运动预测**（目标位置外推）。对于快速移动的目标（Apex Legends 中很常见），运动预测能显著提升跟枪精度。

**改进方向**: 检测结果附带 `capture_timestamp`，AimController 消费时加上运动补偿。

#### 4. pyautogui 回退时的捕获延迟

Dxshot 的延迟约 3-5ms，但在某些 DXGI 不支持的环境（远程桌面、特定 GPU 驱动）会回退到 `pyautogui.screenshot()`。`pyautogui.screenshot()` 底层调用 `PIL.ImageGrab.grab()`，延迟约 30-100ms，直接导致 FPS 跌至 10-30。系统此时虽然不会崩溃，但辅助瞄准的实用价值大幅下降。

**改进方向**: 增加 `mss` (Multi-Screen Shot) 库作为第二回退选项（比 pyautogui 快 3-5 倍），或使用 `win32gui` / DirectX 直接抓取。

#### 5. 缺少异步 CUDA 流

当前推理使用 `torch.no_grad()` 默认同步流，CPU 端 `infer()` 调用会阻塞直到 GPU 完成。虽然 TensorRT 已经很快，但异步流可以让 CPU 在 GPU 工作时提前准备下一帧数据。

**改进方向**: 使用 CUDA Stream 分离预处理（CPU）和推理（GPU），减小 CPU 空闲时间。

#### 6. NMS 后处理在 CPU 上执行

`non_max_suppression` 默认在 CPU 上运行（`pred[0].cpu().numpy()`），对于 300 个候选框是毫秒级操作，但在 FPS 敏感场景下仍是一个优化点。TensorRT 本身不输出 NMS 结果，需在外部做。

**改进方向**: 当检测目标数量稳定较少时，可考虑 `torchvision.ops.nms`（GPU 端 NMS），减少 CPU-GPU 同步点。

#### 7. 检测结果的线程安全读取

`AimController._on_detection` 写入时持有锁，但 `_acquire_target` 读取 `self._latest_detections` 时**未加锁**。虽然 CPython 的 GIL 使得 ndarray 引用的赋值是原子的（实际运行不会 crash），但从代码正确性角度这是一个 data race——在极低概率下可能读到中间状态的引用。

**改进方向**: 使用 `copy` 或 `queue.Queue` 明确所有权转移，消除隐式共享。

### 总结

| 瓶颈 | 影响 | 优先级 |
|------|------|--------|
| 串行捕获-推理管线 | 有效 FPS 受限于串行延迟 | 🔴 高 |
| 帧率不匹配 (120Hz vs ~60FPS) | 约一半循环处理过期数据 | 🟡 中 |
| 检测缺少时间戳 | 无法做运动预测 | 🟡 中 |
| pyautogui 回退 | 极端场景 FPS 暴跌 | 🟡 中 |
| 无 CUDA Stream | CPU 空闲等待 | 🟢 低 |
| CPU 端 NMS | 微秒级开销 | 🟢 低 |
| 检测读取 data race | 代码正确性 | 🟢 低 |

**最大收益的优化**: 将捕获-推理管线改为双缓冲流水线，可在不改动模型和硬件的前提下提升有效 FPS 约 30-50%。其次是添加捕获时间戳和目标运动预测，提升瞄准精度。

## 屏幕捕获

- 优先使用 **Dxshot**（DXGI Desktop Duplication），延迟 ~5ms
- 自动回退到 `pyautogui.screenshot`
- 截取屏幕中央区域（1080p=640px, 2K=960px, 4K=1280px）

## 鼠标控制

- 优先使用 **罗技 G Hub 驱动**（`ghub_mouse.dll`），更底层不易被游戏屏蔽
- 驱动不可用时自动回退到 Win32 `SendInput`
- PID 闭环控制器（默认）：kp=0.35, ki=0.02, kd=0.08, deadband=2px
- 后备 atan2 开环控制器（保留未删除）

## 后续改进

**已完成**
- [x] 截图方式优化（Dxshot）
- [x] 添加自瞄开关（PgUp/PgDn 控制）
- [x] 取消对驱动的依赖（回退 SendInput）
- [x] 多目标识别优先级判断
- [x] 项目架构优化（EventBus + StateMachine + 策略模式）
- [x] 对不同机器参数自适应
- [x] Apple 风格 GUI（customtkinter）
- [x] 支持更多 YOLO 系列模型（v5/v8/v11 统一 API）
- [x] PID 闭环鼠标控制
- [x] TensorRT 推理加速（自动 .pt → .engine 导出）
- [x] 高精度瞄准定时器（sleep + 2ms 自旋校准）
- [x] 分离显示线程
- [x] 多目标优先级策略扩展（nearest / largest / crosshair）
- [x] 配置热加载
- [x] FPS 监控与显示
- [x] 日志写入文件（自动轮转）
- [x] 减少检测结果拷贝开销

**待优化（按优先级排序）**
- [ ] 敌我识别模型重训 — 数据集只有 `person`/`else` 两类，无法区分敌我
- [ ] 捕获-推理双缓冲流水线 — 解决串行管线瓶颈，提升有效 FPS 30-50%
- [ ] 检测结果添加捕获时间戳 — 支持目标运动预测（target leading）
- [ ] 推理部分 C++ 重写
- [ ] 生成安装包

## 项目来源

基于 [EthanH3514/AL_Yolo](https://github.com/EthanH3514/AL_Yolo) 重构优化，感谢原作者的灵感。
