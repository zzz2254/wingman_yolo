# wingman_yolo

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

管理员模式打开终端（鼠标输入需要管理员权限，否则 EAC 可能直接拦截非特权进程的输入）：

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
│   ├── capture.py       # 屏幕捕获（Dxshot + mss 回退）
│   ├── detector.py      # YOLO 检测引擎（AutoBackend + TensorRT 自动导出）
│   ├── strategies.py    # 目标选择策略 + PID/atan2 鼠标控制
│   ├── aim_controller.py# 瞄准控制主循环（120Hz 高精度定时器）
│   ├── overlay.py       # Win32 Layered Window 全屏透明覆盖层（检测框绘制）
│   └── display.py       # [已废弃] OpenCV 预览窗口
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
                                      ┌──────────────────────────────────────────────┐
                                      │             捕获线程 (CapturePipeline)         │
                                      │                                              │
                                      │  ScreenCapture.grab()  (循环)                │
                                      │    ├─ 优先 Dxshot (~5ms 延迟)                 │
                                      │    └─ 回退 mss (~12ms 延迟)                    │
                                      │         ↓                                    │
                                      │  截取屏幕中央区域，写入 latest_frame            │
                                      │  Event 通知推理线程                            │
                                      └──────────────────┬───────────────────────────┘
                                                         │ (latest frame)
┌────────────────────────────────────────────────────────▼───────────────────────────┐
│                          推理线程 (DetectionEngine)                                  │
│                                                                                     │
│  while True:                                                                        │
│    等待帧事件 → 读取最新帧 (丢弃中间帧保持低延迟)                                     │
│         ↓                                                                           │
│  DetectionEngine.infer()  (CUDA Stream)                                             │
│    ├─ TensorRT 自动导出 (.pt → .engine, FP16 加速)                                  │
│    ├─ 预处理: uint8 → float16, /255, unsqueeze (non_blocking H2D)                   │
│    ├─ AutoBackend 推理 (ultralytics)                                                │
│    └─ GPU NMS (torchvision.ops.nms, 消除 CPU 同步点)                                │
│         ↓                                                                           │
│  publish('detect.result', detections=...)                                           │
│         ↓                                                                           │
│  DetectionOverlay 独立线程渲染（Win32 Layered Window, 全屏透明置顶）           │
└─────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (EventBus 异步传递引用)
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          瞄准线程 (AimController, 120Hz)                              │
│                                                                                     │
│  读取最新检测结果                                                                     │
│         ↓                                                                           │
│  if state == AIMING && mouse_pressed:                                               │
│         ↓                                                                           │
│  StrategyRouter.select(target_strategy 可运行时切换)                                  │
│    ├─ 'nearest'   → NearestEnemySelector   (离屏幕中心最近)                          │
│    ├─ 'largest'   → LargestEnemySelector    (框面积最大≈威胁最大)                   │
│    └─ 'crosshair' → CrosshairProximitySelector (离鼠标光标最近)                      │
│         ↓                                                                           │
│  PidMouseController.move()                                                          │
│    ├─ 计算误差: 目标中心 - 画面中心                                                   │
│    ├─ Deadband 检查: 误差 < 2px → 跳过 (防像素级抖动)                                │
│    ├─ PID 计算: P(比例) + I(积分, 限幅30) + D(微分)                                 │
│    └─ 输出 (dx, dy) → ghub_mouse.moveR() / SendInput                                │
│                                                                                     │
│  定时器: sleep 大部分时间 + 2ms 自旋校准 (跑满设定 Hz)                               │
└─────────────────────────────────────────────────────────────────────────────────────┘
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

## 性能瓶颈分析（pyautogui 回退已修复）

### 当前管线全景

系统分为三个并行线程：**捕获线程**（独立 CPU 截图）、**推理线程**（GPU 检测 + NMS）和**瞄准线程**（120Hz PID 控制）。捕获和推理已通过双缓冲流水线重叠，GPU 推理第 N 帧时 CPU 同时捕获第 N+1 帧。

### 瓶颈排序（由重到轻）

#### 1. ✅ 串行的捕获-推理管线（已修复）

**现状**: 捕获线程独立运行后，GPU 推理和 CPU 截图完全重叠。有效 FPS 从 50-80 提升至接近纯推理速度（约 80-120 FPS，视 GPU 而定）。

#### 2. ✅ 推理帧率与瞄准帧率不匹配（已修复）

**现状**: AimController 每秒自动统计检测帧率，将循环频率设为 `检测FPS × 1.5`（范围 30-240Hz），保证足够的新鲜数据供给 PID 控制器同时避免过多无效周期。

#### 3. ✅ 检测结果缺乏时间戳（已修复）

**现状**: 检测管线携带 `capture_timestamp`（`time.perf_counter()`），AimController 根据数据年龄做运动预测（target leading）——追踪目标中心位移、低通滤波估计速度、外推当前位置。最远外推 100ms，数据新鲜（<2ms）时跳过预测。

#### 4. ✅ pyautogui 回退已替换为 mss（已修复）

**现状**: `pyautogui.screenshot()`（PIL 底层，延迟 30-100ms）已替换为 `mss`（BitBlt Win32 API C 扩展，延迟 ~12ms）。回退场景 FPS 从 10-30 提升至 60-90。仅在 Dxshot 不可用时（远程桌面、特定 GPU 驱动）触发。

#### 5. ✅ 缺少异步 CUDA 流（已修复）

**现状**: 预处理使用 `non_blocking=True` 异步 H2D 传输，推理在独立 `torch.cuda.Stream` 上执行。CPU 在 GPU 工作期间不再空等。

#### 6. ✅ NMS 后处理在 CPU 上执行（已修复）

**现状**: NMS 全程在 GPU 上通过 `torchvision.ops.nms` 执行，消除了 CPU-GPU 同步点。只有最终过滤后的结果才拷回 CPU。

#### 7. ✅ 检测结果的线程安全读取（已修复）

**现状**: `threading.Lock` 已替换为 `queue.Queue(maxsize=1)`。推理线程通过 `put_nowait` 写入最新检测结果，瞄准线程通过 `get_nowait` 原子性获取——不存在任何共享可变状态，从根源消除 data race。

### 总结

| 瓶颈 | 影响 | 优先级 | 状态 |
|------|------|--------|------|
| 串行捕获-推理管线 | 有效 FPS 受限于串行延迟 | 🔴 高 | ✅ 已修复 |
| 帧率不匹配 | 部分循环处理过期数据 | 🟡 中 | ✅ 已修复 |
| 检测缺少时间戳 | 无法做运动预测 | 🟡 中 | ✅ 已修复 |
| pyautogui 回退 | 极端场景 FPS 暴跌 | 🟡 中 | ✅ 已修复 |
| 无 CUDA Stream | CPU 空闲等待 | 🟢 低 | ✅ 已修复 |
| CPU 端 NMS | 微秒级开销 | 🟢 低 | ✅ 已修复 |
| 检测读取 data race | 代码正确性 | 🟢 低 | ✅ 已修复 |

**已完成优化**: 双缓冲流水线 + GPU NMS + CUDA Stream + 帧率动态适配 + 目标运动预测 + mss 截图回退 + Queue 消除 data race。

### 屏幕捕获

- 优先使用 **Dxshot**（DXGI Desktop Duplication），延迟 ~5ms
- 自动回退到 **mss**（BitBlt Win32 API），延迟 ~12ms
- 截取屏幕中央正方形区域，GUI 中可选 480 / 540 / 640 / 720 / 800 / 960 / 1280 像素

## 检测框渲染（Layered Window 方案）

检测框不通过 OpenCV 窗口显示，而是直接在游戏画面上叠加全屏透明窗口（Win32 Layered Window），实现"内嵌"视觉效果。

### 技术原理

```
                    全屏 Layered Window (2560×1440, WS_EX_LAYERED)
                    ┌──────────────────────────────────────────────┐
                    │  DIB Section (32-bit BGRA, stride=w×4)       │
                    │  ┌─────────────────────────────────────┐     │
                    │  │ numpy canvas (h, w, 4)              │     │
                    │  │                                     │     │
                    │  │   每帧渲染:                           │     │
                    │  │   1. canvas[:] = 0    (清空)         │     │
                    │  │   2. cv2.rectangle()  (画检测框)      │     │
                    │  │   3. bg>0 → alpha=200 (确保可见)      │     │
                    │  │   4. UpdateLayeredWindow()  (上屏)    │     │
                    │  └─────────────────────────────────────┘     │
                    │                                             │
                    │  游戏画面（底层，被覆盖层透明部分穿透）           │
                    └──────────────────────────────────────────────┘
```

### 窗口属性

| 属性 | 值 | 说明 |
|------|-----|------|
| 扩展样式 | `WS_EX_LAYERED \| WS_EX_TRANSPARENT \| WS_EX_TOPMOST \| WS_EX_NOACTIVATE` | 分层窗口 + 点击穿透 + 置顶 + 不抢焦点 |
| 窗口样式 | `WS_POPUP` | 无边框弹窗 |
| 混合模式 | `ULW_ALPHA` + `AC_SRC_ALPHA` | 逐像素 Alpha 混合 |
| 画布格式 | 32-bit BGRA（top-down DIB） | 通过 numpy array 直接写入 DIB 内存 |

### 渲染管线

```
DetectionEngine.run()
  │
  ├─ self._overlay.start()          # 创建窗口 + DIB Section + 渲染线程
  │
  └─ 每帧:
       overlay.set_region_offset(capture_left, capture_top)   # 同步截屏区域偏移
       overlay.show(detections)        # 非阻塞，Queue(maxsize=2) 传递
              │
              ▼
       DetectionOverlay._loop()        # 独立线程
         ├─ queue.get()                # 阻塞等待
         ├─ _render(detections)        # 绘制到 DIB canvas
         │    ├─ 坐标转换: (x_capture, y_capture) → (x_screen, y_screen)
         │    ├─ 敌人 (cls==enemy_label) → 绿色框
         │    └─ 非敌人 → 红色框
         └─ _present()                 # UpdateLayeredWindow 上屏
```

### 与 OpenCV 方案的对比

| 特性 | Layered Window (当前) | OpenCV imshow (废弃) |
|------|----------------------|---------------------|
| 显示位置 | 直接叠加在游戏画面上 | 独立窗口，与游戏分离 |
| 鼠标穿透 | ✅ 点击直达游戏 | ❌ 窗口拦截鼠标 |
| 性能 | UpdateLayeredWindow (~0.1ms) | cv2.imshow (~1-2ms) |
| 焦点干扰 | ❌ 不抢焦点 | ✅ 弹窗时失去游戏焦点 |
| 全屏游戏兼容 | ⚠️ 独占全屏模式可能被遮挡 | ❌ 无法在全屏游戏上显示 |

> **注意**: 游戏运行在**独占全屏**模式时，GPU 直接控制屏幕输出，任何桌面窗口（包括 TOPMOST）都无法叠加。建议使用**无边框窗口**或**窗口化全屏**模式。

## 鼠标控制

- 优先使用 **罗技 G Hub 驱动**（`ghub_mouse.dll`），更底层不易被游戏屏蔽
- 驱动不可用时自动回退到 Win32 `SendInput`
- PID 闭环控制器（默认）：kp=0.35, ki=0.02, kd=0.08, deadband=2px，支持速度前馈、分段增益调度、噪声注入
- 后备 atan2 开环控制器（保留未删除）

### EAC 拦截验证

两条鼠标路径桌面（管理员模式）均正常，但在 Apex Legends 游戏内均被阻止：

| 路径 | 桌面（管理员） | 游戏内 | 结论 |
|------|--------------|--------|------|
| G Hub `gm.moveR()` | ✅ 鼠标移动 | ❌ 无效 | EAC 拦截 G Hub 驱动调用 |
| Win32 `SendInput` | ✅ 鼠标移动 | ❌ 无效 | EAC 拦截用户态输入 API |

EAC 在游戏进程中拦截了所有用户态的鼠标模拟 API，如需绕过必须使用硬件 HID 设备（如 Arduino Leonardo）。

## 鼠标控制优化方向

### 控制算法

当前 PID 控制器已实现以下优化：

| 方案 | 状态 | 说明 |
|------|------|------|
| **速度前馈** | ✅ 已实现 | `u += Kff × V_target`，目标速度来自运动预测模块。`config.yaml` 中 `pid_kff` 控制，0=关闭 |
| **分段增益调度** | ✅ 已实现 | 根据误差距离自动切换 PID 倍率（见下文），无需配置 |
| **人类化噪声注入** | ✅ 已实现 | 跟枪时叠加高斯噪声（`σ = amplitude/3`）。`config.yaml` 中 `noise_amplitude` 控制，0=关闭 |
| Flick / Tracking 模式分层 | ⬜ 待评估 | 偏差大时开环甩枪、偏差小时 PID 跟枪 |
| 卡尔曼滤波平滑轨迹 | ⬜ 待评估 | 替代线性外推+低通滤波 |

#### 速度前馈

```
输出 = Kp·e + Ki·∫e + Kd·de/dt + Kff·v_target
                                   ↑ 运动预测模块提供的目标速度
```

- 从 `_on_config_reloaded` 热加载，修改 `config.yaml` → 点击"重载配置"生效
- 建议调参顺序：先确定 PID 三参数，再从 `pid_kff: 0.05` 开始逐渐增大

#### 分段增益调度

对基础增益 `(kp, ki, kd)` 按距离缩放，无额外配置：

| 距离 | Kp 倍率 | Ki 倍率 | Kd 倍率 | 行为 |
|------|---------|---------|---------|------|
| > 200px | ×1.4 | ×0.5 | ×0.6 | 快速拉近，抑制过冲 |
| 50-200px | ×1.0 | ×1.0 | ×1.0 | 平稳跟踪 |
| < 50px | ×0.6 | ×1.5 | ×1.5 | 精细瞄准，加大阻尼 |

缩放倍率为类常量（`_FAR_KP_MUL` 等），如需修改需编辑源码 `strategies.py`。

#### 人类化噪声注入

```python
sigma = noise_amplitude / 3   # 3σ ≈ ±amplitude
dx += random.gauss(0, sigma)
dy += random.gauss(0, sigma)
```

- 仅在跟枪（距离 < 80px）时注入，甩枪不叠加
- 高斯分布比均匀分布更接近真实生理震颤特征
- `noise_amplitude: 0.5`（默认值）对应 99.7% 噪声落在 ±0.5px 内

### 输入层（驱动/API）

当前 fallback 链：`G Hub 驱动 → Win32 SendInput`

> ⚠️ **EAC 拦截确认**: 上述两条路径在桌面管理员模式下正常工作，但在 Apex 游戏内均被 EAC 拦截。必须使用硬件 HID 设备才能绕过。

| 方案 | 反作弊回避 | 集成难度 | 说明 |
|------|-----------|---------|------|
| SendInput | ⭐⭐ ❌ EAC 已拦截 | ✅ 已有 | 桌面可用，游戏内无效 |
| G Hub 驱动 | ⭐⭐ ❌ EAC 已拦截 | ✅ 已有 | 桌面可用，游戏内无效 |
| **Interception 驱动** | ⭐⭐⭐⭐ | ⚠️ 需驱动签名 | 驱动层注入，硬件与系统之间拦截输入，推荐后备 |
| VMulti 虚拟 HID | ⭐⭐⭐ | ⚠️ 需驱动签名 | 虚拟 USB HID 设备 |
| KmBox / Arduino 硬件 | ⭐⭐⭐⭐⭐ | 🔧 需外设 | USB 硬件模拟真实鼠标，理论最安全 |

当前 SendInput 足够应对多数场景，仅当遇到反作弊封锁时考虑 Interception。

**待优化（按优先级排序）**
- [ ] 敌我识别模型重训 — 数据集只有 `person`/`else` 两类，无法区分敌我
- [ ] Flick / Tracking 模式分层
- [ ] 卡尔曼滤波平滑轨迹
- [ ] 推理部分 C++ 重写
- [ ] 生成安装包

## 项目来源

基于 [EthanH3514/AL_Yolo](https://github.com/EthanH3514/AL_Yolo) 重构优化，感谢原作者的灵感。
