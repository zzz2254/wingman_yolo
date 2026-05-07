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
- **AIMING**: 检测 + 吸附辅助均激活
- **SHUTDOWN**: 停止所有线程，清理资源

### 吸附辅助状态机（AimController 内部）

```
FREE ──[准星靠近敌框]──→ ENGAGED ──[人控背离目标]──→ OVERRIDE
  ↑                         │                            │
  └── [目标消失] ───────────┘                            │
  ↑                                                      │
  └──────── [人控靠近新目标] ─────────────────────────────┘
```

- **FREE**: 扫描检测框，光标进入 `aim_engage_range` 时自动吸附
- **ENGAGED**: 对锁定目标施加磁力，越近越强；监测人控接管
- **OVERRIDE**: 人手主动脱离后完全放手，直到光标靠近新目标才重新吸附

> 参考：手柄辅瞄的"粘性瞄准"（sticky aim / aim magnetism）。以人控鼠标为第一优先级，仅在准星靠近敌人时自动吸附。

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
│                          瞄准线程 (AimController, 自适应 Hz)                         │
│                                                                                     │
│  读取最新检测结果 + 光标位置                                                          │
│         ↓                                                                           │
│  if state == AIMING:                                                                │
│         ↓                                                                           │
│  吸附状态机 (FREE / ENGAGED / OVERRIDE)                                              │
│    ├─ FREE: 光标靠近敌框 → 锁定, 进入 ENGAGED                                        │
│    ├─ ENGAGED: IoU 匹配跟踪, 磁力吸附, 检测人控脱离                                  │
│    └─ OVERRIDE: 人控优先, 靠近新目标 → 重新 ENGAGED                                  │
│         ↓                                                                           │
│  磁力计算 (P 控制器, 越近越强) → move_raw(dx, dy)                                    │
│    ├─ 强度 = magnet_strength × base_speed × 距离系数                                 │
│    ├─ 人控同向 → 叠加, 人控反向 → 减弱                                               │
│    └─ 人控位移 > override_threshold 且背离 → OVERRIDE                                │
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
`pid_kff` / `noise_amplitude` / `mouse_sensitivity` /
`aim_magnet_strength` / `aim_engage_range` / `aim_override_threshold` / `aim_base_speed` /
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

## 瞄准策略

### 吸附辅助 (Sticky Aim Assist)

模拟手柄 FPS 的"粘性瞄准"（aim magnetism），以人控鼠标为第一优先级。

实现于 `core/aim_controller.py` 的 `AssistState` 状态机：

#### 状态流转

```
准星在敌框外        准星进入 engage_range     人控拖开
   FREE ──────────→ ENGAGED ──────────→ OVERRIDE
    ↑                   │                     │
    └── 敌人消失 ───────┘                     │
    ↑                                         │
    └────── 人控靠近另一个敌框 ←────────────────┘
```

- **FREE**: 无辅助，每帧扫描是否有检测框进入 `aim_engage_range`（默认 60px）
- **ENGAGED**: 用 IoU 匹配锁定目标，施加磁力拉向目标中心。越靠近目标磁力越强（30%→100%）。如果人也在动鼠标，人控同向则磁力叠加，人控反向则磁力减到 30%。检测人控位移是否 > `aim_override_threshold` 且方向背离目标→切换 OVERRIDE
- **OVERRIDE**: 完全放手，人控优先。持续扫描新目标进入范围，排除刚脱离的同一目标（防立即回吸）

#### 磁力计算

不使用完整 PID，而用简单的 P 控制器实现"重力井"效果：

```python
distance = hypot(target_cx - cursor_x, target_cy - cursor_y)
t = 1.0 - min(distance / engage_range, 1.0)   # 0(边缘) → 1(中心)
pull = magnet_strength * (0.3 + 0.7 * t)       # 最低 30% 磁力
output = direction_to_target * pull * base_speed
```

- `magnet_strength`: 吸附力度（0.4 默认，0=关闭, 1=全锁）
- `engage_range`: 触发距离（60px 默认）
- `base_speed`: 基础速度（4px/帧 默认，100Hz = 400px/s）

### PID 控制器（备用）

`PidMouseController` 保留可用，包含以下优化：

| 方案 | 状态 | 说明 |
|------|------|------|
| 速度前馈 | ✅ | `u += Kff × V_target`，`pid_kff` 控制 |
| 分段增益调度 | ✅ | 根据误差距离自动切换，无需配置 |
| 人类化噪声注入 | ✅ | 跟枪时高斯噪声，`noise_amplitude` 控制 |
| 输出限幅 | ✅ | 单帧最大 ±100px |
| 目标切换检测 | ✅ | Δerror > 200px 自动重置积分+微分 |
| 自动校准 | ✅ | 启动时测 `px_per_unit` 比率 |

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
- [x] 手柄式吸附辅助（FREE/ENGAGED/OVERRIDE 状态机 + 磁力 + 人控检测）
- [ ] 敌我识别模型重训 — 数据集只有 `person`/`else` 两类，无法区分敌我
- [ ] 卡尔曼滤波平滑轨迹 — 替代线性外推+低通滤波
- [ ] 推理部分 C++ 重写
- [ ] 生成安装包

## 瞄准策略设计理念

手柄吸附式瞄准：以人控鼠标为第一优先级。准星移动到检测框附近时自动吸附，维持锁定直到敌人消失。人控鼠标移出吸附范围时自动放手，保持人控直到移动到另一个检测框附近再重新吸附。实现细节见上文「吸附辅助」章节。

## 项目来源

基于 [EthanH3514/AL_Yolo](https://github.com/EthanH3514/AL_Yolo) 重构优化，感谢原作者的灵感。
