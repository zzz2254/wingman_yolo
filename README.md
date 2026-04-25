# 一个基于yolov5的Apex Legend AI辅瞄外挂
## 🅿环境依赖
* 罗技驱动(版本不超过21.9)
* python >= 3.10 && python < 3.11
* CUDA 11
* torch >= 2.0
* 更多的依赖在 `requirements.txt` 中

## ♿快速开始
> 默认在windows系统下

#### 配置[scoop](https://scoop.sh/)(非常好用的windows包管理器)
打开PowerShell terminal(version 5.1 or later)
```
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
```

#### 配置anaconda
```
scoop bucket add extras
scoop install anaconda3
```

#### 创建conda环境并安装依赖
在项目文件夹下

```
conda create -n apex python=3.10
conda activate apex
pip install -r requirements.txt
```
#### 配置cuda、cudnn、pytorch
使用gpu进行推理所必须，配置过程较复杂，请去搜索专门的资料

#### 安装罗技驱动
提供一份2021.9版本，仅供学习

#### 运行
管理员模式打开一个终端
```
python apex.py
```

### 模型训练
#### 数据集地址
[https://github.com/goldjee/AL-YOLO-dataset](https://github.com/goldjee/AL-YOLO-dataset)
#### 训练方法
使用[Yolov5](https://github.com/ultralytics/yolov5) 的train.py文件，修改数据集路径参数
#### 训练日志
![log](https://user-images.githubusercontent.com/103171084/240091844-4bfd3af3-9d92-412e-8697-6d3ec8adc904.png)

训练日志以及过程的数据存放在`exp`目录下
### 屏幕捕捉
~~使用[DXcam](https://github.com/ra1nty/DXcam)截图来替代yolov5自带的mss截图，将截图时间从15ms优化到5ms~~

使用[Dxshot](https://github.com/AI-M-BOT/DXcam/releases)将截图时间进一步优化

注意：因为是通过屏幕截图的方式来获取的图像，所以屏幕刷新率会限制FPS的上限

### 鼠标控制
#### 如何控制鼠标

加入了win函数控制鼠标移动的选项，但是在训练场里试了试没有效果，有可能游戏更新之后已经把这个给修复掉了(？)，而且正常情况下win函数的效率太低，所以默认还是鼠标驱动控制鼠标

#### 鼠标移动函数
![](https://user-images.githubusercontent.com/103171084/241761731-2293a5f2-6421-4d37-b353-d6ec7ea2ccc7.png)

暂时没想出来如何将鼠标直接移过去，网上也找不到前人的经验。

基于内存的外挂自瞄原理是可以拿到三维坐标，直接修改方向角来瞄准敌人，而基于计算机视觉的外挂只能拿到目标在屏幕上的投影，这是一个二维坐标，要解算出移动的向量很依赖游戏底层的参数(视场角等，详见[issue](https://github.com/EthanH3514/AL_Yolo/issues/3))，目前还没想明白怎么一帧锁敌，也许将来会去实现。

况且一帧锁敌会大大提高被检测的风险性，实现的价值不大，短期内不会再向这个方向去努力了。

目前采用的是将准星逐步移动到目标身上的方法，牺牲了一点点效率，达到了准星吸附一样的效果。

update: 将fov的算法初步投入使用，效果还不错

### 🤔如何使用
- 将代码下载到本地
- 部署环境依赖
- 管理员模式打开一个终端，进入项目文件夹下运行`python apex.py`

### 🎯后续改进

**已完成**
- [x] 截图方式优化（Dxshot）
- [x] 添加自瞄开关（PgUp/PgDn 控制）
- [x] 取消对驱动的依赖（回退 SendInput）
- [x] 多目标识别优先级判断
- [x] 项目架构优化（EventBus + StateMachine + 策略模式）
- [x] 对不同机器参数自适应
- [x] 做个前端（Tkinter GUI）
- [x] 支持更多YOLO系列模型（v5/v8/v11 统一 API）

**待优化（按优先级排序）**

- [ ] **屏幕捕获改用 Dxshot** — 当前 `pyautogui.screenshot` 约 15ms/帧，Dxshot 可压到 5ms（已编译 `dxshot.pyd`  但未启用）
- [ ] **敌我识别模型重训** — 数据集只有 `person`/`else` 两类，无法区分敌我。用 [AL-YOLO-dataset](https://github.com/goldjee/AL-YOLO-dataset) 重训为 `enemy`/`teammate`/`else`
- [ ] **PID 平滑控制鼠标** — 当前 atan2 开环控制手感突兀，PID 闭环可显著提升平滑度
- [ ] **TensorRT 推理加速** — FP16 → INT8 量化，预期推理再快 2-3 倍
- [ ] **高精度瞄准定时器** — `time.sleep` 在 Windows 上精度约 15ms，无法跑满 120Hz。改用 `time.perf_counter` 自旋或多媒体定时器
- [ ] **分离显示线程** — `cv2.imshow` 阻塞推理循环，应单独开线程渲染检测可视化
- [ ] **多目标优先级策略扩展** — 增加残血优先、准星距离优先等策略
- [ ] **配置热加载** — 修改参数需重启 GUI，改为文件监听或运行时修改
- [ ] **FPS 监控与显示** — 已有 `config.show_fps` 字段，但未实现
- [ ] **日志写入文件** — 当前仅输出到控制台，崩溃后无历史可查
- [ ] **减少检测结果拷贝开销** — `latest_detections.copy()` 在高频推理下有一定性能损耗
- [ ] **推理部分 C++ 重写** — 极限优化性能
- [ ] **生成安装包** — 打包为可分发安装程序

### 🤩如果有帮到你就点一个star吧

## 🎉Star History

[![Star History Chart](https://api.star-history.com/svg?repos=EthanH3514/AL_Yolo&type=Date)](https://star-history.com/#EthanH3514/AL_Yolo&Date)
