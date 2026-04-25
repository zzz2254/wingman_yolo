from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import yaml


@dataclass
class AppConfig:
    # ── 模型 ──
    weights: str = './weights/best.pt'
    """模型权重路径。支持 .pt (PyTorch) / .engine (TensorRT) / .onnx。"""

    data: str = './configs/AL_data.yaml'
    """数据集配置文件路径（仅模型加载时读一次）。"""

    imgsz: Sequence[int] = field(default_factory=lambda: (640, 640))
    """模型输入尺寸 (w, h)。训练时确定，不建议改。"""

    device: str = '0'
    """推理设备。'0' 为第一张 GPU，'cpu' 为 CPU。"""

    half: bool = True
    """FP16 推理。开启后显存减半、速度翻倍。"""

    augment: bool = False
    """TTA 测试时增强。会大幅降低帧率，一般不开。"""

    dnn: bool = False
    """使用 OpenCV DNN 替代 PyTorch 加载 ONNX。需配合 .onnx 权重。"""

    use_tensorrt: bool = True
    """优先使用 TensorRT 推理。会自动将 .pt 导出为 .engine 再加载。"""

    # ── 检测后处理（支持热加载） ──
    conf_thres: float = 0.45
    """置信度阈值。低于此值的检测框被丢弃。调低→更多框/更多误报，调高→更少框/可能漏检。"""

    iou_thres: float = 0.45
    """NMS 的 IoU 阈值。值越低重叠框删得越狠，适合稀疏目标；值越高允许更多重叠框共存，适合密集目标。"""

    max_det: int = 300
    """NMS 后保留的最大检测框数。"""

    agnostic_nms: bool = False
    """跨类别 NMS。True 时不同类别也做 NMS 合并。数据集只有 person/else 两类，一般不用。"""

    classes: Optional[Sequence[int]] = None
    """只保留指定类别的检测结果。None = 不过滤。"""

    # ── 目标识别（支持热加载） ──
    enemy_label: int = 0
    """敌人标签 ID。与 AL_data.yaml 的类别编号对应。默认 0=person(敌人), 1=else。"""

    target_strategy: str = 'nearest'
    """目标选择策略。可选值: nearest(最近中心) / largest(最大威胁) / crosshair(光标附近)。"""

    # ── 屏幕捕获（支持热加载，仅 IDLE 状态生效） ──
    capture_size: int = 640
    """截屏区域边长（正方形），范围 320-960。中心对齐。值越大视野越广但目标细节越少。仅在 IDLE 状态下可修改。"""

    # ── 鼠标控制（支持热加载） ──
    smooth_factor: float = 0.42
    """（旧版 atan2 控制器用）开环平滑系数。越大移动越灵敏。PID 控制器不使用此参数。"""

    mouse_sensitivity: float = 5.0
    """鼠标灵敏度倍率。"""

    pid_kp: float = 0.35
    """PID 比例增益。决定瞄准响应速度 —— 目标移动时准星跟多快。"""

    pid_ki: float = 0.02
    """PID 积分增益。消除稳态误差 —— 目标在准星附近时的微调力度。太大易过冲。"""

    pid_kd: float = 0.08
    """PID 微分增益。抑制抖动 —— 阻尼作用，太大手感变"肉"。"""

    pid_max_integral: float = 30.0
    """PID 积分限幅。防止积分项发散（windup）。"""

    pid_deadband: float = 2.0
    """PID 死区（像素）。误差小于此值时停止移动，避免像素级抖动。"""

    pid_kff: float = 0.0
    """速度前馈增益。利用目标速度信息补偿积分滞后，减少过冲。0=关闭。"""

    noise_amplitude: float = 0.5
    """人类化噪声注入幅度。在鼠标指令上叠加亚像素级随机抖动。0=关闭。"""

    # ── 显示（支持热加载） ──
    view_img: bool = True
    """显示检测预览窗口（OpenCV）。"""

    show_fps: bool = False
    """在预览窗口角落显示实时帧率。"""

    # ── 性能（支持热加载） ──
    aim_loop_hz: int = 120
    """瞄准控制循环频率。越高响应越快但 CPU 开销也大。120 为 8.3ms/次。"""

    # ── 配置持久化 ──
    _config_path: str = ''

    @classmethod
    def from_yaml(cls, path: str) -> 'AppConfig':
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'config file not found: {path}')
        raw = yaml.safe_load(p.read_text(encoding='utf-8'))
        raw['imgsz'] = tuple(raw.get('imgsz', (640, 640)))
        cfg = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        cfg._config_path = path
        cfg._ensure_defaults(raw)
        return cfg

    def to_yaml(self, path: str):
        d = {k: v for k, v in self.__dict__.items() if k in self.__dataclass_fields__}
        Path(path).write_text(yaml.safe_dump(d, allow_unicode=True), encoding='utf-8')
        self._config_path = path

    def reload(self) -> bool:
        """从 YAML 文件重新加载配置（热加载）。返回 True 表示成功。"""
        if not self._config_path or not Path(self._config_path).exists():
            return False
        try:
            raw = yaml.safe_load(Path(self._config_path).read_text(encoding='utf-8'))
        except Exception:
            return False
        changed = False
        for k, v in raw.items():
            if k in self.__dataclass_fields__ and k != '_config_path':
                setattr(self, k, tuple(v) if k == 'imgsz' else v)
                changed = True
        return changed

    def _ensure_defaults(self, raw: dict):
        """补充 config.yaml 中没有显式写出的字段（使用代码默认值）。"""
        for f in self.__dataclass_fields__:
            if f not in raw and f != '_config_path':
                setattr(self, f, getattr(self, f))
