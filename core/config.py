from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import yaml


@dataclass
class AppConfig:
    weights: str = './weights/best.pt'
    data: str = './configs/AL_data.yaml'
    imgsz: Sequence[int] = field(default_factory=lambda: (640, 640))
    conf_thres: float = 0.45
    iou_thres: float = 0.45
    max_det: int = 300
    device: str = '0'
    half: bool = True
    augment: bool = False
    agnostic_nms: bool = False
    dnn: bool = False
    enemy_label: int = 0
    classes: Optional[Sequence[int]] = None

    capture_size: int = 640

    smooth_factor: float = 0.42
    mouse_sensitivity: float = 5.0

    view_img: bool = True
    show_fps: bool = False

    aim_loop_hz: int = 120

    @classmethod
    def from_yaml(cls, path: str) -> 'AppConfig':
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'config file not found: {path}')
        raw = yaml.safe_load(p.read_text(encoding='utf-8'))
        raw['imgsz'] = tuple(raw.get('imgsz', (640, 640)))
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str):
        Path(path).write_text(yaml.safe_dump(self.__dict__, allow_unicode=True), encoding='utf-8')
