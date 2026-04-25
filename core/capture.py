import logging
import time

import numpy as np
import pyautogui

log = logging.getLogger(__name__)

try:
    import dxshot as _dxshot
    _HAVE_DXSHOT = True
    log.info('dxshot available, will use DXGI screen capture')
except ImportError:
    _HAVE_DXSHOT = False
    log.info('dxshot not available, falling back to pyautogui')


class ScreenCapture:

    def __init__(self, capture_size: int = 640):
        self.capture_size = capture_size
        screen_w, screen_h = pyautogui.size()
        self._center = (screen_w / 2, screen_h / 2)
        self._region = self._make_region()

        self._dxshot_cam = None
        self._retry_at = 0.0      # 时间戳,在此之前不重试
        self._retry_delay = 1.0   # 当前退避秒数
        if _HAVE_DXSHOT:
            self._init_dxshot()
            if self._dxshot_cam is None:
                self._schedule_retry()

    def _make_region(self):
        half = self.capture_size / 2
        left = int(self._center[0] - half)
        top = int(self._center[1] - half)
        return left, top, left + self.capture_size, top + self.capture_size

    def _init_dxshot(self):
        try:
            cam = _dxshot.create()
            cam.start(region=self._region, target_fps=self.capture_size // 4)
            self._dxshot_cam = cam
            log.info(
                'dxshot camera initialized, region=(%d,%d,%d,%d)', *self._region,
            )
        except Exception:
            log.exception('dxshot camera init failed')
            self._dxshot_cam = None

    def _schedule_retry(self):
        """失败后安排一次重试,退避时间翻倍 (1s → 2s → 4s → 8s → 16s → 30s cap)。"""
        self._retry_at = time.monotonic() + self._retry_delay
        self._retry_delay = min(self._retry_delay * 2, 30.0)

    def _try_recover(self) -> bool:
        """尝试重建 dxshot 相机。成功时重置退避,失败时继续退避。"""
        self._init_dxshot()
        if self._dxshot_cam is not None:
            self._retry_delay = 1.0
            self._retry_at = 0.0
            log.info('dxshot recovered')
            return True
        self._schedule_retry()
        return False

    def grab(self) -> tuple[np.ndarray, np.ndarray]:
        # 相机缺失且到重试时间 → 尝试重建
        if self._dxshot_cam is None and _HAVE_DXSHOT and time.monotonic() >= self._retry_at:
            self._try_recover()

        if self._dxshot_cam is not None:
            try:
                return self._grab_dxshot()
            except Exception:
                log.warning('dxshot grab failed, stopping camera')
                try:
                    self._dxshot_cam.stop()
                except Exception:
                    pass
                self._dxshot_cam = None
                self._schedule_retry()

        return self._grab_fallback()

    def _grab_dxshot(self) -> tuple[np.ndarray, np.ndarray]:
        frame = self._dxshot_cam.get_latest_frame()
        if frame is None:
            return self._grab_fallback()
        im0 = np.ascontiguousarray(frame)
        im = im0.transpose((2, 0, 1)).copy()
        return im, im0

    def _grab_fallback(self) -> tuple[np.ndarray, np.ndarray]:
        left, top, right, bottom = self._region
        width = right - left
        height = bottom - top
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        im0 = np.array(screenshot)
        im = im0.transpose((2, 0, 1)).copy()
        im = np.ascontiguousarray(im)
        return im, im0

    def __iter__(self):
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        return self.grab()

    def resize(self, new_size: int):
        """运行时修改截屏区域大小（需在 IDLE 状态调用）。"""
        if new_size == self.capture_size:
            return
        self.capture_size = new_size
        self._region = self._make_region()
        self._retry_at = 0.0
        self._retry_delay = 1.0
        if self._dxshot_cam is not None:
            try:
                self._dxshot_cam.stop()
            except Exception:
                pass
            self._dxshot_cam = None
            self._init_dxshot()
            if self._dxshot_cam is None:
                self._schedule_retry()
        log.info('capture resized to %dx%d, region=%s', new_size, new_size, self._region)
