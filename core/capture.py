import numpy as np
import pyautogui


class ScreenCapture:

    def __init__(self, capture_size: int = 640):
        self.capture_size = capture_size
        screen_w, screen_h = pyautogui.size()
        self._center = (screen_w / 2, screen_h / 2)
        self._recalculate_region()

    def _recalculate_region(self):
        left = int(self._center[0] - self.capture_size / 2)
        top = int(self._center[1] - self.capture_size / 2)
        self._region = (left, top, left + self.capture_size, top + self.capture_size)

    def grab(self) -> tuple[np.ndarray, np.ndarray]:
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
