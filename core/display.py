import logging
import threading
import time
from queue import Queue, Full as QueueFull
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class Display:
    """在独立线程中运行 cv2.imshow,不阻塞调用方。"""

    def __init__(self, window_name: str = 'AL_Yolo_Detect', on_esc: Optional[Callable] = None):
        self._window_name = window_name
        self._on_esc = on_esc
        self._queue: Queue[Optional[np.ndarray]] = Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._should_stop = threading.Event()
        self._show_fps = False

    def set_show_fps(self, show: bool):
        self._show_fps = show

    def show(self, frame: np.ndarray):
        """发送帧到显示线程 (非阻塞,丢弃旧帧)。"""
        try:
            self._queue.put_nowait(frame)
        except QueueFull:
            pass

    def start(self):
        if self._thread is not None:
            return
        self._should_stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info('display thread started')

    def stop(self):
        self._should_stop.set()
        try:
            self._queue.put_nowait(None)
        except QueueFull:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        cv2.destroyWindow(self._window_name)
        log.info('display thread stopped')

    def _loop(self):
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        fps = 0.0
        frame_count = 0
        fps_timer = time.perf_counter()

        while not self._should_stop.is_set():
            frame = self._queue.get()
            if frame is None:
                break

            # FPS 统计
            frame_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.perf_counter()

            if self._show_fps and fps > 0:
                cv2.putText(
                    frame, f'FPS: {fps:.1f}', (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            cv2.imshow(self._window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                if self._on_esc:
                    self._on_esc()
                break

        cv2.destroyWindow(self._window_name)
