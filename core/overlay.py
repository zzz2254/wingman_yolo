import ctypes
import ctypes.wintypes
import logging
import threading
from queue import Queue, Full as QueueFull
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Win32 Constants ──
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
WS_EX_TOPMOST = 0x8
WS_EX_NOACTIVATE = 0x08000000
WS_POPUP = 0x80000000
SW_SHOW = 5
AC_SRC_ALPHA = 0x01
ULW_ALPHA = 0x02
BI_RGB = 0
DIB_RGB_COLORS = 0

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

WNDPROC = ctypes.WINFUNCTYPE(
    ctypes.c_longlong, ctypes.wintypes.HWND, ctypes.wintypes.UINT,
    ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM,
)

WINDOW_CLASS = 'Wingman_Yolo_Overlay_Cls'

# ── Win32 Structures ──

class POINT(ctypes.Structure):
    _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]

class SIZE(ctypes.Structure):
    _fields_ = [('cx', ctypes.c_long), ('cy', ctypes.c_long)]

class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ('BlendOp', ctypes.c_byte),
        ('BlendFlags', ctypes.c_byte),
        ('SourceConstantAlpha', ctypes.c_byte),
        ('AlphaFormat', ctypes.c_byte),
    ]

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ('biSize', ctypes.wintypes.DWORD),
        ('biWidth', ctypes.c_long),
        ('biHeight', ctypes.c_long),
        ('biPlanes', ctypes.wintypes.WORD),
        ('biBitCount', ctypes.wintypes.WORD),
        ('biCompression', ctypes.wintypes.DWORD),
        ('biSizeImage', ctypes.wintypes.DWORD),
        ('biXPelsPerMeter', ctypes.c_long),
        ('biYPelsPerMeter', ctypes.c_long),
        ('biClrUsed', ctypes.wintypes.DWORD),
        ('biClrImportant', ctypes.wintypes.DWORD),
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [('bmiHeader', BITMAPINFOHEADER)]

class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ('cbSize', ctypes.wintypes.UINT),
        ('style', ctypes.wintypes.UINT),
        ('lpfnWndProc', WNDPROC),
        ('cbClsExtra', ctypes.c_int),
        ('cbWndExtra', ctypes.c_int),
        ('hInstance', ctypes.wintypes.HINSTANCE),
        ('hIcon', ctypes.wintypes.HICON),
        ('hCursor', ctypes.wintypes.HANDLE),
        ('hbrBackground', ctypes.wintypes.HBRUSH),
        ('lpszMenuName', ctypes.wintypes.LPCWSTR),
        ('lpszClassName', ctypes.wintypes.LPCWSTR),
        ('hIconSm', ctypes.wintypes.HICON),
    ]

# ── Win32 DLLs ──
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

# ── Configure function signatures ──
user32.GetDC.argtypes = [ctypes.wintypes.HWND]
user32.GetDC.restype = ctypes.wintypes.HDC
user32.ReleaseDC.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.HDC]
user32.ReleaseDC.restype = ctypes.c_int
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int
user32.CreateWindowExW.argtypes = [
    ctypes.wintypes.DWORD, ctypes.wintypes.LPCWSTR, ctypes.wintypes.LPCWSTR,
    ctypes.wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.wintypes.HWND, ctypes.wintypes.HMENU, ctypes.wintypes.HINSTANCE, ctypes.wintypes.LPVOID,
]
user32.CreateWindowExW.restype = ctypes.wintypes.HWND
user32.DestroyWindow.argtypes = [ctypes.wintypes.HWND]
user32.DestroyWindow.restype = ctypes.wintypes.BOOL
user32.ShowWindow.argtypes = [ctypes.wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = ctypes.wintypes.BOOL
user32.SetWindowPos.argtypes = [
    ctypes.wintypes.HWND, ctypes.wintypes.HWND,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.wintypes.UINT,
]
user32.SetWindowPos.restype = ctypes.wintypes.BOOL
user32.UpdateLayeredWindow.argtypes = [
    ctypes.wintypes.HWND, ctypes.wintypes.HDC,
    ctypes.POINTER(POINT), ctypes.POINTER(SIZE),
    ctypes.wintypes.HDC, ctypes.POINTER(POINT),
    ctypes.wintypes.COLORREF, ctypes.POINTER(BLENDFUNCTION),
    ctypes.wintypes.DWORD,
]
user32.UpdateLayeredWindow.restype = ctypes.wintypes.BOOL
user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
user32.RegisterClassExW.restype = ctypes.wintypes.ATOM
user32.DefWindowProcW.argtypes = [
    ctypes.wintypes.HWND, ctypes.wintypes.UINT,
    ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM,
]
user32.DefWindowProcW.restype = ctypes.c_longlong

gdi32.CreateCompatibleDC.argtypes = [ctypes.wintypes.HDC]
gdi32.CreateCompatibleDC.restype = ctypes.wintypes.HDC
gdi32.DeleteDC.argtypes = [ctypes.wintypes.HDC]
gdi32.DeleteDC.restype = ctypes.wintypes.BOOL
gdi32.CreateDIBSection.argtypes = [
    ctypes.wintypes.HDC, ctypes.POINTER(BITMAPINFO), ctypes.wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p), ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD,
]
gdi32.CreateDIBSection.restype = ctypes.wintypes.HBITMAP
gdi32.SelectObject.argtypes = [ctypes.wintypes.HDC, ctypes.wintypes.HGDIOBJ]
gdi32.SelectObject.restype = ctypes.wintypes.HGDIOBJ
gdi32.DeleteObject.argtypes = [ctypes.wintypes.HGDIOBJ]
gdi32.DeleteObject.restype = ctypes.wintypes.BOOL

kernel32.GetModuleHandleW.argtypes = [ctypes.wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = ctypes.wintypes.HINSTANCE

FONT = cv2.FONT_HERSHEY_SIMPLEX


class DetectionOverlay:
    """全屏透明覆盖层,在游戏画面上直接绘制检测框。

    原理:
      Win32 Layered Window + 逐像素 Alpha (UpdateLayeredWindow)。
      用 OpenCV 在 BGRA numpy 数组上绘制,数组与 DIB Section 共享内存,
      然后通过 UpdateLayeredWindow 显示。

    特性:
      - 全屏置顶,鼠标点击穿透 (WS_EX_TRANSPARENT)
      - 不接收焦点 (WS_EX_NOACTIVATE)
      - 逐像素 Alpha,抗锯齿边缘平滑
    """

    def __init__(self):
        self._hwnd = None
        self._screen_x = 0
        self._screen_y = 0
        self._screen_w = 0
        self._screen_h = 0
        self._thread: Optional[threading.Thread] = None
        self._should_stop = threading.Event()
        self._queue: Queue[Optional[np.ndarray]] = Queue(maxsize=2)

        self._mem_dc = None
        self._dib_hbitmap = None
        self._canvas: Optional[np.ndarray] = None

        self._region_left = 0
        self._region_top = 0
        self._enemy_label = 0

        self._screen_x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        self._screen_y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        self._screen_w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        self._screen_h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

        # Keep WNDPROC alive to prevent GC
        self._wndproc = WNDPROC(
            lambda h, m, w, l: user32.DefWindowProcW(h, m, w, l),
        )

    def set_region_offset(self, left: int, top: int):
        self._region_left = left
        self._region_top = top

    def set_enemy_label(self, label: int):
        self._enemy_label = label

    def start(self):
        if self._thread is not None:
            return
        self._should_stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info('overlay thread started')

    def stop(self):
        self._should_stop.set()
        try:
            self._queue.put_nowait(None)
        except QueueFull:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._cleanup()
        log.info('overlay thread stopped')

    def show(self, detections: Optional[np.ndarray]):
        try:
            self._queue.put_nowait(detections)
        except QueueFull:
            pass

    # ── 窗口管理 ──

    def _register_class(self):
        """注册窗口类。如果类已存在则忽略。"""
        hinstance = kernel32.GetModuleHandleW(None)
        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.style = 0
        wc.lpfnWndProc = self._wndproc
        wc.hInstance = hinstance
        wc.hCursor = 0
        wc.hbrBackground = 0
        wc.lpszClassName = WINDOW_CLASS
        user32.RegisterClassExW(ctypes.byref(wc))

    def _create_window(self) -> bool:
        self._register_class()
        hinstance = kernel32.GetModuleHandleW(None)

        self._hwnd = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE,
            WINDOW_CLASS, None,
            WS_POPUP,
            self._screen_x, self._screen_y,
            self._screen_w, self._screen_h,
            None, None, hinstance, None,
        )
        if not self._hwnd:
            log.error('failed to create overlay window')
            return False

        user32.ShowWindow(self._hwnd, SW_SHOW)
        user32.SetWindowPos(
            self._hwnd, -1,  # HWND_TOPMOST
            self._screen_x, self._screen_y,
            self._screen_w, self._screen_h,
            0x0010 | 0x0040,  # SWP_NOACTIVATE | SWP_SHOWWINDOW
        )

        # 创建 DIB Section（32-bit BGRA, top-down）
        screen_dc = user32.GetDC(None)
        self._mem_dc = gdi32.CreateCompatibleDC(screen_dc)

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = self._screen_w
        bmi.bmiHeader.biHeight = -self._screen_h
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB

        bits_ptr = ctypes.c_void_p()
        self._dib_hbitmap = gdi32.CreateDIBSection(
            screen_dc, ctypes.byref(bmi), DIB_RGB_COLORS,
            ctypes.byref(bits_ptr), None, 0,
        )
        if not self._dib_hbitmap or not bits_ptr.value:
            log.error('failed to create DIB section')
            user32.ReleaseDC(None, screen_dc)
            self._cleanup()
            return False

        gdi32.SelectObject(self._mem_dc, self._dib_hbitmap)
        user32.ReleaseDC(None, screen_dc)

        # 将 DIB bits 映射为 numpy 数组
        uint8_ptr = ctypes.cast(bits_ptr, ctypes.POINTER(ctypes.c_uint8))
        self._canvas = np.ctypeslib.as_array(
            uint8_ptr, shape=(self._screen_h, self._screen_w, 4),
        )

        log.info('overlay created (%dx%d)', self._screen_w, self._screen_h)
        return True

    def _cleanup(self):
        self._canvas = None
        self._thread = None
        if self._dib_hbitmap:
            gdi32.DeleteObject(self._dib_hbitmap)
            self._dib_hbitmap = None
        if self._mem_dc:
            gdi32.DeleteDC(self._mem_dc)
            self._mem_dc = None
        if self._hwnd:
            user32.DestroyWindow(self._hwnd)
            self._hwnd = None

    # ── 渲染 ──

    def _render(self, detections: Optional[np.ndarray]):
        if self._canvas is None:
            return

        self._canvas[:] = 0

        n_dets = len(detections) if detections is not None else -1
        self._render_log_counter = getattr(self, '_render_log_counter', 0) + 1
        if self._render_log_counter % 60 == 1:
            log.info('overlay render #%d: %d detections', self._render_log_counter, n_dets)

        if detections is not None and len(detections) > 0:
            ox, oy = self._region_left, self._region_top

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                x1_s = int(x1) + ox
                y1_s = int(y1) + oy
                x2_s = int(x2) + ox
                y2_s = int(y2) + oy

                box_color = (
                    (0, 255, 0, 255) if int(cls) == self._enemy_label
                    else (255, 0, 0, 255)
                )

                cv2.rectangle(self._canvas, (x1_s, y1_s), (x2_s, y2_s), box_color, 2)

                label = f'{conf:.2f}'
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)

                lx1 = max(0, x1_s)
                ly1 = max(0, y1_s - th - 4)
                lx2 = min(self._screen_w, x1_s + tw + 4)
                ly2 = min(self._screen_h, y1_s)
                if lx2 > lx1 and ly2 > ly1:
                    cv2.rectangle(
                        self._canvas,
                        (lx1, ly1), (lx2, ly2),
                        (30, 30, 30, 200), -1,
                    )

                tx = min(lx1 + 2, self._screen_w - 2)
                ty = min(ly2 - 4, self._screen_h - 2)
                cv2.putText(
                    self._canvas, label, (tx, ty),
                    FONT, 0.5, (255, 255, 255, 255), 1, cv2.LINE_AA,
                )

            # 非零 BGR → 确保 Alpha 可见
            drawn = np.any(self._canvas[:, :, :3] > 0, axis=2)
            self._canvas[drawn, 3] = np.maximum(self._canvas[drawn, 3], 200)

        self._present()

    def _present(self):
        if not self._hwnd or not self._mem_dc:
            return

        pt_zero = POINT(0, 0)
        pt_screen = POINT(self._screen_x, self._screen_y)
        size = SIZE(self._screen_w, self._screen_h)
        blend = BLENDFUNCTION(0, 0, 255, AC_SRC_ALPHA)

        user32.UpdateLayeredWindow(
            self._hwnd, None,
            ctypes.byref(pt_screen), ctypes.byref(size),
            self._mem_dc, ctypes.byref(pt_zero),
            0, ctypes.byref(blend), ULW_ALPHA,
        )

    def _loop(self):
        if not self._create_window():
            return

        # 窗口创建后立即渲染一帧空画面，使窗口透明不遮挡
        self._render(np.empty((0, 6), dtype=np.float32))

        while not self._should_stop.is_set():
            detections = self._queue.get()
            if detections is None:
                break
            self._render(detections)

        self._cleanup()
