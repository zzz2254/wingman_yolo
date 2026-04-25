import logging
import threading

import customtkinter as ctk
from pynput import keyboard

from core.config import AppConfig
from core.event_bus import EventBus
from core.state_machine import AppState, StateMachine

log = logging.getLogger(__name__)

ctk.set_appearance_mode('System')
ctk.set_default_color_theme('blue')


class MainWindow:

    STRATEGY_MAP = {
        'nearest': 'nearest（最近中心）',
        'largest': 'largest（最大威胁）',
        'crosshair': 'crosshair（光标附近）',
    }
    STRATEGY_REVERSE = {v: k for k, v in STRATEGY_MAP.items()}

    def __init__(self, config: AppConfig, event_bus: EventBus, state_machine: StateMachine):
        self._config = config
        self._event_bus = event_bus
        self._state_machine = state_machine

        self._root = ctk.CTk()
        self._root.title('wingman_yolo')
        self._root.geometry('380x420')
        self._root.resizable(False, False)
        self._root.protocol('WM_DELETE_WINDOW', self._on_close)

        self._build_ui()
        self._init_keyboard()

        self._sync_strategy_combo()
        self._event_bus.subscribe('state.changed', self._on_state_changed)
        self._event_bus.subscribe('cmd.aim_on', lambda: self._update_aim_ui(True))
        self._event_bus.subscribe('cmd.aim_off', lambda: self._update_aim_ui(False))

    def _build_ui(self):
        self._root.grid_columnconfigure(0, weight=1)

        # ── Header ──
        header = ctk.CTkLabel(
            self._root, text='wingman_yolo', font=ctk.CTkFont(size=22, weight='bold'),
        )
        header.grid(row=0, column=0, pady=(20, 4))

        subtitle = ctk.CTkLabel(
            self._root, text='AI-powered auto-aim', font=ctk.CTkFont(size=11),
            text_color=('gray60', 'gray40'),
        )
        subtitle.grid(row=1, column=0, pady=(0, 12))

        # ── Status card ──
        self._status_frame = ctk.CTkFrame(self._root, corner_radius=10)
        self._status_frame.grid(row=2, column=0, padx=30, sticky='ew')
        self._status_frame.grid_columnconfigure(1, weight=1)

        self._status_dot = ctk.CTkLabel(
            self._status_frame, text='●', font=ctk.CTkFont(size=14), text_color='gray',
        )
        self._status_dot.grid(row=0, column=0, padx=(14, 6), pady=12)

        self._status_label = ctk.CTkLabel(
            self._status_frame, text='就绪', font=ctk.CTkFont(size=13),
        )
        self._status_label.grid(row=0, column=1, padx=(0, 14), pady=12, sticky='w')

        # ── Toggles ──
        self._detect_switch = ctk.CTkSwitch(
            self._root, text='目标检测', font=ctk.CTkFont(size=13),
            command=self._on_toggle_detect,
        )
        self._detect_switch.grid(row=3, column=0, pady=(12, 0))

        self._aim_switch = ctk.CTkSwitch(
            self._root, text='自动瞄准', font=ctk.CTkFont(size=13),
            command=self._on_toggle_aim,
            state=ctk.DISABLED,
        )
        self._aim_switch.grid(row=4, column=0, pady=(6, 0))

        # ── FPS switch ──
        self._fps_switch = ctk.CTkSwitch(
            self._root, text='显示 FPS', font=ctk.CTkFont(size=12),
            command=self._on_toggle_fps,
        )
        self._fps_switch.grid(row=5, column=0, pady=(8, 0))

        # ── Strategy selector ──
        strategy_frame = ctk.CTkFrame(self._root, fg_color='transparent')
        strategy_frame.grid(row=6, column=0, pady=(8, 0))

        strategy_label = ctk.CTkLabel(
            strategy_frame, text='锁定策略', font=ctk.CTkFont(size=12),
            text_color=('gray40', 'gray60'),
        )
        strategy_label.grid(row=0, column=0, padx=(0, 8))

        self._strategy_combo = ctk.CTkComboBox(
            strategy_frame,
            values=['nearest（最近中心）', 'largest（最大威胁）', 'crosshair（光标附近）'],
            command=self._on_strategy_changed,
            width=180, height=30,
            font=ctk.CTkFont(size=12),
        )
        self._strategy_combo.grid(row=0, column=1)

	    # ── Capture resolution ──
        res_frame = ctk.CTkFrame(self._root, fg_color='transparent')
        res_frame.grid(row=7, column=0, pady=(8, 0))

        res_label = ctk.CTkLabel(
            res_frame, text='截屏区域', font=ctk.CTkFont(size=12),
            text_color=('gray40', 'gray60'),
        )
        res_label.grid(row=0, column=0, padx=(0, 8))

        self._res_combo = ctk.CTkComboBox(
            res_frame,
            values=list(self.RES_MAP.keys()),
            command=self._on_resolution_changed,
            width=180, height=30,
            font=ctk.CTkFont(size=12),
        )
        self._res_combo.grid(row=0, column=1)
        self._res_combo.set(self._size_to_res(self._config.capture_size))

        # ── Bottom row: reload + quit ──
        btn_frame = ctk.CTkFrame(self._root, fg_color='transparent')
        btn_frame.grid(row=8, column=0, pady=(14, 14))
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        self._reload_btn = ctk.CTkButton(
            btn_frame, text='重载配置', command=self._on_reload_config,
            fg_color='transparent', text_color=('gray30', 'gray60'),
            hover_color=('gray85', 'gray25'), corner_radius=8,
            font=ctk.CTkFont(size=12), height=32, width=100,
        )
        self._reload_btn.grid(row=0, column=0, padx=(0, 6))

        self._quit_btn = ctk.CTkButton(
            btn_frame, text='退出程序', command=self._on_close,
            fg_color='transparent', text_color=('gray30', 'gray60'),
            hover_color=('gray85', 'gray25'), corner_radius=8,
            font=ctk.CTkFont(size=12), height=32, width=100,
        )
        self._quit_btn.grid(row=0, column=1, padx=(6, 0))

    def _init_keyboard(self):
        def on_press(key):
            try:
                if key == keyboard.Key.page_up:
                    self._toggle_aim(True)
                elif key == keyboard.Key.page_down:
                    self._toggle_aim(False)
            except Exception:
                log.exception('keyboard callback error')

        self._keyboard_thread = threading.Thread(
            target=lambda: keyboard.Listener(on_press=on_press).run(),
            daemon=True,
        )
        self._keyboard_thread.start()

    def _on_toggle_detect(self):
        if self._detect_switch.get():
            if self._state_machine.transition(AppState.SCANNING):
                self._event_bus.publish('cmd.start_detection')
        else:
            if self._state_machine.state != AppState.IDLE:
                if self._state_machine.transition(AppState.IDLE):
                    self._event_bus.publish('cmd.stop_detection')
                    self._aim_switch.configure(state=ctk.DISABLED)
                    self._aim_switch.deselect()

    def _on_toggle_aim(self):
        if self._aim_switch.get():
            self._toggle_aim(True)
        else:
            self._toggle_aim(False)

    def _toggle_aim(self, enable: bool):
        if enable:
            if self._state_machine.state == AppState.SCANNING:
                if self._state_machine.transition(AppState.AIMING):
                    self._aim_switch.select()
                    self._event_bus.publish('cmd.aim_on')
                    self._update_aim_ui(True)
            elif self._state_machine.state == AppState.AIMING:
                self._aim_switch.select()
                self._event_bus.publish('cmd.aim_on')
                self._update_aim_ui(True)
        else:
            if self._state_machine.state == AppState.AIMING:
                self._aim_switch.deselect()
                self._event_bus.publish('cmd.aim_off')
                self._update_aim_ui(False)

    def _sync_strategy_combo(self):
        display = self.STRATEGY_MAP.get(self._config.target_strategy)
        if display:
            self._strategy_combo.set(display)

    def _on_toggle_fps(self):
        self._config.show_fps = self._fps_switch.get()
        self._event_bus.publish('config.reloaded')

    def _on_strategy_changed(self, choice: str):
        key = self.STRATEGY_REVERSE.get(choice)
        if key:
            self._config.target_strategy = key
            self._event_bus.publish('config.reloaded')
            log.info('target strategy changed to: %s', key)

    RES_MAP = {
        '480 × 480': 480,
        '540 × 540': 540,
        '640 × 640（默认）': 640,
        '720 × 720': 720,
        '800 × 800': 800,
        '960 × 960': 960,
        '1280 × 1280': 1280,
    }
    RES_REVERSE = {v: k for k, v in RES_MAP.items()}

    @staticmethod
    def _size_to_res(size: int) -> str:
        return MainWindow.RES_REVERSE.get(size, '640 × 640（默认）')

    def _on_resolution_changed(self, choice: str):
        new_size = self.RES_MAP.get(choice, 640)
        if new_size == self._config.capture_size:
            return
        self._config.capture_size = new_size
        self._event_bus.publish('cmd.resize_capture', size=new_size)
        log.info('capture resolution changed to %s (%dpx)', choice, new_size)

    def _on_reload_config(self):
        """从配置文件热加载所有支持热加载的参数。"""
        ok = self._config.reload()
        if ok:
            self._fps_switch.deselect() if not self._config.show_fps else self._fps_switch.select()
            self._sync_strategy_combo()
            self._res_combo.set(self._size_to_res(self._config.capture_size))
            self._event_bus.publish('config.reloaded')
            self._event_bus.publish('cmd.resize_capture', size=self._config.capture_size)
            log.info('config reloaded from file')
        else:
            log.warning('config reload failed (no config file)')

    def _on_close(self):
        if self._state_machine.state != AppState.IDLE:
            self._event_bus.publish('cmd.stop_detection')
        self._state_machine.transition(AppState.SHUTDOWN)
        self._event_bus.publish('cmd.shutdown')
        self._root.destroy()

    def _on_state_changed(self, old_state: AppState, new_state: AppState):
        self._root.after(0, lambda: self._apply_ui_state(new_state))

    def _apply_ui_state(self, state: AppState):
        states = {
            AppState.IDLE: {
                'detect_switch': (False, ctk.NORMAL),
                'aim_switch': (False, ctk.DISABLED),
                'status': '就绪',
                'dot_color': 'gray',
                'res_combo': ctk.NORMAL,
            },
            AppState.SCANNING: {
                'detect_switch': (True, ctk.NORMAL),
                'aim_switch': (False, ctk.NORMAL),
                'status': '目标检测运行中',
                'dot_color': '#30D158',
                'res_combo': ctk.DISABLED,
            },
            AppState.AIMING: {
                'detect_switch': (True, ctk.NORMAL),
                'aim_switch': (True, ctk.NORMAL),
                'status': '自动瞄准已激活',
                'dot_color': '#FF453A',
                'res_combo': ctk.DISABLED,
            },
        }
        ui = states.get(state)
        if not ui:
            return

        detect_on, detect_state = ui['detect_switch']
        aim_on, aim_state = ui['aim_switch']

        self._detect_switch.configure(state=detect_state)
        if detect_on:
            self._detect_switch.select()
        else:
            self._detect_switch.deselect()

        self._aim_switch.configure(state=aim_state)
        if aim_on:
            self._aim_switch.select()
        else:
            self._aim_switch.deselect()

        self._status_label.configure(text=ui['status'])
        self._status_dot.configure(text_color=ui['dot_color'])
        self._res_combo.configure(state=ui.get('res_combo', ctk.NORMAL))

    def _update_aim_ui(self, enabled: bool):
        def _apply():
            if enabled:
                self._status_label.configure(text='自动瞄准已激活')
                self._status_dot.configure(text_color='#FF453A')
            else:
                self._status_label.configure(text='目标检测运行中（瞄准已暂停）')
                self._status_dot.configure(text_color='#FF9F0A')
        self._root.after(0, _apply)

    def run(self):
        self._root.mainloop()
