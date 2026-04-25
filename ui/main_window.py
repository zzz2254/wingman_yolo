import logging
import threading
import tkinter as tk

from pynput import keyboard

from core.config import AppConfig
from core.event_bus import EventBus
from core.state_machine import AppState, StateMachine

log = logging.getLogger(__name__)


class MainWindow:

    def __init__(self, config: AppConfig, event_bus: EventBus, state_machine: StateMachine):
        self._config = config
        self._event_bus = event_bus
        self._state_machine = state_machine

        self._root = tk.Tk()
        self._root.title('AL_Yolo')
        self._root.geometry('360x200')
        self._root.protocol('WM_DELETE_WINDOW', self._on_close)

        self._init_buttons()
        self._init_keyboard()

        self._event_bus.subscribe('state.changed', self._on_state_changed)

    def _init_buttons(self):
        self._btn_start_detect = tk.Button(
            self._root, text='开启目标检测', command=self._cmd_start_detect,
        )
        self._btn_start_detect.grid(row=0, column=0, padx=5, pady=5)

        self._btn_stop_detect = tk.Button(
            self._root, text='停止目标检测', command=self._cmd_stop_detect,
            state=tk.DISABLED,
        )
        self._btn_stop_detect.grid(row=1, column=0, padx=5, pady=5)

        self._btn_aim_on = tk.Button(
            self._root, text='开启鼠标锁定', command=self._cmd_aim_on,
            state=tk.DISABLED,
        )
        self._btn_aim_on.grid(row=0, column=1, padx=5, pady=5)

        self._btn_aim_off = tk.Button(
            self._root, text='暂停鼠标锁定', command=self._cmd_aim_off,
            state=tk.DISABLED,
        )
        self._btn_aim_off.grid(row=1, column=1, padx=5, pady=5)

        self._btn_exit = tk.Button(
            self._root, text='退出', command=self._on_close,
        )
        self._btn_exit.grid(row=2, column=0, columnspan=2, padx=5, pady=10)

        self._status_label = tk.Label(self._root, text='就绪', fg='gray')
        self._status_label.grid(row=3, column=0, columnspan=2)

    def _init_keyboard(self):
        def on_press(key):
            try:
                if key == keyboard.Key.page_up:
                    self._cmd_aim_on()
                elif key == keyboard.Key.page_down:
                    self._cmd_aim_off()
            except Exception:
                log.exception('keyboard callback error')

        self._keyboard_thread = threading.Thread(
            target=lambda: keyboard.Listener(on_press=on_press).run(),
            daemon=True,
        )
        self._keyboard_thread.start()

    def _cmd_start_detect(self):
        if self._state_machine.transition(AppState.SCANNING):
            self._event_bus.publish('cmd.start_detection')

    def _cmd_stop_detect(self):
        if self._state_machine.transition(AppState.IDLE):
            self._event_bus.publish('cmd.stop_detection')

    def _cmd_aim_on(self):
        if self._state_machine.state == AppState.SCANNING:
            if self._state_machine.transition(AppState.AIMING):
                self._event_bus.publish('cmd.aim_on')

    def _cmd_aim_off(self):
        if self._state_machine.state == AppState.AIMING:
            if self._state_machine.transition(AppState.SCANNING):
                self._event_bus.publish('cmd.aim_off')

    def _on_close(self):
        if self._state_machine.state != AppState.IDLE:
            self._event_bus.publish('cmd.stop_detection')
        self._state_machine.transition(AppState.SHUTDOWN)
        self._event_bus.publish('cmd.shutdown')
        self._root.destroy()

    def _on_state_changed(self, old_state: AppState, new_state: AppState):
        self._root.after(0, lambda: self._update_ui_state(new_state))

    def _update_ui_state(self, state: AppState):
        states = {
            AppState.IDLE: {
                'btn_start_detect': tk.NORMAL,
                'btn_stop_detect': tk.DISABLED,
                'btn_aim_on': tk.DISABLED,
                'btn_aim_off': tk.DISABLED,
                'status': '就绪',
                'color': 'gray',
            },
            AppState.SCANNING: {
                'btn_start_detect': tk.DISABLED,
                'btn_stop_detect': tk.NORMAL,
                'btn_aim_on': tk.NORMAL,
                'btn_aim_off': tk.DISABLED,
                'status': '目标检测运行中',
                'color': 'green',
            },
            AppState.AIMING: {
                'btn_start_detect': tk.DISABLED,
                'btn_stop_detect': tk.NORMAL,
                'btn_aim_on': tk.DISABLED,
                'btn_aim_off': tk.NORMAL,
                'status': '鼠标锁定已激活',
                'color': 'red',
            },
        }
        ui = states.get(state)
        if ui:
            self._btn_start_detect.config(state=ui['btn_start_detect'])
            self._btn_stop_detect.config(state=ui['btn_stop_detect'])
            self._btn_aim_on.config(state=ui['btn_aim_on'])
            self._btn_aim_off.config(state=ui['btn_aim_off'])
            self._status_label.config(text=ui['status'], fg=ui['color'])

    def run(self):
        self._root.mainloop()
