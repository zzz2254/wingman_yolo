from enum import Enum, auto

from core.event_bus import EventBus


class AppState(Enum):
    IDLE = auto()
    SCANNING = auto()
    AIMING = auto()
    SHUTDOWN = auto()


_TRANSITIONS = {
    AppState.IDLE: {AppState.SCANNING},
    AppState.SCANNING: {AppState.IDLE, AppState.AIMING, AppState.SHUTDOWN},
    AppState.AIMING: {AppState.SCANNING, AppState.IDLE, AppState.SHUTDOWN},
    AppState.SHUTDOWN: set(),
}


class StateMachine:

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._state = AppState.IDLE

    @property
    def state(self) -> AppState:
        return self._state

    def transition(self, new_state: AppState):
        if new_state == self._state:
            return False
        if new_state not in _TRANSITIONS.get(self._state, set()):
            return False
        old_state = self._state
        self._state = new_state
        self._event_bus.publish('state.changed', old_state=old_state, new_state=new_state)
        return True

    def is_aiming(self) -> bool:
        return self._state == AppState.AIMING
