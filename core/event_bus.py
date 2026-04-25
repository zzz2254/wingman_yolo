from collections import defaultdict
from typing import Any, Callable


class EventBus:

    def __init__(self):
        self._subscribers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable[..., Any]):
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[..., Any]):
        subscribers = self._subscribers.get(event_type)
        if subscribers and callback in subscribers:
            subscribers.remove(callback)

    def publish(self, event_type: str, **data: Any):
        for callback in self._subscribers.get(event_type, []):
            callback(**data)
