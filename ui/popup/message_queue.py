"""Simple priority message queue."""
import heapq
import threading
from typing import Optional
from ui.popup.message_types import PopupMessage


class MessageQueue:
    """Thread-safe priority heap: higher priority first, then FIFO by timestamp."""
    
    def __init__(self):
        self._heap: list[PopupMessage] = []
        self._lock = threading.Lock()
        
    def push(self, message: PopupMessage) -> bool:
        """Add a message unless an identical one is queued. False if it was dropped."""
        with self._lock:
            key = message.dedup_key
            if any(queued.dedup_key == key for queued in self._heap):
                return False
            heapq.heappush(self._heap, message)
            return True

    def contains(self, message: PopupMessage) -> bool:
        """Check if an identical message (same style, title and text) is already queued."""
        with self._lock:
            key = message.dedup_key
            return any(queued.dedup_key == key for queued in self._heap)


    def pop(self) -> Optional[PopupMessage]:
        """Remove and return the highest priority message, or None if empty."""
        with self._lock:
            if self._heap:
                return heapq.heappop(self._heap)
            return None
    
    def peek(self) -> Optional[PopupMessage]:
        """Return the highest priority message without removing it."""
        with self._lock:
            if self._heap:
                return self._heap[0]
            return None
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        with self._lock:
            return len(self._heap) == 0
    
    def size(self) -> int:
        """Return the number of messages in the queue."""
        with self._lock:
            return len(self._heap)
    
    def has_higher_priority_than(self, message: PopupMessage) -> bool:
        """Check if there's a message that should be shown before the given one."""
        with self._lock:
            if self._heap:
                top = self._heap[0]
                # Higher type rank wins outright
                if top.type_rank != message.type_rank:
                    return top.type_rank > message.type_rank
                # Same type: higher priority wins
                return top.priority > message.priority
            return False