import threading
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp: float
    capture_wall: float
    source_id: str
    frame_index: int

class LatestSlot:
    """A thread-safe single-item buffer. Overwrites older items if full."""
    def __init__(self):
        self._lock = threading.Lock()
        self._pkt: Optional[FramePacket] = None
        self.frames_dropped = 0

    def put(self, pkt: FramePacket) -> None:
        with self._lock:
            if self._pkt is not None:
                self.frames_dropped += 1
            self._pkt = pkt

    def get(self) -> Optional[FramePacket]:
        with self._lock:
            p = self._pkt
            self._pkt = None
            return p
