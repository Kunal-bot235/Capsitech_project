"""Consumer thread – reads slots, quality-gates, preprocesses."""

import threading
import time
import logging

from src.slot import LatestSlot
from src.quality import QualityGate
from src.preprocess import letterbox_cv2, letterbox_numpy
from src.metrics import StreamMetrics

logger = logging.getLogger(__name__)


class ConsumerThread(threading.Thread):
    """Single consumer polling all capture slots."""

    def __init__(self, slots: dict[str, LatestSlot],
                 qg: QualityGate,
                 sm: dict[str, StreamMetrics],
                 stop: threading.Event,
                 use_numpy: bool = False) -> None:
        super().__init__(daemon=True, name="consumer")
        self.slots = slots
        self.qg = qg
        self.sm = sm
        self.stop_event = stop
        self._prep = letterbox_numpy if use_numpy else letterbox_cv2

    def run(self) -> None:
        while not self.stop_event.is_set():
            got = False
            for sid, slot in self.slots.items():
                pkt = slot.get()
                if pkt is None:
                    continue
                got = True
                age = (time.monotonic() - pkt.capture_wall) * 1000
                self.sm[sid].record_age(age)
                if not self.qg.check(pkt.frame, sid):
                    continue
                self._prep(pkt.frame)
                self.sm[sid].record_preprocess()
            if not got:
                self.stop_event.wait(timeout=0.001)
        logger.info("consumer exited")
