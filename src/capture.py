"""Camera capture thread – real-time pacing via PTS-based throttle."""

import threading
import time
import logging

import cv2

from src.slot import LatestSlot, FramePacket

logger = logging.getLogger(__name__)


class CaptureThread(threading.Thread):
    """Reads video in its own thread, paces by frame PTS, loops on EOF.

    Throttle uses Event.wait(timeout=delta) on the gap between wall-clock
    and expected presentation time.  This handles variable-FPS correctly
    because each frame's individual PTS drives the wait, not a fixed 1/fps.
    """

    def __init__(self, path: str, slot: LatestSlot,
                 source_id: str, stop: threading.Event) -> None:
        super().__init__(daemon=True, name=f"cap-{source_id}")
        self.path = path
        self.slot = slot
        self.source_id = source_id
        self.stop_event = stop
        self._lock = threading.Lock()
        self.recovery_count = 0
        self._fps_capture = 0.0
        self._frame_idx = 0

    # ── main loop ──────────────────────────────────────────────
    def run(self) -> None:
        cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("%s: cannot open %s", self.source_id, self.path)
            return

        wall_anchor = None
        pts_anchor = None
        fps_n = 0
        fps_t = time.monotonic()

        while not self.stop_event.is_set():
            ret, frame = cap.read()

            # ── EOF or decode glitch ──────────────────────────
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                wall_anchor = None
                pts_anchor = None
                with self._lock:
                    self.recovery_count += 1
                continue

            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            now = time.monotonic()

            # ── anchor on first frame after open / seek ───────
            if wall_anchor is None:
                wall_anchor = now
                pts_anchor = pts
            else:
                delay = wall_anchor + (pts - pts_anchor) - now
                if delay > 0:
                    self.stop_event.wait(timeout=delay)
                    if self.stop_event.is_set():
                        break

            grab_t = time.monotonic()
            pkt = FramePacket(frame=frame, timestamp=pts,
                              capture_wall=grab_t,
                              source_id=self.source_id,
                              frame_index=self._frame_idx)
            self._frame_idx += 1
            self.slot.put(pkt)

            fps_n += 1
            elapsed = grab_t - fps_t
            if elapsed >= 1.0:
                with self._lock:
                    self._fps_capture = fps_n / elapsed
                fps_n = 0
                fps_t = grab_t

        cap.release()
        logger.info("%s: capture exited", self.source_id)

    # ── stats for metrics emitter ─────────────────────────────
    def get_stats(self) -> dict:
        with self._lock:
            return {"fps_capture": round(self._fps_capture, 2),
                    "recovery_count": self.recovery_count}
