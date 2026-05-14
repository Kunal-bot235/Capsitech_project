"""Quality gate – blur, exposure, and stuck-frame detection."""

import cv2
import numpy as np
import imagehash
from PIL import Image
from typing import Optional


class QualityCounters:
    """Per-stream rejection counters."""
    __slots__ = ("blur", "exposure", "stuck", "checked", "accepted")

    def __init__(self) -> None:
        self.blur = self.exposure = self.stuck = 0
        self.checked = self.accepted = 0

    def to_dict(self) -> dict:
        return {"blur_rejected": self.blur, "exposure_rejected": self.exposure,
                "stuck_rejected": self.stuck, "total_checked": self.checked,
                "total_accepted": self.accepted}


class QualityGate:
    """Three-check gate.  Thresholds justified in scripts/pick_thresholds.py.

    1. Blur       – variance of Laplacian < blur_thr  → reject
    2. Exposure   – mean luma outside [exp_lo, exp_hi]
                    OR clipped-pixel ratio > clip_thr  → reject
    3. Stuck      – perceptual-hash distance < hash_thr vs last accepted → reject
    """

    def __init__(self, blur_thr: float = 50.0,
                 exp_lo: float = 30.0, exp_hi: float = 235.0,
                 clip_thr: float = 0.15, hash_thr: int = 5) -> None:
        self.blur_thr = blur_thr
        self.exp_lo, self.exp_hi = exp_lo, exp_hi
        self.clip_thr = clip_thr
        self.hash_thr = hash_thr
        self._prev: dict[str, Optional[imagehash.ImageHash]] = {}
        self._ctr: dict[str, QualityCounters] = {}

    def _ensure(self, sid: str) -> QualityCounters:
        if sid not in self._ctr:
            self._ctr[sid] = QualityCounters()
            self._prev[sid] = None
        return self._ctr[sid]

    def check(self, frame: np.ndarray, source_id: str) -> bool:
        c = self._ensure(source_id)
        c.checked += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. blur
        if cv2.Laplacian(gray, cv2.CV_64F).var() < self.blur_thr:
            c.blur += 1
            return False

        # 2. exposure
        mean_l = float(gray.mean())
        n = gray.size
        clipped = (np.count_nonzero(gray < 10) +
                   np.count_nonzero(gray > 245)) / n
        if mean_l < self.exp_lo or mean_l > self.exp_hi or clipped > self.clip_thr:
            c.exposure += 1
            return False

        # 3. stuck frame
        h = imagehash.phash(Image.fromarray(cv2.cvtColor(frame,
                            cv2.COLOR_BGR2RGB)), hash_size=8)
        if self._prev[source_id] is not None:
            if (h - self._prev[source_id]) < self.hash_thr:
                c.stuck += 1
                return False
        self._prev[source_id] = h

        c.accepted += 1
        return True

    def get_counters(self, source_id: str) -> dict:
        return self._ensure(source_id).to_dict()
