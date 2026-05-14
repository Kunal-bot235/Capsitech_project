"""Letterbox preprocessing (cv2 + numpy) and inverse transform."""

import time
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class LetterboxInfo:
    """Metadata to reverse a letterbox transform."""
    src_h: int
    src_w: int
    target: int
    scale: float
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int


# ── helpers ───────────────────────────────────────────────────
def _compute_params(h: int, w: int, size: int):
    scale = min(size / h, size / w)
    nw, nh = int(w * scale), int(h * scale)
    pl = (size - nw) // 2
    pt = (size - nh) // 2
    return scale, nw, nh, pl, pt, size - nw - pl, size - nh - pt


# ── cv2 implementation ────────────────────────────────────────
def letterbox_cv2(frame: np.ndarray, size: int = 640):
    h, w = frame.shape[:2]
    scale, nw, nh, pl, pt, pr, pb = _compute_params(h, w, size)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, pt, pb, pl, pr,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0,
                                 (size, size), swapRB=True, crop=False)
    info = LetterboxInfo(src_h=h, src_w=w, target=size, scale=scale,
                         pad_top=pt, pad_left=pl, pad_bottom=pb, pad_right=pr)
    return blob, info


# ── numpy implementation ──────────────────────────────────────
def letterbox_numpy(frame: np.ndarray, size: int = 640):
    h, w = frame.shape[:2]
    scale, nw, nh, pl, pt, pr, pb = _compute_params(h, w, size)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pt:pt + nh, pl:pl + nw] = resized
    rgb = canvas[:, :, ::-1].copy()                   # BGR → RGB
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))              # HWC → CHW
    blob = np.expand_dims(blob, 0)                     # → NCHW
    info = LetterboxInfo(src_h=h, src_w=w, target=size, scale=scale,
                         pad_top=pt, pad_left=pl, pad_bottom=pb, pad_right=pr)
    return blob, info


# ── coordinate transforms ────────────────────────────────────
def forward_transform(bbox_src, info: LetterboxInfo):
    """Source coords → letterbox coords."""
    x1, y1, x2, y2 = bbox_src
    return (x1 * info.scale + info.pad_left,
            y1 * info.scale + info.pad_top,
            x2 * info.scale + info.pad_left,
            y2 * info.scale + info.pad_top)


def inverse_transform(bbox_lb, info: LetterboxInfo):
    """Letterbox coords → source coords."""
    x1, y1, x2, y2 = bbox_lb
    return ((x1 - info.pad_left) / info.scale,
            (y1 - info.pad_top) / info.scale,
            (x2 - info.pad_left) / info.scale,
            (y2 - info.pad_top) / info.scale)


# ── benchmark utility ────────────────────────────────────────
def benchmark(frame: np.ndarray, n: int = 200) -> dict:
    t0 = time.perf_counter()
    for _ in range(n):
        letterbox_cv2(frame)
    cv2_ms = (time.perf_counter() - t0) / n * 1000

    t0 = time.perf_counter()
    for _ in range(n):
        letterbox_numpy(frame)
    np_ms = (time.perf_counter() - t0) / n * 1000

    return {"cv2_ms": round(cv2_ms, 3), "numpy_ms": round(np_ms, 3),
            "speedup_factor": round(np_ms / max(cv2_ms, 1e-9), 2)}
