"""Frame composition and encoding. No ROS imports, so it is unit-testable anywhere."""

from __future__ import annotations

import cv2
import numpy as np


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Join a stereo pair left|right, the layout VR players call SBS.

    The right image is resized to the left image's height when they differ, so
    a pair from mismatched sensors still produces one rectangular frame.
    """
    for name, img in (("left", left), ("right", right)):
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            msg = f"{name} image must be HxWx3 uint8, got {img.shape} {img.dtype}"
            raise ValueError(msg)
    if right.shape[0] != left.shape[0]:
        scale = left.shape[0] / right.shape[0]
        right = cv2.resize(right, (round(right.shape[1] * scale), left.shape[0]))
    return np.hstack((left, right))


def fit_width(img: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale to at most ``max_width`` pixels wide, keeping aspect ratio."""
    if max_width <= 0 or img.shape[1] <= max_width:
        return img
    scale = max_width / img.shape[1]
    size = (max_width, max(1, round(img.shape[0] * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    """Encode a BGR image as JPEG; ``quality`` is 1-100."""
    if not 1 <= quality <= 100:
        msg = f"jpeg quality must be 1-100, got {quality}"
        raise ValueError(msg)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        msg = "JPEG encoding failed"
        raise ValueError(msg)
    return buf.tobytes()
