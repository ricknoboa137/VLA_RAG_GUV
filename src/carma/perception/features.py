"""Deterministic image cues.

Every function here takes an OpenCV-convention image (HWC, BGR, uint8) and
returns a plain float or a small float32 vector. No function allocates a model,
touches the network, or depends on call order.
"""

from __future__ import annotations

import cv2
import numpy as np

from carma.types.core import BgrImage, Embedding

_DESCRIPTOR_BINS = 16


def green_fraction(rgb: BgrImage) -> float:
    """Fraction of pixels that read as vegetation, in ``[0, 1]``.

    Uses an HSV hue window rather than an excess-green index because the window
    is stable under the exposure changes typical of a moving field robot.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 40, 30]), np.array([90, 255, 255]))
    return float(np.count_nonzero(mask)) / float(mask.size)


def texture_energy(rgb: BgrImage) -> float:
    """Normalised gradient energy, in ``[0, 1]``.

    Low values indicate bare soil or sky; high values indicate canopy. Used as
    a coarse proxy for how far the scene has moved from the bare-soil stage.
    """
    grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return float(np.clip(mag.mean() / 128.0, 0.0, 1.0))


def traversability_cue(rgb: BgrImage) -> float:
    """Heuristic traversability of the lower-centre image region, in ``[0, 1]``.

    This is a stand-in, not a contribution. It exists so the episode loop runs
    end to end before a learned traversability estimator is integrated; replace
    it behind the same signature.
    """
    h, w = rgb.shape[:2]
    patch = rgb[int(h * 0.6) : h, int(w * 0.3) : int(w * 0.7)]
    if patch.size == 0:
        return 0.0
    smoothness = 1.0 - texture_energy(patch)
    openness = 1.0 - green_fraction(patch)
    return float(np.clip(0.5 * smoothness + 0.5 * openness, 0.0, 1.0))


def scene_descriptor(rgb: BgrImage) -> Embedding:
    """A small, unit-norm colour-histogram descriptor.

    Attached to memory entries so that retrieval can later be conditioned on
    visual similarity as well as text. Deterministic and model-free.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [_DESCRIPTOR_BINS, _DESCRIPTOR_BINS], [0, 180, 0, 256])
    vec = hist.flatten().astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec
