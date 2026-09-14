"""Frame composition helpers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from carma_vision.stereo import encode_jpeg, fit_width, side_by_side


def test_side_by_side_places_left_first() -> None:
    left = np.zeros((4, 6, 3), np.uint8)
    right = np.full((4, 6, 3), 255, np.uint8)
    sbs = side_by_side(left, right)
    assert sbs.shape == (4, 12, 3)
    assert sbs[:, :6].max() == 0
    assert sbs[:, 6:].min() == 255


def test_side_by_side_matches_heights() -> None:
    sbs = side_by_side(np.zeros((10, 10, 3), np.uint8), np.zeros((5, 5, 3), np.uint8))
    assert sbs.shape == (10, 20, 3)


def test_side_by_side_rejects_grey() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        side_by_side(np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.uint8))


def test_fit_width_downscales_only() -> None:
    img = np.zeros((100, 400, 3), np.uint8)
    assert fit_width(img, 200).shape == (50, 200, 3)
    assert fit_width(img, 800) is img


def test_jpeg_round_trip() -> None:
    img = np.full((16, 16, 3), 128, np.uint8)
    decoded = cv2.imdecode(np.frombuffer(encode_jpeg(img, 90), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == img.shape
