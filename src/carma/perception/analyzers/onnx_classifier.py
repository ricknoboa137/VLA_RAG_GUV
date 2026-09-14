"""Run any image classifier exported to ONNX.

This is how a model you train yourself (PyTorch, TensorFlow, Keras, scikit-learn
via skl2onnx) enters the system without new code: export it to ``.onnx``, put
it under ``assets/``, and add an ``onnx_classifier`` entry to an analyzer
config. ONNX Runtime is open source (MIT) and runs on CPU with no account.

The runtime is an optional dependency: ``pip install -e ".[models]"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from carma.perception.registry import ANALYZERS
from carma.types import AnalyzerError
from carma.types.core import AnalysisResult, BgrImage, Observation

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_image(
    bgr: BgrImage,
    width: int,
    height: int,
    mean: Sequence[float] = _IMAGENET_MEAN,
    std: Sequence[float] = _IMAGENET_STD,
    channel_order: str = "rgb",
) -> npt.NDArray[np.float32]:
    """Resize and normalise an image into a ``(1, 3, height, width)`` batch.

    Pixels are scaled to ``[0, 1]`` and then normalised per channel as
    ``(x - mean) / std``, with ``mean`` and ``std`` given in the model's channel
    order. The BGR-to-RGB conversion happens here, at the model boundary, as
    ``AGENTS.md`` section 7 requires.
    """
    if channel_order not in ("rgb", "bgr"):
        msg = f"channel_order must be 'rgb' or 'bgr', got {channel_order!r}"
        raise AnalyzerError(msg)
    resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    if channel_order == "rgb":
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    scaled = resized.astype(np.float32) / 255.0
    scaled = (scaled - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(scaled.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)


@ANALYZERS.register("onnx_classifier")
class OnnxClassifier:
    """Single-label image classifier backed by ONNX Runtime.

    Args:
        model_path: Path to the ``.onnx`` file, normally under ``assets/``.
        labels: Class names in the order of the model's output vector.
        input_width: Model input width in pixels.
        input_height: Model input height in pixels.
        mean: Per-channel normalisation mean, in the model's channel order.
        std: Per-channel normalisation standard deviation.
        channel_order: ``"rgb"`` or ``"bgr"``, whichever the model was trained on.
        apply_softmax: Set false when the model already outputs probabilities.
        name: Name reported in results, so two classifiers can run side by side.
    """

    def __init__(
        self,
        model_path: str,
        labels: Sequence[str],
        input_width: int = 224,
        input_height: int = 224,
        mean: Sequence[float] = _IMAGENET_MEAN,
        std: Sequence[float] = _IMAGENET_STD,
        channel_order: str = "rgb",
        apply_softmax: bool = True,
        name: str = "onnx_classifier",
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            msg = f"ONNX model not found: {path}. Fetch or export it into assets/ first."
            raise AnalyzerError(msg)
        if not labels:
            msg = "onnx_classifier needs at least one label"
            raise AnalyzerError(msg)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            msg = 'onnxruntime is not installed; run pip install -e ".[models]"'
            raise AnalyzerError(msg) from exc

        self._session: Any = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input_name: str = self._session.get_inputs()[0].name
        self._labels = tuple(labels)
        self._width = input_width
        self._height = input_height
        self._mean = tuple(mean)
        self._std = tuple(std)
        self._channel_order = channel_order
        self._apply_softmax = apply_softmax
        self._name = name

    @property
    def name(self) -> str:
        """Name reported in results."""
        return self._name

    def analyze(self, obs: Observation) -> AnalysisResult:
        """Classify the left image of ``obs``; scores are class probabilities."""
        batch = preprocess_image(
            obs.rgb, self._width, self._height, self._mean, self._std, self._channel_order
        )
        outputs = self._session.run(None, {self._input_name: batch})
        logits = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        if logits.size != len(self._labels):
            msg = f"model returned {logits.size} outputs for {len(self._labels)} labels"
            raise AnalyzerError(msg)
        if self._apply_softmax:
            shifted = np.exp(logits - logits.max())
            probs = shifted / shifted.sum()
        else:
            probs = logits
        return AnalysisResult(
            analyzer=self._name,
            t=obs.t,
            pose=obs.pose,
            label=self._labels[int(np.argmax(probs))],
            scores={label: float(p) for label, p in zip(self._labels, probs, strict=True)},
        )
