"""The fine-tuned navigation SmolVLA behind the act contract.

Loads a checkpoint written by ``training/measure_train.py`` on the dataset from
``training/build_nav_dataset.py`` and answers each request with a base velocity:
``action = [linear_mps, angular_radps, stop]``, ``stop`` thresholded at 0.5.

It feeds the model exactly what training fed it: the left camera frame as
camera1 (RGB, at the recorded resolution, no resizing here), the pose as a
3-value state, and the instruction as the task. The model predicts a chunk of
10 future actions; each request uses only the first, so the policy replans
every step (the server is stateless by contract).

Runs only in the ``carma-train`` environment, with the Hub switched off.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from carma.backbone.wire import ActRequest, ActResponse, decode_image

STOP_THRESHOLD = 0.5
# Command scales used to turn the spread of sampled actions into a [0, 1] number.
_LINEAR_SCALE_MPS = 0.5
_ANGULAR_SCALE_RADPS = 0.8


class NavVlaPolicy:
    """SmolVLA fine-tuned to drive the alley."""

    def __init__(self, checkpoint: Path, device: str = "cuda") -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if not checkpoint.is_dir():
            raise FileNotFoundError(f"{checkpoint} not found; train one first (training/README.md)")
        self._torch = torch
        self._checkpoint = checkpoint
        self._device = torch.device(device)
        self._policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(self._device).eval()
        self._pre, self._post = make_pre_post_processors(
            self._policy.config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self._device)}},
        )

    @property
    def model_id(self) -> str:
        """The checkpoint folder, so a run names the weights that drove it."""
        return f"carma-nav-smolvla:{self._checkpoint.parent.parent.parent.name}/{self._checkpoint.parent.name}"

    def chunk(self, rgb_bgr: np.ndarray, pose: tuple[float, float, float], task: str, seed: int,
              samples: int = 1) -> np.ndarray:  # type: ignore[type-arg]
        """Predicted action chunks, shape (samples, horizon, 3), in command units."""
        from lerobot.policies.utils import prepare_observation_for_inference

        torch = self._torch
        observation: dict[str, Any] = {
            "observation.images.camera1": np.ascontiguousarray(
                cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            ),
            "observation.state": np.asarray(pose, dtype=np.float32),
        }
        batch = self._pre(prepare_observation_for_inference(observation, self._device, task=task))
        batch = {
            k: v.expand(samples, *v.shape[1:]).contiguous()
            if isinstance(v, torch.Tensor) and v.shape[:1] == (1,)
            else v
            for k, v in batch.items()
        }
        self._policy.reset()
        torch.manual_seed(seed)
        with torch.inference_mode():
            raw = self._policy.predict_action_chunk(batch)[..., :3]
            return self._post(raw).float().cpu().numpy().astype(np.float64)  # type: ignore[no-any-return]

    def act(self, request: ActRequest) -> ActResponse:
        """Answer one observation with the first action of the predicted chunk."""
        started = time.monotonic()
        samples = max(1, request.samples)
        chunks = self.chunk(
            decode_image(request.image), request.pose, request.instruction, request.seed, samples
        )
        first = chunks[:, 0, :]
        mean = first.mean(axis=0)
        if samples > 1:
            spread = first.std(axis=0)
            uncertainty = float(
                np.clip(max(spread[0] / _LINEAR_SCALE_MPS, spread[1] / _ANGULAR_SCALE_RADPS), 0, 1)
            )
            method = "sample_spread"
        else:
            uncertainty, method = 1.0, "none"  # one sample cannot disagree with itself
        stop = bool(mean[2] > STOP_THRESHOLD)
        return ActResponse(
            linear=0.0 if stop else float(mean[0]),
            angular=0.0 if stop else float(mean[1]),
            stop=stop,
            uncertainty=uncertainty,
            uncertainty_method=method,
            model_id=self.model_id,
            model_revision=None,
            inference_s=time.monotonic() - started,
            info={"samples": samples, "stop_score": float(mean[2])},
        )
