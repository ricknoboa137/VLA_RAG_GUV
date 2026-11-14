"""Adapter for X-VLA, a 0.9B soft-prompted flow-matching VLA.

X-VLA (Zheng et al., arXiv 2510.10274; code and weights Apache-2.0) encodes
images and language with the Florence-2 encoder (MIT) and denoises an action
chunk with a transformer conditioned on a per-embodiment soft prompt, selected
by ``domain_id``.

The released checkpoints are manipulation policies. Driving this robot needs a
checkpoint fine-tuned on its own data, with a new ``domain_id`` and actions
laid out as ``carma.backbone.chunked`` expects. ``docs/backbones.md`` states
that contract and what has and has not been verified.

Requires the optional ``vla`` extra and a checkpoint already present in
``assets/``. It never downloads: fetching is an explicit step run by
``scripts/fetch_models.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from carma.backbone.chunked import Chunk, ChunkedBackbone
from carma.backbone.registry import BACKBONES
from carma.types import ConfigError
from carma.types.core import BgrImage


class XvlaSampler:
    """Runs X-VLA's denoising loop on noise supplied by the caller.

    The loop is the one in the checkpoint's own ``generate_actions``, with two
    differences: the initial noise is an argument rather than a ``torch.randn``
    draw, which is what makes a run reproducible from its seed, and the
    vision-language encoder runs once for all samples instead of once each.

    Args:
        model_dir: Directory holding the checkpoint, normally under ``assets/``.
        domain_id: Index of the soft prompt for this robot and camera setup.
        denoising_steps: Flow-matching integration steps.
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self, model_dir: str, *, domain_id: int, denoising_steps: int, device: str
    ) -> None:
        path = Path(model_dir)
        if not (path / "config.json").is_file():
            msg = (
                f"X-VLA checkpoint not found in {path}; "
                "run scripts/fetch_models.py or place a fine-tuned checkpoint there"
            )
            raise ConfigError(msg)
        if denoising_steps < 1:
            msg = "denoising_steps must be at least 1"
            raise ConfigError(msg)
        try:
            import torch
            from PIL import Image
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional extra
            msg = 'install the vla extra to use the xvla backbone: pip install -e ".[vla]"'
            raise ConfigError(msg) from exc

        # The checkpoint ships its own model code, hence trust_remote_code. It
        # is loaded from disk only; scripts/fetch_models.py pins the revision.
        load = {"trust_remote_code": True, "local_files_only": True}
        self._torch: Any = torch
        self._image: Any = Image
        self._model: Any = AutoModel.from_pretrained(str(path), **load).to(device).eval()
        self._processor: Any = AutoProcessor.from_pretrained(str(path), **load)
        self._device = device
        self._steps = denoising_steps

        num_domains = int(self._model.config.num_domains)
        if not 0 <= domain_id < num_domains:
            msg = f"domain_id must be in [0, {num_domains}), got {domain_id}"
            raise ConfigError(msg)
        self._domain_id = domain_id

    @property
    def horizon(self) -> int:
        """Number of control steps in one chunk."""
        return int(self._model.num_actions)

    @property
    def noise_dim(self) -> int:
        """Width of the action vector the transformer denoises."""
        return int(self._model.action_space.dim_action)

    @property
    def proprio_dim(self) -> int:
        """Width of the proprioceptive state vector."""
        return int(getattr(self._model.action_space, "dim_proprio", self.noise_dim))

    def sample(
        self,
        views: tuple[BgrImage, ...],
        prompt: str,
        proprio: npt.NDArray[np.float32],
        noise: Chunk,
    ) -> Chunk:
        """Denoise ``noise`` of shape ``(k, horizon, noise_dim)`` into action chunks."""
        torch = self._torch
        model = self._model
        k = noise.shape[0]

        # The model was trained on RGB; the conversion happens here, at its boundary.
        images = [self._image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)) for v in views]
        inputs = self._processor(images, prompt)
        dtype = next(model.parameters()).dtype

        with torch.inference_mode():
            enc = model.forward_vlm(
                inputs["input_ids"].to(self._device),
                inputs["image_input"].to(device=self._device, dtype=dtype),
                inputs["image_mask"].to(self._device),
            )
            enc = {name: feats.expand(k, *feats.shape[1:]) for name, feats in enc.items()}
            state = torch.from_numpy(proprio).to(device=self._device, dtype=dtype)
            state = state.unsqueeze(0).expand(k, -1)
            domain = torch.full((k,), self._domain_id, dtype=torch.long, device=self._device)

            x1 = torch.from_numpy(noise).to(device=self._device, dtype=dtype)
            action = torch.zeros_like(x1)
            for i in range(self._steps, 0, -1):
                t = torch.full((k,), i / self._steps, device=self._device, dtype=dtype)
                x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
                state_m, x_t_m = model.action_space.preprocess(state, x_t)
                action = model.transformer(
                    domain_id=domain,
                    action_with_noise=x_t_m,
                    proprio=state_m,
                    t=t,
                    **enc,
                )
            out = model.action_space.postprocess(action)
        return np.asarray(out.float().cpu().numpy(), dtype=np.float32)


@BACKBONES.register("xvla")
class XvlaBackbone(ChunkedBackbone):
    """X-VLA behind the chunk-playback backbone.

    Args:
        model_dir: Directory holding the checkpoint, normally under ``assets/``.
        domain_id: Index of the soft prompt for this robot and camera setup.
        denoising_steps: Flow-matching integration steps.
        device: Torch device string.
        max_linear: Forward speed cap in metres per second.
        max_angular: Yaw rate cap in radians per second.
        num_samples: Chunks sampled per plan; see :class:`ChunkedBackbone`.
        replan_every: Control steps between plans.
        dispersion_full: Normalised dispersion that maps to uncertainty 1.0.
        stop_threshold: Stop-channel value above which the goal is declared.
    """

    def __init__(
        self,
        model_dir: str,
        domain_id: int,
        denoising_steps: int = 10,
        device: str = "cuda",
        max_linear: float = 0.6,
        max_angular: float = 0.8,
        num_samples: int = 8,
        replan_every: int = 5,
        dispersion_full: float = 1.0,
        stop_threshold: float = 0.5,
    ) -> None:
        sampler = XvlaSampler(
            model_dir, domain_id=domain_id, denoising_steps=denoising_steps, device=device
        )
        super().__init__(
            sampler,
            max_linear=max_linear,
            max_angular=max_angular,
            num_samples=num_samples,
            replan_every=replan_every,
            dispersion_full=dispersion_full,
            stop_threshold=stop_threshold,
        )
