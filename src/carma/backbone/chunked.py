"""Backbones that plan a chunk of actions and play it back.

Flow-matching and diffusion policies do not emit one action per call. They
denoise a whole sequence, a *chunk*, from random noise. This module turns any
such model into a :class:`~carma.types.NavigationBackbone`: it draws the noise
from the caller's generator, plays the chunk back one control step at a time,
and reports how much independently sampled chunks disagree as the uncertainty.

A model enters by implementing :class:`ChunkSampler`. Everything the arbiter
sees (caps, playback, uncertainty, prompt composition) lives here, so two
models compared under one config differ only in the sampler. See
``docs/backbones.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from carma.types import BackboneError, ConfigError
from carma.types.core import Action, BgrImage, Observation

Chunk = npt.NDArray[np.float32]

# Channel layout of a navigation chunk. A model fine-tuned for this project
# must be trained on actions laid out this way; see ``docs/backbones.md``.
LINEAR = 0
ANGULAR = 1
STOP = 2


@runtime_checkable
class ChunkSampler(Protocol):
    """A generative policy that maps noise to chunks of actions.

    Internal to ``carma.backbone``: it is the seam between the arbitration-facing
    behaviour in :class:`ChunkedBackbone` and one model's inference code, not a
    component named in config.
    """

    @property
    def horizon(self) -> int:
        """Number of control steps in one chunk."""
        ...

    @property
    def noise_dim(self) -> int:
        """Width of the noise the model denoises, per step."""
        ...

    @property
    def proprio_dim(self) -> int:
        """Width of the proprioceptive state vector the model expects."""
        ...

    def sample(
        self,
        views: tuple[BgrImage, ...],
        prompt: str,
        proprio: npt.NDArray[np.float32],
        noise: Chunk,
    ) -> Chunk:
        """Denoise ``noise`` into action chunks.

        Args:
            views: Camera images, HWC, BGR, uint8. The first is the primary
                view; for a stereo camera it is the left image.
            prompt: The instruction, with any retrieved context appended.
            proprio: State vector of length ``proprio_dim``.
            noise: Standard-normal array of shape ``(k, horizon, noise_dim)``.
                The sampler must draw no randomness of its own.

        Returns:
            Array of shape ``(k, horizon, channels)`` with ``channels >= 2``:
            forward velocity in metres per second, yaw rate in radians per
            second, and optionally a stop indicator in ``[0, 1]``.
        """
        ...


def compose_prompt(instruction: str, context: tuple[str, ...], prefix: str) -> str:
    """Join the instruction and retrieved context into one model prompt.

    The instruction comes first so that a model with a short text window
    truncates context, never the instruction.
    """
    base = instruction.strip()
    notes = "; ".join(text.strip() for text in context if text.strip())
    if not notes:
        return base
    return f"{base} {prefix} {notes}"


class ChunkedBackbone:
    """Plays back sampled action chunks and measures their disagreement.

    Args:
        sampler: The model.
        max_linear: Forward speed cap in metres per second.
        max_angular: Yaw rate cap in radians per second.
        num_samples: Chunks sampled per plan. The first is executed; all of
            them feed the uncertainty. At least 2.
        replan_every: Control steps a chunk is played for before a new one is
            sampled. At most the sampler's horizon.
        dispersion_full: Normalised dispersion that maps to uncertainty 1.0.
            Dimensionless; see :meth:`act`. Fit it on held-out data.
        stop_threshold: Stop-channel value above which the executed chunk
            declares the goal reached.
        context_prefix: Text placed between the instruction and the context.
    """

    def __init__(
        self,
        sampler: ChunkSampler,
        *,
        max_linear: float = 0.6,
        max_angular: float = 0.8,
        num_samples: int = 8,
        replan_every: int = 5,
        dispersion_full: float = 1.0,
        stop_threshold: float = 0.5,
        context_prefix: str = "Operator notes:",
    ) -> None:
        if max_linear <= 0.0 or max_angular <= 0.0:
            msg = "max_linear and max_angular must be positive"
            raise ConfigError(msg)
        if num_samples < 2:
            msg = "num_samples must be at least 2; uncertainty is the spread between samples"
            raise ConfigError(msg)
        if not 1 <= replan_every <= sampler.horizon:
            msg = f"replan_every must be in [1, {sampler.horizon}], got {replan_every}"
            raise ConfigError(msg)
        if dispersion_full <= 0.0:
            msg = "dispersion_full must be positive"
            raise ConfigError(msg)

        self._sampler = sampler
        self._max_linear = max_linear
        self._max_angular = max_angular
        self._num_samples = num_samples
        self._replan_every = replan_every
        self._dispersion_full = dispersion_full
        self._stop_threshold = stop_threshold
        self._context_prefix = context_prefix

        self._chunks: npt.NDArray[np.float64] | None = None
        self._chunk_context: tuple[str, ...] = ()
        self._uncertainty = 0.0
        self._cursor = 0
        self._step_t: float | None = None
        self._emitted = (0.0, 0.0)
        self._previous = (0.0, 0.0)

    def reset(self) -> None:
        """Clear per-episode state."""
        self._chunks = None
        self._chunk_context = ()
        self._uncertainty = 0.0
        self._cursor = 0
        self._step_t = None
        self._emitted = (0.0, 0.0)
        self._previous = (0.0, 0.0)

    def act(
        self,
        obs: Observation,
        context: tuple[str, ...] = (),
        *,
        rng: np.random.Generator,
    ) -> tuple[Action, float]:
        """Return an action and an uncertainty in ``[0, 1]``.

        A control step is identified by ``obs.t``: a second call at the same
        time is a re-proposal for the same step and does not advance playback.
        A new chunk is sampled when none exists, when the current one has been
        played for ``replan_every`` steps, or when non-empty ``context``
        arrives that differs from what the current chunk was conditioned on. A
        chunk conditioned on context therefore keeps steering until it expires.

        The uncertainty is the standard deviation across samples of the
        commanded velocities, each divided by its cap, taking the larger of the
        two channels, averaged over the steps that will be executed, divided by
        ``dispersion_full`` and clipped. It is constant while a chunk plays.
        """
        if self._step_t is None or obs.t != self._step_t:
            if self._step_t is not None:
                self._previous = self._emitted
                self._cursor += 1
            self._step_t = obs.t

        stale = self._chunks is None or self._cursor >= self._replan_every
        redirected = bool(context) and context != self._chunk_context
        if stale or redirected:
            self._chunks = self._plan(obs, context, rng)
            self._chunk_context = context
            self._cursor = 0
        chunks = self._chunks
        assert chunks is not None

        executed = chunks[0, self._cursor]
        stop = chunks.shape[2] > STOP and float(executed[STOP]) > self._stop_threshold
        self._emitted = (float(executed[LINEAR]), float(executed[ANGULAR]))
        action = Action(linear=self._emitted[0], angular=self._emitted[1], stop=stop)
        return action, self._uncertainty

    def _plan(
        self, obs: Observation, context: tuple[str, ...], rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """Sample ``num_samples`` chunks, cap them, and score their spread."""
        sampler = self._sampler
        views = (obs.rgb,) if obs.rgb_right is None else (obs.rgb, obs.rgb_right)
        proprio = np.zeros(sampler.proprio_dim, dtype=np.float32)
        proprio[: min(2, proprio.size)] = self._previous[: proprio.size]
        shape = (self._num_samples, sampler.horizon, sampler.noise_dim)
        noise = rng.standard_normal(shape, dtype=np.float32)

        prompt = compose_prompt(obs.instruction, context, self._context_prefix)
        raw = np.asarray(sampler.sample(views, prompt, proprio, noise), dtype=np.float32)
        if raw.ndim != 3 or raw.shape[:2] != shape[:2] or raw.shape[2] <= ANGULAR:
            msg = f"sampler returned shape {raw.shape}, expected {shape[:2]} + (>=2,)"
            raise BackboneError(msg)
        if not np.isfinite(raw).all():
            msg = "sampler returned non-finite actions"
            raise BackboneError(msg)

        # Capped and scored in float64 so the caps hold exactly and identical
        # samples score exactly zero.
        chunks = raw.astype(np.float64)
        chunks[..., LINEAR] = np.clip(chunks[..., LINEAR], 0.0, self._max_linear)
        chunks[..., ANGULAR] = np.clip(chunks[..., ANGULAR], -self._max_angular, self._max_angular)

        caps = np.asarray([self._max_linear, self._max_angular], dtype=np.float64)
        window = chunks[:, : self._replan_every, : ANGULAR + 1]
        dispersion = float((window.std(axis=0) / caps).max(axis=1).mean())
        self._uncertainty = float(np.clip(dispersion / self._dispersion_full, 0.0, 1.0))
        return chunks
