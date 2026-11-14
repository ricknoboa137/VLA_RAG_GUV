"""The chunk-playback backbone, exercised with a model-free sampler.

No checkpoint, no torch, no GPU: the sampler below is a deterministic function
of its inputs, which is the contract a real model has to meet.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from carma.backbone import BACKBONES, ChunkedBackbone, ChunkSampler
from carma.backbone.chunked import compose_prompt
from carma.config import ExperimentConfig, build
from carma.runtime import run_episode
from carma.runtime.episodes import synthetic_episodes
from carma.types import BackboneError, ConfigError, NavigationBackbone
from carma.types.core import Observation


class FakeSampler:
    """Turns noise into chunks by a fixed affine map, and records its calls."""

    horizon = 6
    noise_dim = 4
    proprio_dim = 4

    def __init__(self, spread: float = 0.05, stop: float = 0.0) -> None:
        self.spread = spread
        self.stop = stop
        self.calls: list[tuple[int, str, np.ndarray]] = []

    def sample(self, views, prompt, proprio, noise):  # type: ignore[no-untyped-def]
        self.calls.append((len(views), prompt, proprio.copy()))
        out = np.zeros((noise.shape[0], self.horizon, 3), dtype=np.float32)
        out[..., 0] = 0.3 + self.spread * noise[..., 0]
        out[..., 1] = (0.4 if "left" in prompt else 0.0) + self.spread * noise[..., 1]
        out[..., 2] = self.stop
        return out


def _at(obs: Observation, step: int) -> Observation:
    return replace(obs, t=0.1 * step)


def test_satisfies_the_protocols() -> None:
    sampler = FakeSampler()
    assert isinstance(sampler, ChunkSampler)
    assert isinstance(ChunkedBackbone(sampler), NavigationBackbone)


def test_action_respects_caps_and_uncertainty_range(
    observation: Observation, rng: np.random.Generator
) -> None:
    backbone = ChunkedBackbone(FakeSampler(spread=5.0), max_linear=0.6, max_angular=0.8)
    for step in range(12):
        action, uncertainty = backbone.act(_at(observation, step), rng=rng)
        assert 0.0 <= action.linear <= 0.6
        assert -0.8 <= action.angular <= 0.8
        assert 0.0 <= uncertainty <= 1.0


def _rollout(observation: Observation, seed: int) -> list[tuple[float, float, float]]:
    backbone = ChunkedBackbone(FakeSampler())
    rng = np.random.default_rng(seed)
    out = []
    for step in range(12):
        action, uncertainty = backbone.act(_at(observation, step), rng=rng)
        out.append((action.linear, action.angular, uncertainty))
    return out


def test_same_seed_gives_identical_actions(observation: Observation) -> None:
    assert _rollout(observation, 7) == _rollout(observation, 7)
    assert _rollout(observation, 7) != _rollout(observation, 8)


def test_chunk_is_played_back_then_replanned(
    observation: Observation, rng: np.random.Generator
) -> None:
    sampler = FakeSampler()
    backbone = ChunkedBackbone(sampler, replan_every=3)
    for step in range(7):
        backbone.act(_at(observation, step), rng=rng)
    assert len(sampler.calls) == 3  # planned at steps 0, 3 and 6


def test_repeat_call_in_one_step_does_not_advance(
    observation: Observation, rng: np.random.Generator
) -> None:
    sampler = FakeSampler()
    backbone = ChunkedBackbone(sampler)
    assert backbone.act(observation, rng=rng) == backbone.act(observation, rng=rng)
    assert len(sampler.calls) == 1


def test_context_replans_and_keeps_steering(
    observation: Observation, rng: np.random.Generator
) -> None:
    sampler = FakeSampler(spread=0.0)
    backbone = ChunkedBackbone(sampler, replan_every=4)

    plain, _ = backbone.act(observation, rng=rng)
    steered, _ = backbone.act(observation, ("keep left of the ditch",), rng=rng)
    assert plain.angular == pytest.approx(0.0)
    assert steered.angular == pytest.approx(0.4)
    assert sampler.calls[1][1].endswith("Operator notes: keep left of the ditch")

    # The next step arrives without context, as the episode loop always calls
    # it; the context-conditioned chunk must still be the one playing.
    following, _ = backbone.act(_at(observation, 1), rng=rng)
    assert following.angular == pytest.approx(0.4)
    assert len(sampler.calls) == 2


def test_stereo_pair_becomes_two_views(observation: Observation, rng: np.random.Generator) -> None:
    sampler = FakeSampler()
    ChunkedBackbone(sampler).act(observation, rng=rng)
    stereo = replace(observation, rgb_right=observation.rgb.copy())
    backbone = ChunkedBackbone(sampler)
    backbone.act(stereo, rng=rng)
    assert [views for views, _, _ in sampler.calls] == [1, 2]


def test_proprio_carries_the_previous_command(
    observation: Observation, rng: np.random.Generator
) -> None:
    sampler = FakeSampler(spread=0.0)
    backbone = ChunkedBackbone(sampler, replan_every=1)
    backbone.act(observation, ("turn left",), rng=rng)
    backbone.act(_at(observation, 1), rng=rng)
    assert sampler.calls[0][2].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert sampler.calls[1][2].tolist() == pytest.approx([0.3, 0.4, 0.0, 0.0])


def test_disagreement_between_samples_is_the_uncertainty(
    observation: Observation, rng: np.random.Generator
) -> None:
    _, agreed = ChunkedBackbone(FakeSampler(spread=0.0)).act(observation, rng=rng)
    _, mild = ChunkedBackbone(FakeSampler(spread=0.05)).act(observation, rng=rng)
    _, split = ChunkedBackbone(FakeSampler(spread=50.0)).act(observation, rng=rng)
    assert agreed == 0.0
    assert 0.0 < mild < split <= 1.0


def test_dispersion_matches_hand_computation(observation: Observation) -> None:
    """Two samples at +/-cap on yaw are the reference for full disagreement.

    With angular = +0.8 and -0.8 the population standard deviation is 0.8,
    which divided by the 0.8 cap is 1.0; linear agrees, so contributes 0.
    """

    class Split(FakeSampler):
        def sample(self, views, prompt, proprio, noise):  # type: ignore[no-untyped-def]
            out = np.zeros((2, self.horizon, 2), dtype=np.float32)
            out[0, :, 1], out[1, :, 1] = 0.8, -0.8
            return out

    rng = np.random.default_rng(0)
    full = ChunkedBackbone(Split(), num_samples=2, max_angular=0.8)
    half = ChunkedBackbone(Split(), num_samples=2, max_angular=0.8, dispersion_full=2.0)
    assert full.act(observation, rng=rng)[1] == pytest.approx(1.0)
    assert half.act(observation, rng=rng)[1] == pytest.approx(0.5)


def test_stop_channel_ends_the_episode(observation: Observation, rng: np.random.Generator) -> None:
    going, _ = ChunkedBackbone(FakeSampler(stop=0.1)).act(observation, rng=rng)
    done, _ = ChunkedBackbone(FakeSampler(stop=0.9)).act(observation, rng=rng)
    assert not going.stop
    assert done.stop


def test_reset_forgets_the_chunk(observation: Observation, rng: np.random.Generator) -> None:
    sampler = FakeSampler()
    backbone = ChunkedBackbone(sampler)
    backbone.act(observation, rng=rng)
    backbone.reset()
    backbone.act(observation, rng=rng)
    assert len(sampler.calls) == 2
    assert sampler.calls[1][2].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_malformed_sampler_output_raises(
    observation: Observation, rng: np.random.Generator
) -> None:
    class WrongShape(FakeSampler):
        def sample(self, views, prompt, proprio, noise):  # type: ignore[no-untyped-def]
            return np.zeros((noise.shape[0], self.horizon, 1), dtype=np.float32)

    class NotFinite(FakeSampler):
        def sample(self, views, prompt, proprio, noise):  # type: ignore[no-untyped-def]
            return np.full((noise.shape[0], self.horizon, 3), np.nan, dtype=np.float32)

    for sampler in (WrongShape(), NotFinite()):
        with pytest.raises(BackboneError):
            ChunkedBackbone(sampler).act(observation, rng=rng)


@pytest.mark.parametrize(
    "params",
    [{"num_samples": 1}, {"replan_every": 0}, {"replan_every": 7}, {"dispersion_full": 0.0}],
)
def test_invalid_parameters_are_config_errors(params: dict[str, float]) -> None:
    with pytest.raises(ConfigError):
        ChunkedBackbone(FakeSampler(), **params)  # type: ignore[arg-type]


def test_prompt_puts_the_instruction_first() -> None:
    assert compose_prompt(" follow the row ", (), "Notes:") == "follow the row"
    assert (
        compose_prompt("follow the row", ("gate is shut", " "), "Notes:")
        == "follow the row Notes: gate is shut"
    )


def test_xvla_is_registered_and_refuses_a_missing_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert "xvla" in set(BACKBONES.keys())
    with pytest.raises(ConfigError, match="checkpoint not found"):
        BACKBONES.create("xvla", model_dir=str(tmp_path / "absent"), domain_id=20)


def test_runs_deterministically_inside_the_episode_loop(smoke_config: ExperimentConfig) -> None:
    """All three arbitration paths work with chunk playback, and reproduce."""

    def fingerprint() -> list[tuple[float, float, float, str]]:
        assembly = replace(build(smoke_config), backbone=ChunkedBackbone(FakeSampler(spread=0.3)))
        spec = synthetic_episodes(1, rng=np.random.default_rng(smoke_config.seed))[0]
        record = run_episode(spec, assembly, seed=3)
        return [
            (s.action.linear, s.action.angular, s.uncertainty, s.decision.choice.value)
            for s in record.steps
        ]

    first = fingerprint()
    assert first == fingerprint()
    assert len({choice for *_, choice in first}) > 1
