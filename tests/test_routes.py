"""The row-alley route generator, the optional ``route`` config field and start poses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from carma.config import ComponentSpec, ExperimentConfig, load
from carma.runtime.episodes import row_alley_episodes
from carma.runtime.experiment import _episodes
from carma.sim.synthetic import SyntheticSim
from carma.types import ConfigError
from carma.types.core import PhenologyStage, Pose

_SMOKE = Path(__file__).resolve().parents[1] / "configs/experiment/smoke.yaml"


def _specs(count: int = 50, seed: int = 0, **kwargs: float) -> tuple:  # type: ignore[type-arg]
    return row_alley_episodes(count, rng=np.random.default_rng(seed), **kwargs)


def test_goals_lie_on_the_alley_centreline_within_the_length_range() -> None:
    for spec in _specs(min_length_m=10.0, max_length_m=20.0):
        assert spec.goal.y == 0.0
        assert 10.0 <= spec.goal.x <= 20.0


def test_starts_respect_the_offset_and_heading_bounds() -> None:
    for spec in _specs(start_offset_m=0.25, start_yaw_rad=0.2):
        assert spec.start.x == 0.0
        assert abs(spec.start.y) <= 0.25
        assert abs(spec.start.yaw) <= 0.2


def test_shortest_path_is_the_straight_line_from_start_to_goal() -> None:
    spec = _specs(1, min_length_m=12.0, max_length_m=12.0, start_offset_m=0.0)[0]
    # No lateral offset and a 12 m goal straight ahead: the path is exactly 12 m.
    assert spec.shortest_path_length == pytest.approx(12.0)
    off = _specs(1, min_length_m=12.0, max_length_m=12.0, start_offset_m=0.5)[0]
    assert off.shortest_path_length == pytest.approx(np.hypot(12.0, off.start.y))


def test_same_seed_gives_the_same_episodes_and_ids_are_stable() -> None:
    assert _specs(5, seed=3) == _specs(5, seed=3)
    assert _specs(5, seed=3) != _specs(5, seed=4)
    assert [s.episode_id for s in _specs(3)] == ["row-000", "row-001", "row-002"]
    # Adding an episode never changes the ones before it.
    assert _specs(4, seed=3)[:3] == _specs(3, seed=3)


def test_stage_and_limits_are_carried_through() -> None:
    spec = _specs(1, stage=PhenologyStage.CANOPY_CLOSURE, success_radius_m=2.5, max_steps=123)[0]
    assert spec.stage is PhenologyStage.CANOPY_CLOSURE
    assert spec.success_radius == 2.5
    assert spec.max_steps == 123


def test_route_is_optional_and_does_not_change_existing_hashes() -> None:
    cfg = load(_SMOKE)
    assert cfg.route is None
    assert "route" not in {k for k, v in cfg.resolved().items() if not (k == "route" and v is None)}
    with_route = replace(cfg, route=ComponentSpec("row_alley", {}))
    assert with_route.hash() != cfg.hash()


def test_config_loads_a_route(tmp_path: Path) -> None:
    text = _SMOKE.read_text() + "\nroute:\n  kind: row_alley\n  params:\n    max_steps: 50\n"
    path = tmp_path / "routed.yaml"
    path.write_text(text)
    cfg = load(path)
    assert cfg.route == ComponentSpec("row_alley", {"max_steps": 50})
    specs = _episodes(replace(cfg, episodes=2))
    assert [s.episode_id for s in specs] == ["row-000", "row-001"]
    assert all(s.max_steps == 50 for s in specs)


def test_an_unknown_route_kind_is_a_config_error() -> None:
    cfg: ExperimentConfig = replace(load(_SMOKE), route=ComponentSpec("loop", {}))
    with pytest.raises(ConfigError, match="loop"):
        _episodes(cfg)


def test_the_route_takes_its_stage_from_the_sim_section() -> None:
    cfg = load(_SMOKE)
    sim = ComponentSpec(cfg.sim.kind, {**cfg.sim.params, "stage": "CANOPY_CLOSURE"})
    routed = replace(cfg, sim=sim, route=ComponentSpec("row_alley", {}))
    assert {s.stage for s in _episodes(routed)} == {PhenologyStage.CANOPY_CLOSURE}


def test_the_synthetic_sim_starts_where_the_episode_says() -> None:
    sim = SyntheticSim()
    obs = sim.reset(seed=0, start=Pose(1.5, -0.25, 0.1))
    assert obs.pose == Pose(1.5, -0.25, 0.1)


def test_a_collision_ends_the_episode_as_a_failure() -> None:
    """A sim that reports a collision on its third step stops the episode there."""
    from carma.config import build
    from carma.runtime.episode import run_episode
    from carma.types.core import Action, Observation

    class Crashing(SyntheticSim):
        steps = 0

        def step(self, action: Action) -> Observation:
            Crashing.steps += 1
            return replace(super().step(action), collided=Crashing.steps == 3)

    cfg = replace(load(_SMOKE), route=ComponentSpec("row_alley", {}))
    assembly = replace(build(cfg), sim=Crashing())
    spec = _episodes(replace(cfg, episodes=1))[0]
    record = run_episode(spec, assembly, seed=0, run_id="t")
    assert record.collided
    assert not record.success
    assert Crashing.steps == 3
