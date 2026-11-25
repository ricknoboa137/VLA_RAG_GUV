"""Hidden hazards: routes, the scripted operator, spatial retrieval and scoring."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from carma.memory.embedders.hashing import HashingEmbedder
from carma.memory.retriever import SpatialRetriever
from carma.memory.stores.in_memory import InMemoryStore
from carma.metrics import hazard_rate, success_rate
from carma.operator.scripted import ScriptedOperator
from carma.runtime.episodes import row_alley_episodes
from carma.types.core import (
    Correction,
    EpisodeSpec,
    Hazard,
    Observation,
    OperatorQuery,
    PhenologyStage,
    Pose,
)
from conftest import make_episode

_HAZARD = Hazard(Pose(8.0, 0.0, 0.0), 1.0, "wet patch at 8 metres: stop at 6 metres")


def _obs(x: float) -> Observation:
    return Observation(
        rgb=np.zeros((4, 4, 3), np.uint8),
        depth=None,
        pose=Pose(x, 0.0, 0.0),
        t=0.0,
        instruction="drive 13 metres down the row and stop",
        stage=PhenologyStage.VEGETATIVE,
    )


def _spec(hazards: tuple[Hazard, ...]) -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="e",
        instruction="drive 13 metres down the row and stop",
        start=Pose(0.0, 0.0, 0.0),
        goal=Pose(6.0, 0.0, 0.0),
        success_radius=1.5,
        shortest_path_length=6.0,
        stage=PhenologyStage.VEGETATIVE,
        hazards=hazards,
    )


# ---------------------------------------------------------------------- routes


def test_without_hazards_the_routes_are_exactly_as_before() -> None:
    plain = row_alley_episodes(6, rng=np.random.default_rng(3))
    zero = row_alley_episodes(6, rng=np.random.default_rng(3), hazard_probability=0.0)
    assert plain == zero
    assert all(s.hazards == () for s in plain)


def test_a_hazard_moves_the_goal_to_the_stop_point() -> None:
    specs = row_alley_episodes(
        8, rng=np.random.default_rng(0), hazard_probability=1.0, min_length_m=10.0
    )
    for s in specs:
        (h,) = s.hazards
        length = float(s.instruction.split()[1])
        assert 6.0 <= h.position.x <= length - 1.0
        assert s.goal == Pose(h.position.x - 2.0, 0.0, 0.0)
        assert h.advice == (
            f"wet patch at {h.position.x:.0f} metres: stop at {h.position.x - 2.0:.0f} metres"
        )


# --------------------------------------------------------------------- scoring


def test_entering_a_hazard_is_a_failure_and_is_counted() -> None:
    """Two episodes inside the radius, one entered a hazard: success 0.5, hazard rate 0.5."""
    clean = make_episode(final=Pose(1.0, 0.0, 0.0), goal=Pose(0.0, 0.0, 0.0), path_length=10.0)
    wet = replace(clean, entered_hazard=True)
    assert not wet.success
    assert success_rate([clean, wet]) == pytest.approx(0.5)
    assert hazard_rate([clean, wet]) == pytest.approx(0.5)
    assert hazard_rate([]) == 0.0


# -------------------------------------------------------------------- operator


def test_asked_near_a_hazard_the_operator_gives_its_advice() -> None:
    op = ScriptedOperator(refusal_rate=0.0, seed=1)
    op.begin_episode(_spec((_HAZARD,)))
    # 3 m before the hazard: within the 6 m advice range.
    near = op.ask(OperatorQuery(question="?", observation=_obs(5.0)))
    assert near.text == _HAZARD.advice
    # Past it, or 7 m short of it, the operator has nothing specific to say.
    assert op.ask(OperatorQuery(question="?", observation=_obs(9.0))).text != _HAZARD.advice
    assert op.ask(OperatorQuery(question="?", observation=_obs(1.0))).text != _HAZARD.advice


def test_hazard_knowledge_leaves_the_operator_draws_unchanged_without_hazards() -> None:
    a, b = ScriptedOperator(seed=4), ScriptedOperator(seed=4)
    b.begin_episode(_spec(()))
    for x in (0.0, 2.0, 4.0):
        q = OperatorQuery(question="?", observation=_obs(x))
        assert a.ask(q) == b.ask(q)
        assert a.poll_corrections(_obs(x)) == b.poll_corrections(_obs(x))


def test_a_warning_is_volunteered_once_when_decided() -> None:
    op = ScriptedOperator(correction_rate=0.0, hazard_warning_rate=1.0)
    op.begin_episode(_spec((_HAZARD,)))
    assert op.poll_corrections(_obs(1.0)) == ()  # 7 m short: out of range
    (warning,) = op.poll_corrections(_obs(2.5))
    assert warning.text == _HAZARD.advice
    assert op.poll_corrections(_obs(3.0)) == ()  # said once already


# -------------------------------------------------------------------- retrieval


def _retriever(radius_m: float = 4.0) -> SpatialRetriever:
    r = SpatialRetriever(radius_m=radius_m)
    r.bind(HashingEmbedder(dim=32), InMemoryStore())
    return r


def test_spatial_scores_fall_linearly_with_distance() -> None:
    r = _retriever(radius_m=4.0)
    r.ingest(Correction(text=_HAZARD.advice, observation=_obs(4.0), operator_id="s"), run_t=0.0)
    # 1 m away: 1 - 1/4 = 0.75. 3 m away: 0.25. 5 m away: beyond the radius.
    assert r.retrieve(_obs(5.0), 3).scores == pytest.approx((0.75,))
    assert r.retrieve(_obs(1.0), 3).scores == pytest.approx((0.25,))
    assert r.retrieve(_obs(9.0), 3).entries == ()


def test_spatial_ties_keep_insertion_order_and_k_is_respected() -> None:
    r = _retriever()
    for text in ("first", "second", "third"):
        r.ingest(Correction(text=text, observation=_obs(4.0), operator_id="s"), run_t=0.0)
    result = r.retrieve(_obs(4.0), 2)
    assert [e.text for e in result.entries] == ["first", "second"]
    assert result.scores == pytest.approx((1.0, 1.0))


# --------------------------------------------------------------- episode loop


def test_driving_into_a_hazard_ends_the_episode_as_a_failure() -> None:
    """The sim moves 1 m per step from x=0. At x=7 the robot is exactly 1.0 m away, not
    inside the 1.0 m keep-out (strictly closer counts); at x=8 it is on the hazard."""
    from pathlib import Path

    from carma.config import build, load
    from carma.runtime.episode import run_episode
    from carma.sim.synthetic import SyntheticSim
    from carma.types.core import Action

    class Rail(SyntheticSim):
        x = 0.0

        def reset(self, *, seed: int, start: Pose) -> Observation:
            Rail.x = 0.0
            return replace(super().reset(seed=seed, start=start), pose=Pose(0.0, 0.0, 0.0))

        def step(self, action: Action) -> Observation:
            Rail.x += 1.0
            return replace(super().step(action), pose=Pose(Rail.x, 0.0, 0.0))

    cfg = load(Path(__file__).resolve().parents[1] / "configs/experiment/smoke.yaml")
    assembly = replace(build(cfg), sim=Rail())
    spec = replace(_spec((_HAZARD,)), max_steps=50)
    record = run_episode(spec, assembly, seed=0, run_id="t")
    assert record.entered_hazard
    assert not record.success
    # Passing the stop point at x=6 did not end it: only the robot's own stop could.
    assert record.final_pose.x == pytest.approx(8.0)


# ------------------------------------------------------------ context to policy


def test_task_text_folds_context_into_the_instruction() -> None:
    from carma.backbone.context import stop_distance, task_text

    assert task_text("drive 13 metres", ()) == "drive 13 metres"
    assert task_text("drive 13 metres", ("  ",)) == "drive 13 metres"
    assert (
        task_text("drive 13 metres", (_HAZARD.advice, "go slow"))
        == "drive 13 metres. operator: wet patch at 8 metres: stop at 6 metres; go slow"
    )
    assert stop_distance((_HAZARD.advice,)) == 6.0
    assert stop_distance(("keep to the left of the row",)) is None


def test_the_expert_stops_where_the_operator_says() -> None:
    from carma.backbone.alley_expert import AlleyExpert

    expert = AlleyExpert(stop_margin_m=0.3)
    rng = np.random.default_rng(0)
    at_six = _obs(5.8)  # past 6 - 0.3, the stop margin
    assert not expert.act(at_six, (), rng=rng)[0].stop  # the instruction says 13 m
    assert expert.act(at_six, (_HAZARD.advice,), rng=rng)[0].stop  # the operator says 6 m


def test_demonstrations_carry_the_advice_only_within_range() -> None:
    from carma.backbone.alley_expert import AlleyExpert
    from carma.runtime.demonstrations import record_episode
    from carma.sim.synthetic import SyntheticSim
    from carma.types.core import Action

    class Rail(SyntheticSim):
        x = 0.0

        def reset(self, *, seed: int, start: Pose) -> Observation:
            Rail.x = 0.0
            return replace(super().reset(seed=seed, start=start), pose=Pose(0.0, 0.0, 0.0))

        def step(self, action: Action) -> Observation:
            Rail.x += 0.5
            return replace(super().step(action), pose=Pose(Rail.x, 0.0, 0.0))

    spec = replace(_spec((_HAZARD,)), max_steps=60)
    ep = record_episode(
        Rail(), AlleyExpert(), spec, seed=0, rng=np.random.default_rng(0), disturbance=None
    )
    with_advice = [s for s in ep.steps if "operator:" in s.task]
    # Advice in force from x >= 8 - 6 = 2; the expert stops near the 6 m stop point.
    assert min(s.pose.x for s in with_advice) == pytest.approx(2.0)
    assert all(s.pose.x < 2.0 for s in ep.steps if "operator:" not in s.task)
    assert ep.steps[-1].label.stop
    assert ep.steps[-1].pose.x < 8.0 - _HAZARD.keep_out_m


def test_entry_ids_are_reproducible() -> None:
    def ids() -> list[str]:
        r = _retriever()
        return [
            r.ingest(
                Correction(text=f"c{i}", observation=_obs(float(i)), operator_id="s"), run_t=0.0
            ).entry_id
            for i in range(3)
        ]

    assert ids() == ids() == ["e000000", "e000001", "e000002"]


def test_a_fixed_hazard_stays_in_one_place() -> None:
    specs = row_alley_episodes(
        5, rng=np.random.default_rng(2), hazard_probability=1.0, hazard_x_m=8.0
    )
    assert {s.hazards[0].position for s in specs} == {Pose(8.0, 0.0, 0.0)}
    assert {s.goal for s in specs} == {Pose(6.0, 0.0, 0.0)}
