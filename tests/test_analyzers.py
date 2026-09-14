"""Scene analyzers: bounded, deterministic, and buildable from config."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from carma.config import ComponentSpec, build_analyzers, load_analyzers
from carma.perception.analyzers.onnx_classifier import OnnxClassifier, preprocess_image
from carma.perception.analyzers.plant_health import PlantHealthAnalyzer
from carma.types import AnalyzerError, ConfigError, SceneAnalyzer
from carma.types.core import Observation

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_plant_health_reads_green_half_as_healthy(observation: Observation) -> None:
    """The fixture's lower half is green foliage and its upper half is not."""
    result = PlantHealthAnalyzer().analyze(observation)
    assert result.label == "healthy"
    assert result.scores["vegetation_fraction"] == pytest.approx(0.5)
    assert result.scores["green_fraction_of_vegetation"] == pytest.approx(1.0)
    assert result.scores["chlorotic_fraction_of_vegetation"] == pytest.approx(0.0)


def test_yellow_foliage_reads_as_stressed(observation: Observation) -> None:
    """Yellow leaves (RGB 220, 200, 30) are chlorotic."""
    img = observation.rgb.copy()
    img[30:] = (30, 200, 220)
    result = PlantHealthAnalyzer().analyze(replace(observation, rgb=img))
    assert result.label == "stressed"
    assert result.scores["chlorotic_fraction_of_vegetation"] == pytest.approx(1.0)


def test_bare_scene_has_no_vegetation(observation: Observation) -> None:
    """A uniform grey image contains no foliage and divides by nothing."""
    img = np.full_like(observation.rgb, 128)
    result = PlantHealthAnalyzer().analyze(replace(observation, rgb=img))
    assert result.label == "no_vegetation"
    assert result.scores["green_fraction_of_vegetation"] == 0.0


def test_plant_health_is_bounded_and_deterministic(observation: Observation) -> None:
    """Same image, same scores; fractions stay in [0, 1], ExG in [-1, 2]."""
    analyzer = PlantHealthAnalyzer()
    first, second = analyzer.analyze(observation), analyzer.analyze(observation)
    assert first == second
    for key, value in first.scores.items():
        low, high = (-1.0, 2.0) if key == "excess_green_mean" else (0.0, 1.0)
        assert low <= value <= high


def test_plant_health_rejects_out_of_range_threshold() -> None:
    """A threshold outside [0, 1] is a configuration mistake."""
    with pytest.raises(AnalyzerError, match="stress_threshold"):
        PlantHealthAnalyzer(stress_threshold=1.5)


def test_preprocess_converts_to_rgb_nchw() -> None:
    """An all-red BGR image lands in channel 0 when the model expects RGB."""
    red = np.zeros((4, 6, 3), dtype=np.uint8)
    red[..., 2] = 255
    batch = preprocess_image(red, width=3, height=2, mean=(0, 0, 0), std=(1, 1, 1))
    assert batch.shape == (1, 3, 2, 3)
    assert batch.dtype == np.float32
    assert np.allclose(batch[0, 0], 1.0)
    assert np.allclose(batch[0, 2], 0.0)


def test_preprocess_keeps_bgr_when_asked() -> None:
    """``channel_order='bgr'`` leaves red in channel 2."""
    red = np.zeros((4, 4, 3), dtype=np.uint8)
    red[..., 2] = 255
    batch = preprocess_image(red, 4, 4, (0, 0, 0), (1, 1, 1), channel_order="bgr")
    assert np.allclose(batch[0, 2], 1.0)


def test_missing_onnx_model_is_an_analyzer_error(tmp_path: Path) -> None:
    """A missing model fails with our error type and says where to look."""
    with pytest.raises(AnalyzerError, match="not found"):
        OnnxClassifier(model_path=str(tmp_path / "absent.onnx"), labels=["a", "b"])


def test_shipped_analyzer_configs_build() -> None:
    """Every analyzer config in the repository builds into working analyzers."""
    paths = sorted((REPO_ROOT / "configs" / "analyzers").glob("*.yaml"))
    assert paths
    for path in paths:
        for analyzer in build_analyzers(load_analyzers(path)):
            assert isinstance(analyzer, SceneAnalyzer)


def test_unknown_analyzer_names_its_alternatives() -> None:
    """The error lists what is registered."""
    with pytest.raises(ConfigError, match="plant_health"):
        build_analyzers([ComponentSpec(kind="nonesuch")])


def test_analyzer_config_rejects_extra_keys(tmp_path: Path) -> None:
    """A typo in an analyzer file fails loudly."""
    path = tmp_path / "bad.yaml"
    path.write_text("analyzers: []\nanalyser: []\n")
    with pytest.raises(ConfigError, match="exactly one key"):
        load_analyzers(path)


def test_observation_is_monocular_by_default(observation: Observation) -> None:
    """Stereo is opt-in; existing callers keep working."""
    assert observation.rgb_right is None
