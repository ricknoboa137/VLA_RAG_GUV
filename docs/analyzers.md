# Scene analyzers

Analyzers are how your own perception models join the system: plant health,
weed detection, disease classification, obstacle classes. Each takes an
`Observation` and returns an `AnalysisResult` — a label plus named float
scores. The contract is `SceneAnalyzer` in `carma/types/protocols.py`.

## What an analyzer may and may not do

An analyzer **describes** the scene. It never chooses an action and never reads
memory, and analyzers are configured in `configs/analyzers/`, outside the
experiment config. Together those mean adding a model to the robot cannot change
an experiment's decisions or its config hash. If a model's output should
influence navigation, that is a backbone or arbiter change and goes through
`AGENTS.md` section 8 instead.

## Path 1: a trained model, no code

1. Train in any framework and export to ONNX (`torch.onnx.export`,
   `tf2onnx`, `skl2onnx`).
2. Place the file under `assets/models/`. `assets/` is gitignored; add a
   `scripts/fetch_*.py` entry so others can obtain it.
3. Add an entry to `configs/analyzers/field.yaml`:

```yaml
  - kind: onnx_classifier
    params:
      name: leaf_disease
      model_path: assets/models/leaf_disease.onnx
      labels: [healthy, rust, blight]
      input_width: 224
      input_height: 224
      channel_order: rgb          # what the model was trained on
      mean: [0.485, 0.456, 0.406] # the training normalisation
      std:  [0.229, 0.224, 0.225]
```

4. Install the runtime locally with `pip install -e ".[models]"`.

The classifier resizes, converts BGR to RGB, normalises, runs the model on CPU
and reports softmax probabilities per label. Getting `channel_order`, `mean`
and `std` wrong does not fail — it silently degrades accuracy — so copy them
from the training script.

## Path 2: custom logic

For anything that is not single-label classification (segmentation, detection,
stereo depth, classic OpenCV):

1. Create `src/carma/perception/analyzers/<name>.py`.
2. Implement a class with a `name` property and `analyze(obs) -> AnalysisResult`,
   decorated with `@ANALYZERS.register("<name>")`. `plant_health.py` is the
   smallest complete example.
3. Import the module in `carma/perception/analyzers/__init__.py`.
4. Add tests to `tests/test_analyzers.py` with a synthetic image whose correct
   answer you can state by hand.

Keep scores as plain floats with their units documented. Put anything larger
(masks, boxes) in `meta` only if it is JSON-serialisable and small, because
results are published over MQTT.

## Stereo

`Observation.rgb` is always the left image; `Observation.rgb_right` holds the
right image when the camera is stereo, `None` otherwise. An analyzer that needs
disparity must check for `None` and raise `AnalyzerError` when it is absent.

## On the robot

`carma_vision/analysis_node` loads the analyzer file named by its
`analyzers_config` parameter and publishes each result to `/carma/analysis` and
to MQTT under `carma/analysis/<name>`.
