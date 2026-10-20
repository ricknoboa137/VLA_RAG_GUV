"""Serve a policy over the act contract.

    python scripts/policy_server.py --policy echo
    carma run --config configs/experiment/remote_smoke.yaml

``--policy echo`` serves the in-repo heuristic, which needs no checkpoint and
no GPU. It exists so the whole client-server path -- encoding, seeding,
uncertainty, manifests -- can be exercised and tested before any large model
is involved, and so a failure can be attributed to the plumbing or to the
model rather than to both at once.

``--policy smolvla`` and ``--policy xvla`` serve a vision-language-action model
for the SO-101 arm, through LeRobot, from a spec in ``configs/arm_policy/``
(``scripts/arm_policy.py``). They answer with a chunk of joint targets in
``ActResponse.arm`` and never drive the base. They need the ``carma-train``
environment and a fetched checkpoint, and run with the Hub switched off:

    $env:PYTHONPATH = "src"
    & $env:USERPROFILE/anaconda3/envs/carma-train/python.exe scripts/policy_server.py `
        --policy smolvla --self-check --host 0.0.0.0

Neither drives the wheeled base. Every released X-VLA and SmolVLA checkpoint
is a manipulator policy, and there is no honest mapping from arm poses to a
forward velocity and a yaw rate (docs/vla-integration.md), so a navigation
experiment pointed at one of these servers is refused by RemoteBackbone.

Determinism: the seed arrives with each request and is the only source of
randomness. The same request twice must give the same action, and
``--self-check`` asserts exactly that before serving.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Protocol

import numpy as np

from carma.backbone.wire import ActRequest, ActResponse, decode_image

_MAX_BODY_BYTES = 32 * 1024 * 1024


class Policy(Protocol):
    """Turns one request into one response."""

    @property
    def model_id(self) -> str:
        """What to record in the run manifest."""
        ...

    def act(self, request: ActRequest) -> ActResponse:
        """Answer one observation."""
        ...


class EchoPolicy:
    """The in-repo heuristic, served remotely.

    Not a contribution and not a model: it wraps
    :class:`~carma.backbone.heuristic.HeuristicBackbone` so the transport can
    be tested end to end. Uncertainty comes from the heuristic itself, so it
    reports ``heuristic`` rather than claiming a sampled estimate.
    """

    def __init__(self) -> None:
        from carma.backbone.heuristic import HeuristicBackbone

        self._inner = HeuristicBackbone()

    @property
    def model_id(self) -> str:
        """Names itself honestly, so no run mistakes this for a VLA."""
        return "carma:heuristic (not a learned model)"

    def act(self, request: ActRequest) -> ActResponse:
        """Run the heuristic on the decoded frame."""
        from carma.types.core import Observation, PhenologyStage, Pose

        started = time.monotonic()
        image = decode_image(request.image)
        obs = Observation(
            rgb=image,
            depth=None,
            pose=Pose(x=request.pose[0], y=request.pose[1], yaw=request.pose[2]),
            t=request.t,
            instruction=request.instruction,
            stage=PhenologyStage[request.stage],
        )
        rng = np.random.default_rng(request.seed)
        action, uncertainty = self._inner.act(obs, request.context, rng=rng)
        return ActResponse(
            linear=action.linear,
            angular=action.angular,
            stop=action.stop,
            uncertainty=uncertainty,
            uncertainty_method="heuristic",
            model_id=self.model_id,
            model_revision=None,
            inference_s=time.monotonic() - started,
            # What arrived, so a live pipeline can be checked end to end
            # before a model that reads the extra views is plugged in.
            info={
                "image_shape": list(image.shape),
                "stereo": request.image_right is not None,
                "views": [name for name, _ in request.views],
                "instruction": request.instruction,
            },
        )


ARM_SPECS = {
    "smolvla": Path("configs/arm_policy/svla_so101_red_cube.yaml"),
    "xvla": Path("configs/arm_policy/xvla_so101_red_cube.yaml"),
}


def build_policy(kind: str, arm_spec: Path | None = None, device: str = "cuda") -> Policy:
    """Construct the named policy; an arm policy reads its spec, by default the kind's own."""
    if kind == "echo":
        return EchoPolicy()
    if kind in ARM_SPECS:
        from arm_policy import LeRobotArmPolicy, load_spec

        spec = load_spec(arm_spec or ARM_SPECS[kind])
        if spec.kind != kind:
            msg = f"--policy {kind} given a {spec.kind} spec"
            raise SystemExit(msg)
        return LeRobotArmPolicy(spec, device=device)
    msg = f"unknown policy {kind!r}"
    raise SystemExit(msg)


def _self_check(policy: Policy) -> None:
    """Assert the policy is deterministic under a fixed seed, before serving.

    A server that samples from its own entropy makes every run that used it
    irreproducible, and the failure is silent: the numbers simply differ next
    time. Cheaper to catch here than in an experiment.
    """
    from carma.backbone.wire import encode_image

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    world = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    request = ActRequest(
        image=encode_image(frame),
        instruction="pick up the red cube",
        pose=(0.0, 0.0, 0.0),
        t=0.0,
        stage="VEGETATIVE",
        seed=12345,
        views=(("world_left", encode_image(world)),),
        arm_state=tuple((name, 0.0) for name in joints),
    )
    first = policy.act(request)
    second = policy.act(request)

    def answer(r: ActResponse) -> tuple[object, ...]:
        return (r.linear, r.angular, r.stop, r.uncertainty, r.arm)

    if answer(first) != answer(second):
        msg = (
            "policy is not deterministic under a fixed seed: the same request gave "
            f"{first.linear:.6f}/{first.angular:.6f} then "
            f"{second.linear:.6f}/{second.angular:.6f}"
            + ("" if first.arm == second.arm else ", and different arm chunks")
        )
        raise SystemExit(msg)
    detail = "" if first.arm is None else f", {len(first.arm.positions_rad)}-step arm chunk"
    print(f"self-check: deterministic under a fixed seed{detail}")


def make_handler(policy: Policy) -> type[BaseHTTPRequestHandler]:
    """A request handler bound to one policy."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - the name BaseHTTPRequestHandler dispatches to
            """Answer one act request."""
            if self.path.rstrip("/") != "/act":
                self._fail(404, f"no endpoint {self.path}")
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > _MAX_BODY_BYTES:
                self._fail(413, f"body must be 1..{_MAX_BODY_BYTES} bytes, got {length}")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                request = ActRequest.from_json(payload)
            except (ValueError, UnicodeDecodeError) as exc:
                self._fail(400, f"bad request: {exc}")
                return

            try:
                response = policy.act(request)
            except NotImplementedError as exc:
                self._fail(501, str(exc))
                return
            except ValueError as exc:
                # The request was well formed but this policy cannot answer it,
                # e.g. an arm policy asked without the arm's joint angles.
                self._fail(422, f"cannot answer: {exc}")
                return
            except Exception as exc:
                self._fail(500, f"{type(exc).__name__}: {exc}")
                return

            body = json.dumps(response.to_json()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _fail(self, code: int, message: str) -> None:
            body = json.dumps({"error": message}).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """One line per request, to stdout rather than stderr."""
            print(f"{self.address_string()} {format % args}", flush=True)

    return Handler


def main() -> int:
    """Serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", choices=("echo", *ARM_SPECS), default="echo")
    parser.add_argument(
        "--arm-spec",
        type=Path,
        default=None,
        help="spec for --policy smolvla/xvla (default: the kind's spec in configs/arm_policy/)",
    )
    parser.add_argument("--device", default="cuda", help="torch device for an arm policy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="assert determinism under a fixed seed, then serve",
    )
    args = parser.parse_args()

    policy = build_policy(args.policy, args.arm_spec, args.device)
    if args.self_check:
        _self_check(policy)

    server = HTTPServer((args.host, args.port), make_handler(policy))
    print(
        f"serving {policy.model_id} on http://{args.host}:{args.port}/act — Ctrl+C to stop",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
