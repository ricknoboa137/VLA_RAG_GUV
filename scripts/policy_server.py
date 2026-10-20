"""Serve a navigation policy over the act contract.

    python scripts/policy_server.py --policy echo
    carma run --config configs/experiment/remote_smoke.yaml

``--policy echo`` serves the in-repo heuristic, which needs no checkpoint and
no GPU. It exists so the whole client-server path -- encoding, seeding,
uncertainty, manifests -- can be exercised and tested before any large model
is involved, and so a failure can be attributed to the plumbing or to the
model rather than to both at once.

``--policy xvla`` is the slot for X-VLA (Apache 2.0, 0.9B parameters,
https://github.com/2toinf/X-VLA). It is deliberately unimplemented: every
released X-VLA checkpoint is a **manipulator** policy whose action is a chunk
of end-effector poses in EE6D, and this project drives a wheeled robot whose
action is a forward velocity and a yaw rate. There is no honest mapping
between the two, so the adapter refuses rather than inventing one. See
docs/vla-integration.md.

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


class XvlaPolicy:
    """X-VLA, once a navigation checkpoint exists.

    Unimplemented on purpose. X-VLA's released weights predict a chunk of
    thirty end-effector poses -- three position, six rotation, one gripper --
    for a manipulator. Turning that into a wheeled robot's forward velocity
    and yaw rate is not an adapter but a modelling decision, and guessing one
    would produce confident nonsense in every experiment that followed.

    What this needs before it can be written, in order:

    1. A navigation action space the checkpoint actually predicts, which means
       fine-tuning from ``2toINF/X-VLA-Pt`` on navigation data, with a soft
       prompt for this embodiment.
    2. An uncertainty estimator. Flow matching yields no logits, so the
       honest route is to draw ``request.samples`` chunks from different noise
       and take their disagreement, then state that in
       ``uncertainty_method`` and calibrate it, as section 8 requires.
    3. A recorded revision, so a run manifest names the weights.
    """

    def __init__(self, checkpoint: str) -> None:
        self._checkpoint = checkpoint

    @property
    def model_id(self) -> str:
        """The checkpoint this would serve."""
        return self._checkpoint

    def act(self, request: ActRequest) -> ActResponse:
        """Refuse, with the reason."""
        msg = (
            f"{self._checkpoint} predicts end-effector poses for a manipulator, and this "
            "project commands a wheeled robot's linear and angular velocity. No released "
            "X-VLA checkpoint predicts navigation actions. See docs/vla-integration.md."
        )
        raise NotImplementedError(msg)


def build_policy(kind: str, checkpoint: str) -> Policy:
    """Construct the named policy."""
    if kind == "echo":
        return EchoPolicy()
    if kind == "xvla":
        return XvlaPolicy(checkpoint)
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
    request = ActRequest(
        image=encode_image(frame),
        instruction="follow the row",
        pose=(0.0, 0.0, 0.0),
        t=0.0,
        stage="VEGETATIVE",
        seed=12345,
    )
    first = policy.act(request)
    second = policy.act(request)
    if (first.linear, first.angular, first.stop, first.uncertainty) != (
        second.linear,
        second.angular,
        second.stop,
        second.uncertainty,
    ):
        msg = (
            "policy is not deterministic under a fixed seed: the same request gave "
            f"{first.linear:.6f}/{first.angular:.6f} then "
            f"{second.linear:.6f}/{second.angular:.6f}"
        )
        raise SystemExit(msg)
    print("self-check: deterministic under a fixed seed")


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
    parser.add_argument("--policy", choices=("echo", "xvla"), default="echo")
    parser.add_argument(
        "--checkpoint",
        default="2toINF/X-VLA-Pt",
        help="checkpoint for --policy xvla",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="assert determinism under a fixed seed, then serve",
    )
    args = parser.parse_args()

    policy = build_policy(args.policy, args.checkpoint)
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
