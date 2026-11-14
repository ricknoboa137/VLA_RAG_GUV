"""A navigation backbone served by another process.

The policy runs behind an HTTP endpoint and this adapter makes it satisfy
:class:`~carma.types.NavigationBackbone`, so a condition still differs from
another by one line of config. The split exists because a 0.9B-parameter
policy wants a GPU the robot may not carry, and because a model can then be
restarted, swapped or profiled without touching the episode loop.

**This is network access, which section 2.3 of the contract forbids at
inference and test time.** The rule is there to stop results depending on a
service nobody can reproduce: a proprietary API, a gated checkpoint, an
account. A policy server the experimenter starts from openly licensed weights
on their own machine is a different thing, and the honest way to keep the
guarantee is to make the dependency explicit rather than to pretend it is
absent:

* Every reply names the weights that produced it, and the episode loop records
  them, so a run says which model answered.
* The seed travels with each request and the server samples from it alone,
  so ``(config, seed, commit)`` still determines the numbers -- provided the
  server is the same. That proviso is why ``model_id`` is recorded.
* **No test in this repository talks to a socket.** The transport is injected,
  and the suite uses an in-process fake.

A result obtained against a server whose weights are not stated is not a
result, which is why :attr:`RemoteBackbone.last_model_id` exists for the
manifest to read.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

import numpy as np

from carma.backbone.registry import BACKBONES
from carma.backbone.wire import ActRequest, ActResponse, encode_image, encode_png16
from carma.types import BackboneError
from carma.types.core import Action, Observation


class Transport(Protocol):
    """Carries one request to a policy and brings back its reply."""

    def post(self, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
        """Send the request body, return the parsed reply."""
        ...


class HttpTransport:
    """Posts JSON to a URL with the standard library, so no dependency is added."""

    def __init__(self, url: str) -> None:
        self._url = url

    def post(self, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
        """POST ``payload`` as JSON and parse the reply."""
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if not self._url.startswith(("http://", "https://")):
            msg = f"policy url must be http(s), got {self._url!r}"
            raise BackboneError(msg)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as reply:
                return dict(json.loads(reply.read().decode("utf-8")))
        except urllib.error.URLError as exc:
            msg = f"policy server at {self._url} did not answer: {exc.reason}"
            raise BackboneError(msg) from exc
        except OSError as exc:
            # A timeout while reading the reply, or a connection dropped
            # mid-answer, arrives as TimeoutError or ConnectionError rather
            # than URLError. Both mean the server did not answer.
            msg = f"policy server at {self._url} did not answer: {exc!r}"
            raise BackboneError(msg) from exc
        except json.JSONDecodeError as exc:
            msg = f"policy server at {self._url} returned invalid JSON"
            raise BackboneError(msg) from exc


@BACKBONES.register("remote")
class RemoteBackbone:
    """Asks a policy server for each action.

    Args:
        url: Endpoint serving the act contract, for example
            ``http://127.0.0.1:8000/act``.
        timeout_s: How long to wait for one action before failing the episode.
        samples: Action chunks the server should draw per step. Their spread
            becomes the uncertainty, so one sample cannot yield a calibrated
            value and the server is expected to say so.
        send_depth: Send the depth image when the observation carries one.
            Off by default: it is a third of the payload and most policies
            ignore it.
        send_right: Send the right stereo frame when present.
        transport: Injected for tests. Defaults to HTTP against ``url``.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8000/act",
        timeout_s: float = 10.0,
        samples: int = 4,
        *,
        send_depth: bool = False,
        send_right: bool = False,
        transport: Transport | None = None,
    ) -> None:
        if samples < 1:
            msg = f"samples must be at least 1, got {samples}"
            raise ValueError(msg)
        self._url = url
        self._timeout_s = timeout_s
        self._samples = samples
        self._send_depth = send_depth
        self._send_right = send_right
        self._transport: Transport = transport if transport is not None else HttpTransport(url)
        self._last_model_id: str | None = None
        self._last_model_revision: str | None = None
        self._last_uncertainty_method: str | None = None

    @property
    def last_model_id(self) -> str | None:
        """Weights that answered most recently, for the run manifest."""
        return self._last_model_id

    @property
    def last_model_revision(self) -> str | None:
        """Revision of those weights, when the server could report one."""
        return self._last_model_revision

    @property
    def last_uncertainty_method(self) -> str | None:
        """How the most recent uncertainty was obtained."""
        return self._last_uncertainty_method

    def reset(self) -> None:
        """Clear per-episode state.

        The server is stateless per request by contract: every step carries its
        own seed and full observation, so there is nothing to tell it.
        """
        self._last_model_id = None
        self._last_model_revision = None
        self._last_uncertainty_method = None

    def act(
        self,
        obs: Observation,
        context: tuple[str, ...] = (),
        *,
        rng: np.random.Generator,
    ) -> tuple[Action, float]:
        """Return an action and an uncertainty in ``[0, 1]``.

        The seed handed to the server is drawn from ``rng``, so the whole
        sequence of requests in an episode follows from the run's seed.
        """
        # A 32-bit draw: wide enough not to collide across an episode, and it
        # survives JSON and any server-side RNG without overflow surprises.
        seed = int(rng.integers(0, 2**31 - 1))
        request = ActRequest(
            image=encode_image(obs.rgb),
            instruction=obs.instruction,
            pose=(obs.pose.x, obs.pose.y, obs.pose.yaw),
            t=obs.t,
            stage=obs.stage.name,
            seed=seed,
            samples=self._samples,
            context=context,
            image_right=(
                encode_image(obs.rgb_right)
                if self._send_right and obs.rgb_right is not None
                else None
            ),
            depth_mm=self._encode_depth(obs) if self._send_depth else None,
        )

        payload = self._transport.post(request.to_json(), self._timeout_s)
        try:
            reply = ActResponse.from_json(payload)
        except ValueError as exc:
            msg = f"policy server returned an unusable reply: {exc}"
            raise BackboneError(msg) from exc
        if reply.arm is not None:
            # A manipulator policy answers the base with zero velocity. Taking
            # that as a navigation action would record a run of a robot that
            # never moves, attributed to a model that never drove it.
            msg = (
                f"policy server answered as an arm policy ({reply.model_id}); "
                "a navigation backbone cannot use it"
            )
            raise BackboneError(msg)

        self._last_model_id = reply.model_id
        self._last_model_revision = reply.model_revision
        self._last_uncertainty_method = reply.uncertainty_method
        action = Action(linear=reply.linear, angular=reply.angular, stop=reply.stop)
        return action, reply.uncertainty

    @staticmethod
    def _encode_depth(obs: Observation) -> str | None:
        """Depth in metres to a base64 16-bit millimetre PNG, the RGB-D convention."""
        if obs.depth is None:
            return None
        millimetres = np.where(
            np.isfinite(obs.depth) & (obs.depth > 0.0),
            obs.depth * 1000.0,
            0.0,
        ).astype(np.uint16)
        return encode_png16(millimetres)
