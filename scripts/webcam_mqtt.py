"""Send a local webcam to the MQTT broker, as a stand-in for the robot's camera.

For testing the ROS 2 pipeline on a machine whose cameras cannot be passed into
Docker (Windows, macOS). The ``mqtt_camera`` node inside the container turns
these frames back into ROS image topics.

    pip install -e ".[stream]"
    python scripts/webcam_mqtt.py --camera 0

With one webcam the frame is sent as mono and the container duplicates it into
both eyes; ``--right-camera`` sends a real side-by-side pair from two cameras.

The Quest client in ``vr/unity`` reads a different topic and expects the JPEG
as base64 text rather than bytes, so feeding the headset directly needs both
overrides:

    python scripts/webcam_mqtt.py --host 192.168.0.153 \
        --topic MqttVidFeed --encoding base64

Base64 costs a third more bandwidth and an extra copy per frame; it exists to
drive the existing Unity receiver unchanged. Prefer ``raw`` for the ROS path.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt


def _open(index: int, width: int | None = None, height: int | None = None) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        msg = f"camera {index} could not be opened"
        raise SystemExit(msg)
    if width and height:
        # MJPG first: most USB cameras only offer their higher modes compressed,
        # and silently stay at the default resolution when asked in raw YUY2.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (got_w, got_h) != (width, height):
            print(
                f"camera {index}: asked {width}x{height}, got {got_w}x{got_h}",
                file=sys.stderr,
            )
    return cap


def main() -> int:
    """Capture, encode and publish until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="left (or only) camera index")
    parser.add_argument("--right-camera", type=int, default=None, help="right camera index")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--prefix", default="carma")
    parser.add_argument(
        "--topic",
        default=None,
        help="frame topic; defaults to <prefix>/source/frame. Use MqttVidFeed for the Quest client",
    )
    parser.add_argument(
        "--encoding",
        choices=("raw", "base64"),
        default="raw",
        help="raw JPEG bytes (ROS path) or base64 text (Unity mqttReceiver)",
    )
    parser.add_argument("--width", type=int, default=None, help="capture width, e.g. 1280")
    parser.add_argument("--height", type=int, default=None, help="capture height, e.g. 480")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    args = parser.parse_args()
    frame_topic = args.topic if args.topic is not None else f"{args.prefix}/source/frame"

    left = _open(args.camera, args.width, args.height)
    right = (
        _open(args.right_camera, args.width, args.height) if args.right_camera is not None else None
    )

    # A single camera that returns a frame far wider than it is tall is already
    # delivering both eyes side by side, which is how most "3D USB" cameras
    # work; the headset splits it. Two cameras are stitched below instead.
    ok, probe = left.read()
    wide = bool(ok) and probe.shape[1] / probe.shape[0] >= 2.0
    layout = "sbs_lr" if right is not None or wide else "mono"

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="carma-webcam")
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()
    client.publish(f"{args.prefix}/source/meta", json.dumps({"layout": layout}), qos=1, retain=True)

    period = 1.0 / args.fps
    sent = 0
    started = time.monotonic()
    print(
        f"publishing {layout} frames to {args.host}:{args.port} "
        f"on {frame_topic} as {args.encoding} — Ctrl+C to stop"
    )
    try:
        while True:
            tick = time.monotonic()
            ok, frame = left.read()
            if not ok:
                print("camera read failed", file=sys.stderr)
                return 1
            if right is not None:
                ok_r, frame_r = right.read()
                if not ok_r:
                    print("right camera read failed", file=sys.stderr)
                    return 1
                frame_r = cv2.resize(frame_r, (frame.shape[1], frame.shape[0]))
                frame = np.hstack((frame, frame_r))
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            if ok:
                jpeg = buf.tobytes()
                payload = base64.b64encode(jpeg) if args.encoding == "base64" else jpeg
                client.publish(frame_topic, payload, qos=0)
                sent += 1
            if sent and sent % int(args.fps * 5) == 0:
                rate = sent / (time.monotonic() - started)
                print(f"{sent} frames, {rate:.1f} fps, {len(buf) // 1024} KiB/frame")
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        left.release()
        if right is not None:
            right.release()
        client.loop_stop()
        client.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
