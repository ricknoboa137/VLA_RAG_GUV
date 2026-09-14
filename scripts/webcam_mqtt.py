"""Send a local webcam to the MQTT broker, as a stand-in for the robot's camera.

For testing the ROS 2 pipeline on a machine whose cameras cannot be passed into
Docker (Windows, macOS). The ``mqtt_camera`` node inside the container turns
these frames back into ROS image topics.

    pip install -e ".[stream]"
    python scripts/webcam_mqtt.py --camera 0

With one webcam the frame is sent as mono and the container duplicates it into
both eyes; ``--right-camera`` sends a real side-by-side pair from two cameras.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt


def _open(index: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        msg = f"camera {index} could not be opened"
        raise SystemExit(msg)
    return cap


def main() -> int:
    """Capture, encode and publish until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="left (or only) camera index")
    parser.add_argument("--right-camera", type=int, default=None, help="right camera index")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--prefix", default="carma")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    args = parser.parse_args()

    left = _open(args.camera)
    right = _open(args.right_camera) if args.right_camera is not None else None
    layout = "sbs_lr" if right is not None else "mono"

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="carma-webcam")
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()
    client.publish(f"{args.prefix}/source/meta", json.dumps({"layout": layout}), qos=1, retain=True)

    period = 1.0 / args.fps
    sent = 0
    started = time.monotonic()
    print(f"publishing {layout} frames to {args.host}:{args.port} — Ctrl+C to stop")
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
                client.publish(f"{args.prefix}/source/frame", buf.tobytes(), qos=0)
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
