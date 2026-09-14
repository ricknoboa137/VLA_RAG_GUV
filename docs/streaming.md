# Camera streaming and VR viewing

## Two different jobs

| Job | Needs | Transport |
|---|---|---|
| Monitoring, dashboards, logging analysis results | Simple, works through a broker, any client | **MQTT** (implemented) |
| Live stereo view in a VR headset | Low and stable latency, adapts to Wi-Fi | **WebRTC** (planned) |

These should not share a transport.

## Why MQTT is fine for monitoring

`carma_vision/stream_node` publishes side-by-side JPEG frames (left eye on the
left) to `carma/camera/frame` with QoS 0, and a retained `carma/camera/meta`
JSON description (`layout`, `width`, `height`, `rate_hz`). Rate, JPEG quality
and maximum width are parameters in `carma_vision/config/vision.yaml`. Any MQTT client that can decode JPEG can show it, and it fits the
tooling from other projects. Analysis results travel the same way, and for
small JSON messages MQTT is the right choice.

## Why MQTT is the wrong choice for VR

- **Every frame is a full JPEG.** Video codecs (H.264/H.265) send only what
  changed and are hardware-encoded on most robot computers; per-frame JPEG
  needs far more bandwidth for the same quality, and a stereo pair doubles it.
- **TCP through a broker.** A single lost packet on field Wi-Fi stalls every
  frame behind it, and the broker adds a hop. In a headset that shows up as
  judder, which causes discomfort quickly.
- **No rate adaptation.** When the link degrades, MQTT keeps sending at the
  same size until buffers fill. WebRTC measures the link and lowers bitrate.

## Planned VR path

```
stereo camera -> ROS 2 -> WebRTC sender on the robot (H.264, hardware encode)
               -> Wi-Fi -> headset browser page using WebXR
               -> left half to the left eye, right half to the right eye
```

WebRTC and WebXR are open standards, run in the headset's built-in browser
without installing an app, and need no account. MQTT stays in the design as the
signalling and telemetry channel (the broker already exposes WebSockets, on
host port 9002 by default, which the viewer page in `docker/viewer/` uses).

## Open decisions

- **Camera model.** Sets resolution, frame rate, whether the driver also gives
  depth, and whether it can hardware-encode. Choose it before the WebRTC sender.
- **Headset model.** Confirms WebXR support in its browser and the per-eye
  resolution worth sending.
- **Stereo baseline.** For comfortable viewing it should be close to human eye
  spacing; a wide-baseline depth camera gives exaggerated depth in VR.
