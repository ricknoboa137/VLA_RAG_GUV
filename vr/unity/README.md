# CARMA Quest client (Unity)

`CarmaQuest/` is the Meta Quest 2 application for CARMA: it shows the robot's
camera in the headset, sends controller and head input to the robot, and lets
the operator talk to the robot with push-to-talk speech.

It is a migrated copy of the original `CleanSetup` project. **The original is
not modified** and stays at `Documents\UnityProjects\CleanSetup` (Unity
2022.3.15f1, Oculus Integration) for demonstrations; a backup of it is at
`Documents\UnityProjects\CleanSetup_backup_2022.3.15f1`.

| | Original `CleanSetup` | `CarmaQuest` (this folder) |
|---|---|---|
| Unity | 2022.3.15f1 | **6000.3.24f1** (LTS) |
| Meta SDK | Oculus Integration copied into `Assets/Oculus` (919 MB) | **Meta XR SDK 205.0.0** from the package manager |
| XR plugin | Oculus XR Plugin 4.1.2 | **OpenXR 1.18.0** + Unity Meta OpenXR 2.6.1 |
| Speech | none | `CarmaVoice` push-to-talk |
| Size in git | too large to push | a few MB |

## Open the project

1. Install **Unity 6000.3.24f1** with **Android Build Support** (including
   *Android SDK & NDK Tools* and *OpenJDK*) from Unity Hub.
2. Clone this repository, then in Unity Hub choose *Add project from disk* and
   pick `vr/unity/CarmaQuest`.
3. The first open takes several minutes: Unity downloads the Meta XR SDK from
   Meta's package registry (declared in `Packages/manifest.json`) and rebuilds
   `Library/`. Nothing else needs installing.
4. *File → Build Profiles* → select **Android** → *Switch Platform*.

If Meta's *Project Setup Tool* shows issues (*Edit → Project Settings → Meta
XR*), apply its fixes.

## Migration status

Verified by opening the project in Unity 6000.3.24f1 in batch mode with the
Android build target:

- Packages resolve: Meta XR SDK Core, Interaction and Interaction OVR 205.0.0;
  OpenXR 1.18.0 and Unity Meta OpenXR 2.6.1.
- XR provider is **OpenXR** on Android and Standalone (the deprecated Oculus XR
  plugin is removed). Enabled OpenXR features: Meta XR Feature (for
  `OVRCameraRig` and `OVRInput`), Oculus Touch Controller Profile, and Meta
  Quest Support on Android. OpenXR project validation fixes applied: Vulkan
  graphics API, Game Activity entry point, input-polling latency optimization.
  One informational notice remains (a feature targets an OpenXR API patch
  version below 1.1.54); it has no automatic fix and is not an error.
- Scripts compile with **0 errors**. The six warnings are obsolete-API notices
  in `Assets/LegacyOVR/FromOVRControllerHandDataSource.cs`.
- All 25 asset references in `SampleScene` resolve (Oculus prefabs map to the
  same GUIDs in the Meta XR packages, or to `Assets/LegacyOVR/`).

Not yet done, needs the editor or the headset: adding the `CarmaVoice`
component to the scene, running Meta's Project Setup Tool, and a build to the
Quest 2.

## What is in the scene

`Assets/Scenes/SampleScene.unity`:

| Object / script | Role | Topic |
|---|---|---|
| `mqttReceiver` | Receives camera frames (base64 JPEG) and shows them on `CarmaVideo` | subscribes `MqttVidFeed` |
| `mqttController` | Publishes head rotation, thumbstick and A button every frame | publishes `test1` |
| `CarmaVoice` | Push-to-talk speech (hold **B**) | publishes `carma/operator/audio`, subscribes `carma/operator/utterance` |

`Assets/LegacyOVR/` holds two prefabs and one script from the old Oculus
Integration that the scene still references and Meta XR SDK 205 no longer
ships; see the README inside.

### Why the video panel moved

The original project shows video on `RawImageLeft` and `RawImageRight`,
parented to `LeftEyeAnchor` and `RightEyeAnchor` and placed on the custom
layers `left` and `right`, so that each eye's own camera draws only its own
panel. Under OpenXR nothing renders those layers and both panels stay
invisible, which looks exactly like a dead video feed: the frames arrive, the
JPEG decodes and the texture is correct, but it is never drawn.

`CarmaVideo` under `CenterEyeAnchor` replaces them. It is on the `Default`
layer, which does render — the same arrangement as the speech status panel.
Both `displayLeft` and `displayRight` point at its single `RawImage`, so a
mono frame fills the view.

The old per-eye objects are left in the scene, unused.

### Stereo

A stereo camera sends both eyes in one synchronised frame, side by side. The
split happens in `Assets/Shaders/StereoSideBySide.shader`, which picks the
half matching `unity_StereoEyeIndex`, because the per-eye-layer approach does
not render at all (above) and one panel therefore has to serve both eyes. It
works under single-pass instanced and multi-pass alike.

`CarmaStereoDisplay` on the same object decides the layout from the frame's
aspect ratio: two 640x480 views arrive as 1280x480, so anything at least 2.2
times wider than tall is treated as a pair. `mode` forces Mono or Stereo for
an ultra-wide single camera, and `swapEyes` corrects a camera whose halves are
right-left — get that wrong and depth inverts uncomfortably.

The development "3D USB Camera" on this PC is index 1 and offers 2560x960,
2560x720, 1280x480 and 640x240, all side-by-side. 1280x480 suits this
transport; the higher modes wait for H.264:

```bash
python scripts/webcam_mqtt.py --camera 1 --width 1280 --height 480 \
    --host <pc-ip> --topic MqttVidFeed --encoding base64
```

Note `--width`/`--height`: without them the camera stays in its 640x240
default, and the publisher sets MJPG first because most USB cameras offer
their higher modes only compressed.

To diagnose a black screen, `mqttReceiver` logs one line per second with the
frame count, payload size, whether `LoadImage` succeeded and the texture size:

```bash
adb logcat -s Unity | grep mqttReceiver
```

Frames counting up with `loaded=True` means the problem is rendering, not the
feed.

## Set up push-to-talk speech

`SampleScene` already contains it:

- **`CarmaVoice`** (root object) with the *Carma Voice* component: **Broker
  Address** `192.168.0.153` (the development PC on the home Wi-Fi), **Broker
  Port** `1883`, **Auto Connect** on, talk button **B**. Change the address to
  the robot's broker on the robot network; it may differ from the video
  receiver's broker (`10.169.15.27` in the scene).
- **`CarmaVoiceStatus`** under `OVRCameraRig/.../CenterEyeAnchor`: a small
  world-space panel 1.2 m ahead and slightly below eye level, linked as the
  component's **Status Label**, showing `listening…`, `transcribing…` and the
  result.
4. Microphone permission is already declared in
   `Assets/Plugins/Android/AndroidManifest.xml`
   (`android.permission.RECORD_AUDIO`). Accept the prompt in the headset on
   first launch.

## Run on the Quest 2

1. Enable developer mode on the headset and connect it by USB.
2. *File → Build Profiles → Android → Build and Run*.
3. On the PC, start the speech service (from the repository root):

   ```bash
   python scripts/fetch_models.py faster-whisper-base
   docker compose -f docker/compose.yaml up -d voice
   ```

4. In the headset, hold **B**, speak ("stop", "gira a la derecha", "sí", or a
   sentence of advice), release. The status label shows the transcription and
   the intent, for example:

   ```
   "gira a la derecha"
   → turn_right  [es, 0.9 s]
   ```

`not understood (no_speech)` means the clip was rejected as silence or noise;
see `docs/speech.md`.

## Signing

Release builds need an Android keystore. **Never commit it**: `*.keystore` and
`*.jks` are ignored. Keep the keystore outside the repository and point
*Project Settings → Player → Publishing Settings* at it.

## Troubleshooting

| Symptom | Check |
|---|---|
| Packages fail to resolve on open | Internet access to `npm.developer.oculus.com` and `packages.unity.com` |
| `broker connection failed` | Quest and broker on the same network; broker address; Windows Firewall allows `mosquitto.exe` inbound on the active profile |
| Always `too short` | Hold B for the whole phrase |
| Always `not understood` | Microphone permission denied: headset *Settings → Apps → Permissions* |
| Nothing after `transcribing…` | `docker compose -f docker/compose.yaml logs voice` on the PC |

Headset logs: `adb logcat -s Unity`, look for `[CarmaVoice]`.
