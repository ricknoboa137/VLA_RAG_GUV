# Legacy Oculus Integration assets kept for the scene

`SampleScene` references these by GUID and Meta XR SDK 205 no longer ships them
(every other Oculus reference maps to the same GUID in the packages). They are
copied verbatim, `.meta` files included, so the scene links survive the
migration:

| File | Why |
|---|---|
| `OVRCameraRig.prefab` | Building Blocks camera rig variant; its scripts exist in `com.meta.xr.sdk.core` |
| `OVRControllerHands.prefab` | Controller-driven hands; replaced by `ControllerHands.prefab` in the Interaction SDK |
| `FromOVRControllerHandDataSource.cs` | Script used by `OVRControllerHands.prefab`, removed from the Interaction SDK |

Replace them with the Meta XR SDK equivalents when the scene is next reworked,
then delete this folder.
