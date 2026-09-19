using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Tells the CARMA/StereoSideBySide material whether the incoming frame holds
/// two eyes side by side, so the shader can give each eye its own half.
///
/// A stereo camera reaches the headset as one wide frame (two 640x480 views
/// become 1280x480), so the layout can be read from the texture itself rather
/// than configured by hand: a frame twice as wide as a mono one is a pair.
/// Set <see cref="mode"/> to Mono or Stereo to override the guess, which is
/// needed for a genuinely ultra-wide single camera.
///
/// Attach next to the RawImage that mqttReceiver draws to. See
/// vr/unity/README.md.
/// </summary>
[RequireComponent(typeof(RawImage))]
public class CarmaStereoDisplay : MonoBehaviour
{
    public enum Layout
    {
        Auto,
        Mono,
        Stereo,
    }

    [Tooltip("Auto treats a frame at least this wide, relative to its height, as side-by-side.")]
    public Layout mode = Layout.Auto;

    [Tooltip("Aspect ratio at or above which Auto decides a frame is side-by-side.")]
    public float stereoAspectThreshold = 2.2f;

    [Tooltip("Swap if the view feels inverted: the pair is right-left rather than left-right.")]
    public bool swapEyes;

    private RawImage image;
    private Material instanced;
    private bool lastStereo;
    private bool everApplied;

    private static readonly int StereoId = Shader.PropertyToID("_Stereo");
    private static readonly int SwapEyesId = Shader.PropertyToID("_SwapEyes");

    private void Awake()
    {
        image = GetComponent<RawImage>();

        // A material asset is shared between every user of it, so edit a copy.
        if (image.material != null && image.material.shader != null
            && image.material.shader.name == "CARMA/StereoSideBySide")
        {
            instanced = new Material(image.material);
            image.material = instanced;
        }
    }

    private void Update()
    {
        if (instanced == null)
        {
            return;
        }

        bool stereo = ResolveStereo();
        if (!everApplied || stereo != lastStereo)
        {
            everApplied = true;
            lastStereo = stereo;
            instanced.SetFloat(StereoId, stereo ? 1f : 0f);
            instanced.SetFloat(SwapEyesId, swapEyes ? 1f : 0f);
            Debug.Log("[CarmaStereoDisplay] " + (stereo ? "side-by-side" : "mono")
                + " (mode=" + mode + ", texture "
                + (image.texture == null ? "none" : image.texture.width + "x" + image.texture.height) + ")");
        }
        else if (instanced.GetFloat(SwapEyesId) != (swapEyes ? 1f : 0f))
        {
            instanced.SetFloat(SwapEyesId, swapEyes ? 1f : 0f);
        }
    }

    private bool ResolveStereo()
    {
        if (mode == Layout.Mono)
        {
            return false;
        }
        if (mode == Layout.Stereo)
        {
            return true;
        }
        var tex = image.texture;
        if (tex == null || tex.height == 0)
        {
            return false;
        }
        return (float)tex.width / tex.height >= stereoAspectThreshold;
    }

    private void OnDestroy()
    {
        if (instanced != null)
        {
            Destroy(instanced);
        }
    }
}
