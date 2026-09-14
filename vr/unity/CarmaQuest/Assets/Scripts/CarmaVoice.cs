using System;
using System.IO;
using System.Text;
using M2MqttUnity;
using UnityEngine;
using UnityEngine.UI;
using uPLibrary.Networking.M2Mqtt.Messages;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Android;
#endif

/// <summary>
/// Push-to-talk operator speech for CARMA on a Meta Quest.
///
/// Hold the talk button (B by default), speak, release. The clip is encoded as
/// 16-bit PCM WAV and published to <c>carma/operator/audio</c>; the carma_voice
/// container transcribes it and publishes the result to
/// <c>carma/operator/utterance</c>, which this component shows on
/// <see cref="statusLabel"/>.
///
/// It keeps its own MQTT connection, so existing video and controller scripts
/// are unaffected. Button A is left alone because mqttController sends it to
/// the robot.
///
/// Requires android.permission.RECORD_AUDIO in the Android manifest.
/// See vr/unity/README.md.
/// </summary>
public class CarmaVoice : M2MqttUnityClient
{
    [Header("CARMA topics")]
    public string audioTopic = "carma/operator/audio";
    public string utteranceTopic = "carma/operator/utterance";

    [Header("Push to talk")]
    [Tooltip("Hold to record. A (Button.One) is used by mqttController for robot control.")]
    public OVRInput.Button talkButton = OVRInput.Button.Two;
    [Tooltip("Requested sample rate in Hz; the server resamples whatever arrives.")]
    public int sampleRate = 16000;
    public int maxClipSeconds = 10;
    public float minClipSeconds = 0.3f;

    [Header("Feedback (optional)")]
    public Text statusLabel;

    private string micDevice;
    private int recordRate;
    private AudioClip recording;
    private bool isRecording;
    private float recordStartTime;

    [Serializable]
    private class Utterance
    {
        public string text;
        public string intent;
        public string rejected;
        public string language;
        public float audio_s;
        public float stt_latency_s;
    }

    protected override void Start()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            Permission.RequestUserPermission(Permission.Microphone);
        }
#endif
        if (Microphone.devices.Length > 0)
        {
            micDevice = Microphone.devices[0];
            int minFreq, maxFreq;
            Microphone.GetDeviceCaps(micDevice, out minFreq, out maxFreq);
            // Both zero means the device accepts any rate.
            recordRate = (minFreq == 0 && maxFreq == 0) ? sampleRate : Mathf.Clamp(sampleRate, minFreq, maxFreq);
            SetStatus("hold B to talk");
        }
        else
        {
            SetStatus("no microphone found");
        }
        base.Start();
    }

    protected override void Update()
    {
        base.Update(); // processes MQTT messages on the main thread

        if (micDevice == null)
        {
            return;
        }
        if (!isRecording && OVRInput.GetDown(talkButton))
        {
            BeginRecording();
        }
        else if (isRecording && (OVRInput.GetUp(talkButton) || Time.time - recordStartTime >= maxClipSeconds))
        {
            EndRecordingAndSend();
        }
    }

    private void BeginRecording()
    {
        recording = Microphone.Start(micDevice, false, maxClipSeconds, recordRate);
        recordStartTime = Time.time;
        isRecording = true;
        SetStatus("listening…");
    }

    private void EndRecordingAndSend()
    {
        int position = Microphone.GetPosition(micDevice);
        bool filledClip = Time.time - recordStartTime >= maxClipSeconds - 0.1f;
        Microphone.End(micDevice);
        isRecording = false;

        if (recording == null)
        {
            SetStatus("nothing recorded");
            return;
        }
        // A non-looping clip that ran to its end reports position 0.
        int sampleCount = position > 0 ? position : (filledClip ? recording.samples : 0);
        float seconds = sampleCount / (float)recording.frequency;
        if (seconds < minClipSeconds)
        {
            SetStatus("too short: hold B while speaking");
            return;
        }
        if (client == null || !client.IsConnected)
        {
            SetStatus("not connected to broker " + brokerAddress);
            return;
        }

        float[] samples = new float[sampleCount * recording.channels];
        recording.GetData(samples, 0);
        byte[] wav = EncodeWav(samples, recording.channels, recording.frequency);
        client.Publish(audioTopic, wav, MqttMsgBase.QOS_LEVEL_AT_LEAST_ONCE, false);
        SetStatus(string.Format("sent {0:F1} s, transcribing…", seconds));
    }

    protected override void SubscribeTopics()
    {
        client.Subscribe(new string[] { utteranceTopic }, new byte[] { MqttMsgBase.QOS_LEVEL_AT_LEAST_ONCE });
    }

    protected override void UnsubscribeTopics()
    {
        client.Unsubscribe(new string[] { utteranceTopic });
    }

    protected override void OnConnected()
    {
        base.OnConnected();
        SetStatus(micDevice == null ? "connected, but no microphone" : "connected: hold B to talk");
    }

    protected override void OnConnectionFailed(string errorMessage)
    {
        base.OnConnectionFailed(errorMessage);
        SetStatus("broker connection failed: " + brokerAddress + ":" + brokerPort);
    }

    protected override void DecodeMessage(string topic, byte[] message)
    {
        if (topic != utteranceTopic)
        {
            return;
        }
        Utterance u = (Utterance)JsonUtility.FromJson(Encoding.UTF8.GetString(message), typeof(Utterance));
        if (u == null)
        {
            return;
        }
        if (!string.IsNullOrEmpty(u.rejected))
        {
            SetStatus(string.Format("not understood ({0}), try again", u.rejected));
        }
        else
        {
            SetStatus(string.Format("\"{0}\"\n→ {1}  [{2}, {3:F1} s]", u.text, u.intent, u.language, u.stt_latency_s));
        }
    }

    private void OnDestroy()
    {
        if (isRecording)
        {
            Microphone.End(micDevice);
        }
        Disconnect();
    }

    private void SetStatus(string text)
    {
        Debug.Log("[CarmaVoice] " + text);
        if (statusLabel != null)
        {
            statusLabel.text = text;
        }
    }

    /// <summary>Encode interleaved float samples in [-1, 1] as 16-bit PCM WAV (little-endian).</summary>
    private static byte[] EncodeWav(float[] samples, int channels, int rate)
    {
        int dataBytes = samples.Length * 2;
        using (MemoryStream stream = new MemoryStream(44 + dataBytes))
        using (BinaryWriter writer = new BinaryWriter(stream))
        {
            writer.Write(Encoding.ASCII.GetBytes("RIFF"));
            writer.Write(36 + dataBytes);
            writer.Write(Encoding.ASCII.GetBytes("WAVE"));
            writer.Write(Encoding.ASCII.GetBytes("fmt "));
            writer.Write(16);                           // fmt chunk size
            writer.Write((short)1);                     // PCM
            writer.Write((short)channels);
            writer.Write(rate);
            writer.Write(rate * channels * 2);          // byte rate
            writer.Write((short)(channels * 2));        // block align
            writer.Write((short)16);                    // bits per sample
            writer.Write(Encoding.ASCII.GetBytes("data"));
            writer.Write(dataBytes);
            foreach (float s in samples)
            {
                writer.Write((short)Mathf.RoundToInt(Mathf.Clamp(s, -1f, 1f) * 32767f));
            }
            writer.Flush();
            return stream.ToArray();
        }
    }
}
