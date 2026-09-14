using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.TextCore.Text;
using UnityEngine.UI;
using static System.Net.Mime.MediaTypeNames;

public class CameraScript : MonoBehaviour
{
    int currentCamIndex = 1;
    WebCamTexture tex;
    public RawImage display;
    public string notification = "";
    public UnityEngine.UI.Text textElement;

    void Start()
    {
        notification = string.Concat("cameras found", WebCamTexture.devices.Length);
        textElement.text = notification;
    }
    public void SwapCamClick()
    {
        if (WebCamTexture.devices.Length == 0)
        {
            UnityEngine.Debug.Log("No Camera detected");
            notification = "No camera detected";
        }
        if (WebCamTexture.devices.Length > 0)
        {
            currentCamIndex += 1;
            if (currentCamIndex >= WebCamTexture.devices.Length)
            {
                currentCamIndex = 0;
            }

        }
        if (tex != null)
        {
            StopWebCam();
            StartStop_CamClick();
            //notification = string.Concat("Camera: ", currentCamIndex);
        }




        textElement.text = notification;
    }
    public void StartStop_CamClick()
    {
        if (tex != null)
        {
            StopWebCam();

        }
        else
        {
            WebCamDevice device = WebCamTexture.devices[currentCamIndex];
            tex = new WebCamTexture(device.name);
            display.texture = tex;
            tex.Play();
            notification = string.Concat("Camera ON: ", currentCamIndex);

        }
        textElement.text = notification;





    }

    private void StopWebCam()
    {
        display.texture = null;
        tex.Stop();
        tex = null;
        notification = "Camera Off";
    }

}