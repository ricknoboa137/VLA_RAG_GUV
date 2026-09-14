using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;

public class mqttController : MonoBehaviour
{
    public string nameController = "Controller 1";
    public string tagOfTheMQTTReceiver = "";
    public mqttReceiver _eventSender;
    

    void Start()
    {
        _eventSender = GameObject.FindGameObjectsWithTag(tagOfTheMQTTReceiver)[0].gameObject.GetComponent<mqttReceiver>();
        _eventSender.OnMessageArrived += OnMessageArrivedHandler;
    }
    int a = 0;
    private void OnMessageArrivedHandler(string newMsg)
    {
        //UnityEngine.Debug.Log("Event Fired. The message, from Object " + nameController + " is = " + newMsg);
        a++;
    }





    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Function to get the input form controllers and headset rotation and send it over MQTT brocker
    /// </summary>
    /// 
    public OVRCameraRig cameraRig;
    void Update()
    {
        string source;
        if (OVRInput.GetDown(OVRInput.Button.One))
        {
            print("button A pressed");
        }
        //print(OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick)[0]);
        source = string.Concat("JoystickHor: ", OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick)[0]);
        //print(source);
        
        //print(OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick)[1]);
        source = string.Concat("JoystickVer: ", OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick)[1]);
        //print(source);
        Vector3 headsetPosition = cameraRig.centerEyeAnchor.position;
        Quaternion headsetRotation = cameraRig.centerEyeAnchor.rotation;
        source = string.Concat("HeadPosition: ",headsetPosition);
        //print(source);
        
        source = string.Concat("HeadRotation: ", headsetRotation);
        //print(source);
        source = string.Concat(headsetRotation[0], ",", headsetRotation[1]);
        source = string.Concat(source,",", OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick)[0]);
        source = string.Concat(source,",", OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick)[1]);
        if (OVRInput.Get(OVRInput.Button.One))
        {
            source = string.Concat(source, ",","1.0");
        }
        else
        {
            source = string.Concat(source, ",", "0");
        }
        //print(source);
        _eventSender.messagePublish = source;
        _eventSender.topicPublish = "test1";
        _eventSender.Publish();

    }
}