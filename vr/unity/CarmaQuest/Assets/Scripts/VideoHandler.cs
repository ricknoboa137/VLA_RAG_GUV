using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.TextCore.Text;
using UnityEngine.UI;

public class VideoHandler : MonoBehaviour
{
    public RawImage display;
    // Start is called before the first frame update
    void Start()
    {
        StartCoroutine(GetTexture());
    }

    // Update is called once per frame
    /*   void Update()
       {

       }
    */

    IEnumerator GetTexture()
    {
        print("GEting texture");
        UnityWebRequest www = UnityWebRequest.Get("http://192.168.0.108:5000/video_feed");
        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            UnityEngine.Debug.Log(www.error);
        }
        else
        {
            // Show results as text
            UnityEngine.Debug.Log(www.downloadHandler.text);

            // Or retrieve results as binary data
            byte[] results = www.downloadHandler.data;
            print(results);
        }

    }

    /*
     * Alternatively, you can implement GetTexture using a helper getter:
    IEnumerator GetTexture() {
        UnityWebRequest www = UnityWebRequest.GetTexture("http://192.168.0.108:5000/video_feed.jpeg");
        yield return www.Send();

        Texture myTexture = DownloadHandlerTexture.GetContent(www);
    }*/
}
