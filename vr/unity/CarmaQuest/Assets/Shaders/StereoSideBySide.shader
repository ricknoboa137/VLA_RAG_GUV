// Draws a side-by-side stereo frame on a single UI panel: the left half of the
// texture to the left eye, the right half to the right eye.
//
// The obvious approach - one panel per eye on its own layer, drawn by that
// eye's camera - is what the original project does and it renders nothing
// under OpenXR, where one camera produces both eyes. Selecting the half in the
// shader from unity_StereoEyeIndex works in both single-pass instanced and
// multi-pass, and needs only one panel.
//
// _Stereo = 0 shows the whole texture to both eyes, which is correct for a
// mono camera. CarmaStereoDisplay sets it from the publisher's layout.
Shader "CARMA/StereoSideBySide"
{
    Properties
    {
        _MainTex ("Frame", 2D) = "black" {}
        _Color ("Tint", Color) = (1,1,1,1)
        _Stereo ("Side by side", Float) = 0
        _SwapEyes ("Swap eyes", Float) = 0

        // Stencil and clipping properties the UI system sets on masked
        // elements; without them a RawImage inside a Mask renders wrongly.
        _StencilComp ("Stencil Comparison", Float) = 8
        _Stencil ("Stencil ID", Float) = 0
        _StencilOp ("Stencil Operation", Float) = 0
        _StencilWriteMask ("Stencil Write Mask", Float) = 255
        _StencilReadMask ("Stencil Read Mask", Float) = 255
        _ColorMask ("Color Mask", Float) = 15
    }

    SubShader
    {
        Tags
        {
            "Queue" = "Transparent"
            "IgnoreProjector" = "True"
            "RenderType" = "Transparent"
            "PreviewType" = "Plane"
            "CanUseSpriteAtlas" = "True"
        }

        Stencil
        {
            Ref [_Stencil]
            Comp [_StencilComp]
            Pass [_StencilOp]
            ReadMask [_StencilReadMask]
            WriteMask [_StencilWriteMask]
        }

        Cull Off
        Lighting Off
        ZWrite Off
        ZTest [unity_GUIZTestMode]
        Blend SrcAlpha OneMinusSrcAlpha
        ColorMask [_ColorMask]

        Pass
        {
            Name "Default"
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 2.0
            #pragma multi_compile_instancing

            #include "UnityCG.cginc"
            #include "UnityUI.cginc"

            struct appdata_t
            {
                float4 vertex : POSITION;
                float4 color : COLOR;
                float2 texcoord : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                fixed4 color : COLOR;
                float2 texcoord : TEXCOORD0;
                float4 worldPosition : TEXCOORD1;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            fixed4 _Color;
            float _Stereo;
            float _SwapEyes;
            float4 _ClipRect;

            v2f vert(appdata_t v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.worldPosition = v.vertex;
                o.vertex = UnityObjectToClipPos(o.worldPosition);
                o.texcoord = TRANSFORM_TEX(v.texcoord, _MainTex);
                o.color = v.color * _Color;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                float2 uv = i.texcoord;
                if (_Stereo > 0.5)
                {
                    // 0 for the left eye, 1 for the right.
                    float eye = unity_StereoEyeIndex;
                    if (_SwapEyes > 0.5)
                    {
                        eye = 1.0 - eye;
                    }
                    uv.x = uv.x * 0.5 + eye * 0.5;
                }

                fixed4 col = tex2D(_MainTex, uv) * i.color;
                col.a *= UnityGet2DClipping(i.worldPosition.xy, _ClipRect);
                clip(col.a - 0.001);
                return col;
            }
            ENDCG
        }
    }
}
