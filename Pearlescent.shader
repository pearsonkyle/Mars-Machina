// Upgrade NOTE: upgraded instancing buffer 'Props' to new syntax.

Shader "Custom/Pearlescent" {
	Properties {
		_Color ("Color", Color) = (1,1,1,1)
		_ColorDirect1("Indirect Color 1", Color) = (1,1,1,1)
		_ColorDirect2("Indirect Color 2", Color) = (1,1,1,1)
		_ColorMix("Color Tolerance", Range(0,1)) = 0.5
		//_MainTex ("Albedo (RGB)", 2D) = "white" {}
		_Glossiness ("Smoothness", Range(0,1)) = 0.5
		_Metallic ("Metallic", Range(0,1)) = 0.0
		_Iterations("Iridescence Function Iterations (Power)", Range(0,20)) = 5
	}
	SubShader {
		Tags { "RenderType"="Opaque" }
		LOD 200
		
		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf Standard fullforwardshadows

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0

		sampler2D _MainTex;

		struct Input {
			float2 uv_MainTex;
			float3 viewDir;
		};

		half _Glossiness;
		half _Metallic;
		fixed4 _Color;
		fixed4 _ColorDirect1;
		fixed4 _ColorDirect2;

		float _ColorMix;
		float _Iterations;

		// Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
		// See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
		// #pragma instancing_options assumeuniformscaling
		UNITY_INSTANCING_BUFFER_START(Props)
			// put more per-instance properties here
		UNITY_INSTANCING_BUFFER_END(Props)

		float map(float val, float min, float max, float newMin, float newMax)
		{
			float ratio = abs(val - min) / abs(max - min);
			return newMin + ratio * abs(newMax - newMin);
		}

		float4 mix(float4 col1, float4 col2, float4 col3, float mixAmount)
		{
			//mixAmount = abs(mixAmount);
			
			if (mixAmount < _ColorMix){
				return float4(col1.r * mixAmount + col2.r * (1 - mixAmount), col1.g * mixAmount + col2.g * (1 - mixAmount), col1.b * mixAmount + col2.b * (1 - mixAmount), col1.a * mixAmount + col2.a * (1 - mixAmount));				
			}
			else //if (mixAmount >= 0.5)
			{
				return float4(col2.r * mixAmount + col3.r * (1 - mixAmount), col2.g * mixAmount + col3.g * (1 - mixAmount), col2.b * mixAmount + col3.b * (1 - mixAmount), col2.a * mixAmount + col3.a * (1 - mixAmount));
			}
			
		}

		float efficientIridFunc(float x) {
			return x*x*x*x*x;
		}
		
		float iridFunc(float x)
		{
			if (x < 0) x = 0; //get rid of the black outline caused by a thin layer of almost-perpendicular pixels
			float result = pow(x, _Iterations);
			return result;
		}

		void surf (Input IN, inout SurfaceOutputStandard o) {
			
			float3 worldNormal = WorldNormalVector(IN, o.Normal);

			float dotProduct = dot(worldNormal, IN.viewDir); //will always be between 1 and 0 for the pixels that we care about because normals facing away will be 0 to -1. I hope I don't have these numbers the wrong way around...
			//float rampPos = map(dotProduct, -1, 1, 0, 1);

			float4 col;

			//col = mix(_Color, _ColorDirect, efficientIridFunc(dotProduct));
			col = mix(_Color, _ColorDirect1, _ColorDirect2, iridFunc(dotProduct));

			o.Albedo = col;
			// Metallic and smoothness come from slider variables
			o.Metallic = _Metallic;
			o.Smoothness = _Glossiness;
			//o.Alpha = c.a;
		}
		ENDCG
	}
	FallBack "Diffuse"
}