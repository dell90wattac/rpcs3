R"(
#version 450

// vanillad1: X467.4 p28: SMAA 1x as three compute passes (SMAA by Jorge Jimenez et al., MIT: see smaa_core.inc).
// The definitions below set SMAA_PRESET_* and SMAA_VD1_PASS (0 edges, 1 blending weights, 2 neighbourhood blending).
// One invocation a pixel; each pass writes every pixel of its target (the edge pass writes "no edge" where the pixel
// shader would discard), so no stencil or clear is needed.

layout(local_size_x = 8, local_size_y = 8) in;

%SMAA_DEFINITIONS%

#define SMAA_GLSL_4 1
#define SMAA_RT_METRICS smaa_rt_metrics

layout(%push_block%) uniform smaa_params
{
	vec4 smaa_rt_metrics; // 1/w, 1/h, w, h of the source (all three targets are its size)
};

%SMAA_CORE%

#if SMAA_VD1_PASS == 0
layout(%set_binding%0) uniform sampler2D colorTex;             // the frame, linear
layout(%set_binding%1, rgba8) uniform writeonly image2D edgesOut;

void smaa_vd1_pixel(ivec2 pos, vec2 texcoord)
{
	vec4 offset[3];
	SMAAEdgeDetectionVS(texcoord, offset);
	vec2 e = SMAAColorEdgeDetectionPS(texcoord, offset, colorTex);
	imageStore(edgesOut, pos, vec4(e, 0.0, 0.0));
}
#elif SMAA_VD1_PASS == 1
layout(%set_binding%0) uniform sampler2D edgesTex;             // linear
layout(%set_binding%1) uniform sampler2D areaTex;              // linear
layout(%set_binding%2) uniform sampler2D searchTex;            // point
layout(%set_binding%3, rgba8) uniform writeonly image2D blendOut;

void smaa_vd1_pixel(ivec2 pos, vec2 texcoord)
{
	vec2 pixcoord;
	vec4 offset[3];
	SMAABlendingWeightCalculationVS(texcoord, pixcoord, offset);
	imageStore(blendOut, pos, SMAABlendingWeightCalculationPS(texcoord, pixcoord, offset, edgesTex, areaTex, searchTex, vec4(0.0)));
}
#else
layout(%set_binding%0) uniform sampler2D colorTex;             // the frame, linear
layout(%set_binding%1) uniform sampler2D blendTex;             // linear
layout(%set_binding%2, rgba8) uniform writeonly image2D colorOut;

void smaa_vd1_pixel(ivec2 pos, vec2 texcoord)
{
	vec4 offset;
	SMAANeighborhoodBlendingVS(texcoord, offset);
	imageStore(colorOut, pos, SMAANeighborhoodBlendingPS(texcoord, offset, colorTex, blendTex));
}
#endif

void main()
{
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	if (pos.x >= int(smaa_rt_metrics.z) || pos.y >= int(smaa_rt_metrics.w))
	{
		return;
	}

	smaa_vd1_pixel(pos, (vec2(pos) + 0.5) * smaa_rt_metrics.xy);
}
)"
