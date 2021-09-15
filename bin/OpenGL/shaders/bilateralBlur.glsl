-- _global

precision highp float;
precision highp int;

layout(rgba32f, binding = 0, location = 0) uniform readonly  image2D uSrc;
layout(rgba32f, binding = 1, location = 1) uniform writeonly image2D uDst;


-- Compute


layout( local_size_x = 128, local_size_y = 1, local_size_z = 1 ) in;

// Blur Information
layout(std140, binding = 7)
uniform BlurData
{
	int Width;          // w
	int Width2;         // 2w
	float Weights[65];  // Weight[2w + 1] = { ... }
} Blur;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform mat4 view;
uniform mat4 projection;
uniform ivec2 direction;
uniform vec2 screenSize;

const float PI = 3.14159265359;

// Shared Workspace (Max w = 32)
shared float v[128 + 64];
shared float d[128 + 64]; // View Depth
shared vec3  n[128 + 64];
shared float D[128 + 64]; // Persp Depth

float calculateD(ivec2 texel)
{
	vec3 worldPos = texelFetch(gPosition, texel, 0).xyz;
	vec3 viewPos = vec3(view * vec4(worldPos, 1.0));
	return viewPos.z;
}

vec3 calculateN(ivec2 texel)
{
	vec3 worldNorm = texelFetch(gNormal, texel, 0).xyz;
	vec3 N = normalize(vec3(view * vec4(worldNorm, 0.0)));
	return N;
}

float calculateDD(ivec2 texel)
{
	float depth = texelFetch(gPosition, texel, 0).w;
	return depth;
}

float RangeKernel(uint baseTexel, uint currTexel)
{
	float StandardDeviation = 5.0;
	float Variance = StandardDeviation * StandardDeviation;
	float N = max(0.0, dot(n[currTexel], n[baseTexel]));
	float D = sqrt(2.0 * PI * Variance);
	float delta = d[currTexel] - d[baseTexel];
	float E = exp(-(delta * delta) / (2.0 * Variance));
	return N * E / D;
}

void main() 
{
	// Note:
	//   WorkGroupSize = (local_size_x,local_size_y,local_size_z)
	//     ^- layout declared at top of compute shader.
	//   WorkGroupId =  [(0,0,0), (num_groups_x,num_groups_y,num_groups_z)]
	//     ^- Parameters passed in from glDispatchCompute().
	//   LocalInvocation = [(0,0,0), (local_size_x-1,local_size_y-1,local_size_z-1]
	//     ^- Essentially the current executing core.
	//
	//   GlobalInvocation = GroupId * GroupSize + LocalInvocation
	ivec2 currTexel = ivec2(gl_GlobalInvocationID.x * direction + gl_GlobalInvocationID.y * (1 - direction));
	vec2 baseUv = vec2(float(currTexel.x), float(currTexel.y)) / screenSize;
	uint texelIndex = gl_LocalInvocationID.x;
	int workWidth = int(gl_WorkGroupSize.x);
	
	// Load image information into temporary workspace
	ivec2 sourceTexel = currTexel - Blur.Width * direction;
	v[texelIndex] = imageLoad(uSrc, sourceTexel).r;
	d[texelIndex] = calculateD(sourceTexel);
	n[texelIndex] = calculateN(sourceTexel);
	D[texelIndex] = calculateDD(sourceTexel);
	
	// First 2w threads will load the last 2w texels.
	if (texelIndex < Blur.Width2)
	{
		ivec2 uv = sourceTexel + workWidth * direction;
		v[texelIndex + workWidth] = imageLoad(uSrc, uv).r;
		d[texelIndex + workWidth] = calculateD(uv);
		n[texelIndex + workWidth] = calculateN(uv);
		D[texelIndex + workWidth] = calculateDD(uv);
	}

	// wait till all work groups catch up.
	barrier();
	
	// Calculate blurred results for each pixel
	float result = 0.0;
	float sum = 0.0;
	uint baseIndex = texelIndex + Blur.Width;

	// If it's the far background, do nothing
	if(D[baseIndex] == 1.0)
	{
		result = 1.0;
		sum = 1.0;
	}

	// Otherwise, it's the foreground, blur.
	else
	{
		for (int j = 0; j <= Blur.Width2; ++j)
		{
			// Don't account for the background
			if (D[texelIndex + j] != 1.0)
			{
				float S = Blur.Weights[j];
				float R = RangeKernel(baseIndex, texelIndex + j);
				float W = S * R;
				result += v[texelIndex + j] * W;
				sum += W;
			}
		}
	}

	imageStore(uDst, currTexel, vec4(result / sum));

}

