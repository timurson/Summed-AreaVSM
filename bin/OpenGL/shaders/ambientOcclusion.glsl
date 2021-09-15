
-- Vertex

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main()
{
    TexCoords = aTexCoords;
    gl_Position = vec4(aPos, 1.0);
}

-- Fragment

out highp vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;

uniform mat4 view;
uniform float sampleRadius = 1.0;
uniform int aoSamples = 20;
uniform int sampleTurns = 16;
uniform float depthThreshold = 0.0005;
uniform float shadowScalar = 1.3;
uniform float shadowContrast = 0.5;

const float PI = 3.14159265359;

float randAngle()
{
	uint x = uint(gl_FragCoord.x);
	uint y = uint(gl_FragCoord.y);
	return (30u * x ^ y + 10u * x * y);
}


void main()
{
	// transform position and normal to view space
	vec3 worldPos = texture(gPosition, TexCoords).xyz;
	vec3 worldNorm = texture(gNormal, TexCoords).xyz;
	vec3 P = vec3(view * vec4(worldPos, 1.0));
	vec3 N = normalize(vec3(view * vec4(worldNorm, 0.0)));
	
	float aoValue = 0.0;
	float perspectiveRadius = (sampleRadius * 100.0 / P.z);
	
	int max_mip = textureQueryLevels(gPosition) - 1;
	const float TAU = 2.0 * PI;
	ivec2 px = ivec2(gl_FragCoord.xy);
	
	// Perform random sampling and estimate ambient occlusion for the current fragment
	for (int i = 0; i < aoSamples; ++i)
	{
		// Alchemy helper variables
		float alpha = 1.f / aoSamples * (i + 0.5);
		float h = perspectiveRadius * alpha;
		float theta = TAU * alpha * sampleTurns + randAngle();
		vec2 u = vec2(cos(theta), sin(theta));
		// McGuire paper MIP calculation
		int m = clamp(findMSB(int(h)) - 4, 0, max_mip);
		ivec2 mip_pos = clamp((ivec2(h * u) + px) >> m, ivec2(0), textureSize(gPosition, m) - ivec2(1));
		
		vec3 worldPi = texelFetch(gPosition, mip_pos, m).xyz;
		vec3 Pi = vec3(view * vec4(worldPi, 1.0));
		vec3 V = Pi - P;
		float sqrLen    = dot(V, V);
		float Heaveside = step(sqrt(sqrLen), sampleRadius);
		float dD = depthThreshold * P.z;
		float c = 0.1 * sampleRadius;
								  
		// Summation of Obscurance Factor
		aoValue += (max(0.0, dot(N, V) + dD) * Heaveside) / (max(c * c, sqrLen) + 0.0001);
	  
	}
	
	// Final scalar multiplications for averaging and intensifying shadowing
	aoValue *= (2.0 * shadowScalar) / aoSamples;
	aoValue = max(0.0, 1.0 - pow(aoValue, shadowContrast));
	
    FragColor = vec4(aoValue, aoValue, aoValue, 1.0);
}