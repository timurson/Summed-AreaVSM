
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

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gDiffuse;
uniform sampler2D gSpecular;
uniform sampler2D shadowSAT;
uniform sampler2D ambientOcclusion;
uniform sampler2D shadowMap;
uniform mat4 lightSpaceMatrix;
uniform float shadowSaturation;
uniform float shadowIntensity = 0.2;
uniform int lightSourceRadius = 16;
uniform int blockerSearchSize = 16;
uniform float zNear;
uniform float zFar;
uniform float PenumbraSize = 0.001;
uniform float momentBias = 0.00003;
uniform bool softSATVSM = false;

// IBL
uniform samplerCube environmentMap;
uniform samplerCube irradianceMap;
uniform sampler2D brdfLUT;

struct Light {
    vec3 Position;
    vec3 Color;
	float Radius;
	float Intensity;
};

uniform Light gLight;
uniform vec3 viewPos;
uniform int iblSamples;

const float PI = 3.14159265359;

// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// ----------------------------------------------------------------------------
vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// ----------------------------------------------------------------------------
vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}   

// ----------------------------------------------------------------------------
// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// efficient VanDerCorpus calculation.
float RadicalInverse_VdC(uint bits) 
{
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
     return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint i, uint N)
{
	return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}

// ----------------------------------------------------------------------------
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
	float a = roughness*roughness;
	
	float phi = 2.0 * PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
	// from spherical coordinates to cartesian coordinates - halfway vector
	vec3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;
	
	// from tangent-space H vector to world-space sample vector
	vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent   = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);
	
	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}

// ----------------------------------------------------------------------------
float CalculateShadow(vec3 fragPos, vec3 normal)
{
	vec4 fragPosLightSpace = lightSpaceMatrix * vec4(fragPos, 1.0);
	// perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
	// transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
	// calculate bias
	vec3 lightDir = normalize(gLight.Position - fragPos);
	float bias = max(0.01 * (1.0 - dot(normal, lightDir)), 0.005);
	
	float shadowDepth = texture(shadowSAT, projCoords.xy).r; 
	
	float shadowCoef = projCoords.z - bias > shadowDepth  ? 0.0 : 1.0;
	
	// keep the shadow at 1.0 when outside the zFar region of the light's frustum.
    if(projCoords.z > 1.0)
        shadowCoef = 1.0;
	
	return shadowCoef;
}

float Linstep(float min, float max, float v)  
{
	return clamp ((v - min) / (max - min), 0.0, 1.0);
}  

float ReduceLightBleeding(float p_max, float amount)  
{  
   return Linstep(amount, 1, p_max);  
} 

vec4 ConvertOptimizedMoments(vec4 opt_moments) {
	opt_moments.x -= 0.035955884801f;
	mat4 convert_mat = transpose(mat4(
			0.2227744146f, 0.1549679261f, 0.1451988946f, 0.163127443f,
			0.0771972861f, 0.1394629426f, 0.2120202157f, 0.2591432266f,
			0.7926986636f, 0.7963415838f, 0.7258694464f, 0.6539092497f,
			0.0319417555f,-0.1722823173f,-0.2758014811f,-0.3376131734f
		));
	vec4 converted = convert_mat * opt_moments;
	return converted;
} 



float LinearizeDepth(float depth) {

	depth = (2.0 * zNear) / (zFar + zNear - depth * (zFar - zNear));
	//return (2.0 * zNear * zFar) / (zFar + zNear - z * (zFar - zNear)); // DOUBLE CHECK!!!	
	return depth;
}

float NonLinearize(float depth) 
{
	depth = -(2.0 * zNear - depth * (zFar + zNear)) / (depth * (zFar - zNear));
	return depth;
}

float ChebyshevUpperBound(vec2 moments, float distanceToLight)
{
	if (distanceToLight <= moments.x)
		return 1.0;

	float variance = moments.y - (moments.x * moments.x);
	variance = max(variance,0.0001);
	float d = distanceToLight - moments.x;
	float p_max = variance / (variance + d*d);
	return p_max;
}

float InterleavedGradientNoise(vec2 position_screen)
{
	 vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
	 return fract(magic.z * fract(dot(position_screen, magic.xy)));	
}

vec2 VogelDiskSample(int sampleIndex, int samplesCount, float phi)
{
	float GoldenAngle = 2.4;
	float r = sqrt(sampleIndex + 0.5) / sqrt(samplesCount);
	float theta = sampleIndex * GoldenAngle + phi;
	
	return vec2(r * cos(theta), r * sin(theta));
	
}

float ComputeAverageBlockerDepthBasedOnPCF(vec4 normalizedShadowCoord) 
{
	float averageDepth = 0.0;
	int numberOfBlockers = 0;
	float blockerSearchWidth = float(lightSourceRadius)/float(textureSize(shadowSAT,0).x);
	float stepSize = 2.0 * blockerSearchWidth/float(blockerSearchSize);
	
	float gradientNoise = 2.0 * PI * InterleavedGradientNoise(gl_FragCoord.xy);
	
	for(int i = 0; i < 16; i++)
	{
		vec2 sampleUV = VogelDiskSample(i, 16, gradientNoise);
		float distanceFromLight = texture(shadowMap, vec2(normalizedShadowCoord.xy + sampleUV * stepSize)).x + 0.5;
		if(normalizedShadowCoord.z - 0.01 > distanceFromLight) 
		{
			averageDepth += distanceFromLight;
			numberOfBlockers++;
		}		
	}
	
	if(numberOfBlockers == 0)
		return 1.0;
	else
		return abs(averageDepth) / float(numberOfBlockers);	
}

float ComputePenumbraWidth(float averageDepth, float distanceToLight)
{
	if(averageDepth >= 1.0)
		return 2.0;
	
	//vec2 step = 1.0 / vec2(textureSize(shadowSAT,0));
	
	float penumbraWidth = ((distanceToLight - averageDepth)/averageDepth) * float(lightSourceRadius);
	return penumbraWidth * distanceToLight;	
}

float VSM(float penumbraWidth, vec4 normalizedShadowCoord)
{
	if(penumbraWidth <= 0.0)
		return 1.0;
	
	vec2 step = 1.0 / vec2(textureSize(shadowSAT,0));
	
	float xmax = normalizedShadowCoord.x + penumbraWidth * step.x;
	float xmin = normalizedShadowCoord.x - penumbraWidth * step.x;
	float ymax = normalizedShadowCoord.y + penumbraWidth * step.x;
	float ymin = normalizedShadowCoord.y - penumbraWidth * step.x;

	vec4 A = texture(shadowSAT, vec2(xmin, ymin));
	vec4 B = texture(shadowSAT, vec2(xmax, ymin));
	vec4 C = texture(shadowSAT, vec2(xmin, ymax));
	vec4 D = texture(shadowSAT, vec2(xmax, ymax));
	
	penumbraWidth *= 2.0;
	
	vec4 moments = (D + A - B - C)/float(penumbraWidth * penumbraWidth);
	
	moments.xy += 0.5;
	return clamp(mix(ChebyshevUpperBound(moments.xy, normalizedShadowCoord.z), 1.0, shadowSaturation), 0.0, 1.0);
}

float SummedAreaVarianceShadowMapping(vec4 normalizedShadowCoord)
{
	if(softSATVSM)
	{
		float averageDepth = ComputeAverageBlockerDepthBasedOnPCF(normalizedShadowCoord);
		float penumbraWidth = ComputePenumbraWidth(averageDepth, normalizedShadowCoord.z);
		penumbraWidth = clamp(penumbraWidth, 2.0, penumbraWidth); // This is a hack to eliminate shadow stippling
		return VSM(penumbraWidth, normalizedShadowCoord);	
	}
	else {
		return VSM(PenumbraSize, normalizedShadowCoord);
	}	
}


float CalculateSATShadow(vec3 fragPos)
{
	vec4 fragPosLightSpace = lightSpaceMatrix * vec4(fragPos, 1.0);
	vec4 normalizedShadowCoord = fragPosLightSpace / fragPosLightSpace.w;
	// transform to [0,1] range
    normalizedShadowCoord = normalizedShadowCoord * 0.5 + 0.5;
	
	// keep the shadow at 1.0 when outside the zFar region of the light's frustum.
    if(normalizedShadowCoord.z > 0.99)
        return 1.0;
	
	if(normalizedShadowCoord.x <= 0.01 || normalizedShadowCoord.x >= 0.99)
		return 1.0;
	
	if(normalizedShadowCoord.y <= 0.01 || normalizedShadowCoord.y >= 0.99)
		return 1.0;
	
	return SummedAreaVarianceShadowMapping(normalizedShadowCoord);
}

// ----------------------------------------------------------------------------
vec3 SpecularIBL(vec3 N, vec3 V, float roughness)
{
	vec3 specularLighting = vec3(0.0);
	float totalWeight = 0.0;
	
	for(uint i = 0u; i < iblSamples; ++i)
	{
		// generates a sample vector that's biased towards the preferred alignment direction (importance sampling).
        vec2 Xi = Hammersley(i, iblSamples);
		vec3 H = ImportanceSampleGGX(Xi, N, roughness);
		vec3 L  = normalize(2.0 * dot(V, H) * H - V);
		float NdotL = max(dot(N, L), 0.0);
		if(NdotL > 0.0)
        {
			// sample from the environment's mip level based on roughness/pdf
            float D   = DistributionGGX(N, H, roughness);
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);
            float pdf = D * NdotH / (4.0 * HdotV) + 0.0001; 
			
			float resolution = 512.0; // resolution of source cubemap (per face)
			float saTexel  = 4.0 * PI / (6.0 * resolution * resolution);
            float saSample = 1.0 / (float(iblSamples) * pdf + 0.0001);
			
			// adding a bias of 1.0 as described in GPU Gems 3 Ch20 article
			float mipLevel = roughness == 0.0 ? 0.0 : (0.5 * log2(saSample / saTexel) + 1.0); 
			
			specularLighting += textureLod(environmentMap, L, mipLevel).rgb * NdotL;
			totalWeight      += NdotL;
			
		}
	}
	specularLighting = specularLighting / totalWeight;	
	return specularLighting;
}

void main()
{             
    // retrieve data from gbuffer
    vec3 FragPos = texture(gPosition, TexCoords).rgb;
    vec3 Normal = texture(gNormal, TexCoords).rgb;
    vec4 Diffuse = texture(gDiffuse, TexCoords);
    vec4 Specular = texture(gSpecular, TexCoords);
	
	// the alpha of diffuse represent roughness of material
	float roughness = Diffuse.a;
	float metallic = Specular.a;
	// the alpha of Specular represents how "metallic" surface is.  For dia-electrics the albedo is mostly diffuse, while metals will use specular 
	vec3 albedo = mix(Diffuse.rgb, Specular.rgb, metallic);
	
	// do PBR lighting
	vec3 N = normalize(Normal);
	vec3 V = normalize(viewPos - FragPos);
	
	// calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metallic);
	
	// reflectance equation
    vec3 Lo = vec3(0.0);
	// calculate per-light radiance
	vec3 L = normalize(gLight.Position - FragPos);
	vec3 H = normalize(V + L);
	float distance = length(gLight.Position - FragPos);
	float attenuation = 1.0 / (distance * distance);
	vec3 radiance = gLight.Intensity * gLight.Color * attenuation;
	
	// Cook-Torrance BRDF
	float NDF = DistributionGGX(N, H, roughness);   
	float G   = GeometrySmith(N, V, L, roughness);    
	vec3 F    = FresnelSchlick(max(dot(H, V), 0.0), F0);  

	vec3 nominator    = NDF * G * F;
	float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; // 0.001 to prevent divide by zero.
	vec3 specular = nominator / denominator;

	// kS is equal to Fresnel
	vec3 kS = F;
	// for energy conservation, the diffuse and specular light can't
	// be above 1.0 (unless the surface emits light); to preserve this
	// relationship the diffuse component (kD) should equal 1.0 - kS.
	vec3 kD = vec3(1.0) - kS;
	// multiply kD by the inverse metalness such that only non-metals 
	// have diffuse lighting, or a linear blend if partly metal (pure metals
	// have no diffuse light).
	kD *= 1.0 - metallic;	

	// scale light by NdotL
	float NdotL = max(dot(N, L), 0.0);        

	// add to outgoing radiance Lo
	// note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
	Lo += (kD * albedo / PI + specular) * radiance * NdotL;
	
	// ambient lighting (use IBL as the ambient term)
    F = FresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
	
	kS = F;
    kD = 1.0 - kS;
    kD *= 1.0 - metallic;
	
	vec3 irradiance = texture(irradianceMap, N).rgb;
	vec3 diffuse      = irradiance * albedo;
	
	vec3 specularIBL = SpecularIBL(N, V, roughness);
	vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    specular = specularIBL * (F * brdf.x + brdf.y);
	
	// calculate shadow using Moment Shadow Map
	float shadowFactor = CalculateSATShadow(FragPos);
		
	// use linear step function to reduce light bleeding more
	shadowFactor = ReduceLightBleeding(shadowFactor, 0.25);	
	
	float AO = texture(ambientOcclusion, TexCoords).r;
	
	// need to saturate shadows quite a bit to make them more plausable
	shadowFactor = mix(shadowFactor, 1.0, shadowSaturation);
	vec3 ambient = (kD * diffuse + specular) * shadowFactor * AO;
	
	vec3 color = ambient + Lo;
	
	// HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correction
    color = pow(color, vec3(1.0/2.2)); 
	
    FragColor = vec4(color , 1.0);
}