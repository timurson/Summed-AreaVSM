
-- Vertex

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

uniform mat4 transform;

void main()
{
    TexCoords = aTexCoords;
    gl_Position = transform * vec4(aPos, 1.0);
}

-- Fragment

out vec4 FragColor;

in vec2 TexCoords;

uniform samplerCube cubeMap;

const float PI = 3.14159265359;

void main()
{
	float phi = (TexCoords.s) * PI * 2.0;
	float theta = (1.0 - TexCoords.t + 0.5) * PI;

	vec3 sampleDir = vec3(cos(phi) * cos(theta), sin(theta), sin(phi) * cos(theta));
    FragColor = texture(cubeMap, sampleDir);
}