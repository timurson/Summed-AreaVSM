
-- Vertex

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main()
{
	TexCoords = aTexCoords;
	gl_Position = vec4(aPos, 1.0);
}

-- FragmentH

out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D image;
uniform int iteration;

void main()
{
	vec2 imageSize = vec2(textureSize(image,0));
	vec2 step = vec2(1.0/imageSize.x, 1.0/imageSize.y);
	vec4 currentPixel = texture(image, TexCoords);
	vec4 leftPixel = texture(image, vec2(TexCoords.s - exp2(float(iteration)) * step.s, TexCoords.t));
    FragColor = currentPixel + leftPixel;
} 

-- FragmentV

out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D image;
uniform int iteration;

void main()
{
	vec2 imageSize = vec2(textureSize(image,0));
	vec2 step = vec2(1.0/imageSize.x, 1.0/imageSize.y);
	vec4 currentPixel = texture(image, TexCoords);
	vec4 topPixel = texture(image, vec2(TexCoords.s, TexCoords.t - exp2(float(iteration)) * step.t));
    FragColor = currentPixel + topPixel;
} 