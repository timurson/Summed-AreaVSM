
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

uniform sampler2D aoTexture;


void main()
{             
    // retrieve data from texture
    //float AO = texture(aoTexture, TexCoords).r;
	vec3 color = texture(aoTexture, TexCoords).rgb;

	//FragColor = vec4(AO, AO, AO, 1.0);
	FragColor = vec4(color, 1.0);
}