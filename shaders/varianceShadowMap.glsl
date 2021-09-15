
-- Vertex

layout (location = 0) in vec3 aPos;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main()
{
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}

-- Fragment

out vec4 FragColor;

void main()
{
    float depth = gl_FragCoord.z;	
    float squared = depth * depth;
	
    vec4 moment = vec4(0);
    moment.x = depth;
    moment.y = squared;
    float dx = dFdx(depth);
	float dy = dFdy(depth);
	moment.x -= 0.5;
	moment.y += 0.25 * (dx * dx + dy * dy);
	moment.y -= 0.5;	
    FragColor = moment;
}