#version 150

// Panda3D vertex inputs
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
in vec3 p3d_Normal;

// Outputs to fragment shader
out vec2 texcoord;
out vec3 worldNormal;
out vec3 worldPos;
out vec3 objectPos;

// Panda3D built-in matrices
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    texcoord = p3d_MultiTexCoord0;
    worldNormal = normalize(p3d_NormalMatrix * p3d_Normal);
    worldPos = (p3d_ModelMatrix * p3d_Vertex).xyz;
    objectPos = p3d_Vertex.xyz;
}
