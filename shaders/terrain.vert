#version 150

// Standard Panda3D matrices
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ModelViewMatrix;

// Vertex inputs
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_MultiTexCoord0;

// Passed to the fragment shader
out vec3 worldPos;
out vec3 worldNormal;
out vec2 texcoord;
out vec4 eyePos;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;

    vec4 wp = p3d_ModelMatrix * p3d_Vertex;
    worldPos = wp.xyz;
    worldNormal = mat3(p3d_ModelMatrix) * p3d_Normal;
    texcoord = p3d_MultiTexCoord0.xy;
    eyePos = p3d_ModelViewMatrix * p3d_Vertex;
}
