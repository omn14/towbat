#version 150

// Standard Panda3D matrices
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ModelViewMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;

out vec3 worldPos;
out vec3 worldNormal;
out vec4 vertColor;
// Height above the model's own origin, for the ambient darkening at its foot.
out float localZ;
out vec4 eyePos;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;

    vec4 wp = p3d_ModelMatrix * p3d_Vertex;
    worldPos = wp.xyz;
    worldNormal = mat3(p3d_ModelMatrix) * p3d_Normal;
    vertColor = p3d_Color;
    localZ = p3d_Vertex.z;
    eyePos = p3d_ModelViewMatrix * p3d_Vertex;
}
