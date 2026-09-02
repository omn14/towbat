#version 150

// Uniform inputs
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ModelViewMatrix;

// Vertex inputs
in vec4 p3d_Vertex;
in vec4 p3d_Color;
//in vec4 customVertexData;
in vec4 p3d_MultiTexCoord0;
//uniform vec3 pos;

// Output to fragment shader
out vec4 color;
out vec2 texcoord;
out vec4 eyePos;

void main() {
  gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
  vec4 vertPosWorldCoord = p3d_ModelMatrix * p3d_Vertex;
  eyePos = p3d_ModelViewMatrix * p3d_Vertex;

  

  texcoord = p3d_MultiTexCoord0.xy; // Access the texture coordinate
  vec3 pos = vec3(0.0,0.0,0.0); // Temporary static position

  if (abs(vertPosWorldCoord.z-pos.z)>4 || abs(vertPosWorldCoord.x-pos.x)>4 ){
  
  color = p3d_Color;
  //color = customVertexData;
  }
  else{
  color = p3d_Color;
  //color = customVertexData;
  }

  if (abs(vertPosWorldCoord.z-pos.z)<1 || abs(vertPosWorldCoord.x-pos.x)<1 ){
  
  color = p3d_Color;
  //color = customVertexData;
  }


  

}