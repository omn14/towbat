chargedist_vertex_shader = """
#version 150

// Uniform inputs
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;

// Vertex inputs
in vec4 p3d_Vertex;
in vec4 p3d_Color;
//in vec4 customVertexData;
in vec4 p3d_MultiTexCoord0;
//uniform vec3 pos;

// Output to fragment shader
out vec4 color;
out vec2 texcoord;

void main() {
  gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
  vec4 vertPosWorldCoord = p3d_ModelMatrix * p3d_Vertex;

  

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
"""


chargedist_fragment_shader = """
#version 150

uniform sampler2D p3d_Texture0;
uniform vec3 pos;

// Input from vertex shader
in vec4 color;
in vec2 texcoord;

// Output to the screen
out vec4 p3d_FragColor;

void main() {
    // Get normalized coordinates in [0,1]
    //vec2 uv = gl_FragCoord.xy / vec2(textureSize(p3d_Texture0, 0));
    vec2 uv = texcoord;
    // Center at (0.5, 0.5)
    vec2 pos_uv = ((pos.xy)/1.0+vec2(1.0,1.0))*0.5;
    //vec2 center = uv - vec2(0.5, 0.5);
    vec2 center = uv - (pos_uv );
    float dist = length(center);

    

    // Draw a circle with radius 0.4
    float radius = 0.1;
    if (dist < radius) {
        //p3d_FragColor = color+vec4(0.5,0.5,0.5,0);
        p3d_FragColor = texture(p3d_Texture0, texcoord)+vec4(0.5,0.5,0.5,0);
    } else {
        //discard;
        //p3d_FragColor = texture(p3d_Texture0, texcoord);
        //p3d_FragColor = texture(p3d_Texture0, texcoord);
        p3d_FragColor = color-texture(p3d_Texture0, texcoord)/2.0;
    }
}
"""