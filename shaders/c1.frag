#version 150

uniform sampler2D p3d_Texture0;
uniform vec3 pos;
#define maxpoints 83
uniform vec2 polygonpoints[maxpoints];

// Input from vertex shader
in vec4 color;
in vec2 texcoord;


// Output to the screen
out vec4 p3d_FragColor;



float sdPolygon( in vec2 p, in vec2[maxpoints] v )
{
    const int num = maxpoints;
    float d = dot(p-v[0],p-v[0]);
    float s = 1.0;
    for( int i=0, j=num-1; i<num; j=i, i++ )
    {
        // distance
        vec2 e = v[j] - v[i];
        vec2 w =    p - v[i];
        vec2 b = w - e*clamp( dot(w,e)/dot(e,e), 0.0, 1.0 );
        d = min( d, dot(b,b) );

        // winding number from http://geomalgorithms.com/a03-_inclusion.html
        bvec3 cond = bvec3( p.y>=v[i].y, 
                            p.y <v[j].y, 
                            e.x*w.y>e.y*w.x );
        if( all(cond) || all(not(cond)) ) s=-s;  
    }
    
    return s*sqrt(d);
}


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
    float radius = 10.1;
    if (dist < radius) {
        //p3d_FragColor = color+vec4(0.5,0.5,0.5,0);
        p3d_FragColor = texture(p3d_Texture0, texcoord)+vec4(0.5,0.5,0.5,0);
        float d = sdPolygon(uv, polygonpoints);
        //vec3 col = (d>0.0) ? vec3(0.9,0.6,0.3) : vec3(0.65,0.85,1.0);
        //vec3 col = (d>0.0) ? vec3(0.9,0.6,0.3) : texture(p3d_Texture0, texcoord).rgb;

        //vec3 col = (d>0.0) ? texture(p3d_Texture0, texcoord).rgb : vec3(0.65,0.85,1.0);
        //vec3 tex = mix( texture(p3d_Texture0, texcoord).rgb, vec3(0.9,1.0,0.8), 0.8);
        vec3 tex = mix( texture(p3d_Texture0, texcoord).rgb, texture(p3d_Texture0, texcoord).bgr, 0.5);
        tex = mix(tex, vec3(0.4, .7, 0.4), 0.6);

        vec3 col = (d>0.0) ? tex : vec3(0.65,0.85,1.0);
        col *= 1.0 - exp(-24.0*abs(d));
        col *= 0.8 + 0.2*cos(2*140.0*d);
        col = mix( col, vec3(1.0), 1.0-smoothstep(0.0,0.015,abs(d)+0.0075) );
        //col = mix(col, vec3(0.7, 1.0, 0.7), 0.6);
        p3d_FragColor = vec4(col,1.0);
    } 
    
    else {
        //discard;
        //p3d_FragColor = texture(p3d_Texture0, texcoord);
        //p3d_FragColor = texture(p3d_Texture0, texcoord);
        p3d_FragColor = color-texture(p3d_Texture0, texcoord)/2.0;

    }
}
