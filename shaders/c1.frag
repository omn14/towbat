#version 150

uniform sampler2D p3d_Texture0;
//uniform sampler2D p3d_Texture1;
uniform vec3 pos;
#define maxpoints 83
uniform vec2 polygonpoints[maxpoints];
uniform bool isActive;
uniform sampler2D bakedMap;

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


// ── Procedural "classic Warhammer Fantasy" grass battle mat ────────────────
float hashG(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float noiseG(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = hashG(i);
    float b = hashG(i + vec2(1.0, 0.0));
    float c = hashG(i + vec2(0.0, 1.0));
    float d = hashG(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbmG(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * noiseG(p); p *= 2.0; a *= 0.5; }
    return v;
}
vec3 battleMat(vec2 uv) {
    vec2 p = uv * 16.0;                          // detail scale across the board
    // Two-tone grass patches — lighter, less-saturated tones.
    float patch = fbmG(p * 0.5);
    vec3 grassDark = vec3(0.22, 0.38, 0.16);
    vec3 grassLite = vec3(0.48, 0.64, 0.34);
    vec3 col = mix(grassDark, grassLite, patch);
    // Worn earth / mud showing through in trampled areas.
    float earth = smoothstep(0.55, 0.85, fbmG(p * 0.3 + 3.1));
    vec3 dirt = vec3(0.46, 0.42, 0.26);
    col = mix(col, dirt, earth * 0.35);
    // Fine blade speckle.
    float speckle = fbmG(p * 2.0);
    col *= 0.90 + 0.20 * speckle;
    // Occasional darker clumps of tall grass.
    col *= 1.0 - 0.12 * smoothstep(0.60, 0.92, fbmG(p * 1.1 + 7.0));
    // Pull saturation down toward luminance for a natural gaming-mat look.
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(lum), col, 0.82);
    // Boost contrast and lift overall brightness.
    col = (col - 0.5) * 1.28 + 0.5 + 0.06;
    return clamp(col, 0.0, 1.0);
}

void main() {
    vec2 uv = texcoord;

    // Base surface is the procedural grass mat.
    vec3 ground = battleMat(uv);

    if (isActive) {
        // Movement / shooting range overlay drawn on top of the grass.
        float d = sdPolygon(uv, polygonpoints);
        vec3 o = (d > 0.0) ? ground : vec3(0.65, 0.85, 1.0);
        o *= 1.0 - exp(-24.0 * abs(d));
        // Bright rim right on the boundary line.
        o = mix(o, vec3(1.0), 1.0 - smoothstep(0.0, 0.015, abs(d) + 0.0075));
        // Keep the indicator a little transparent so the ground shows through.
        p3d_FragColor = vec4(mix(ground, o, 0.9), 1.0);
    } else {
        p3d_FragColor = vec4(ground, 1.0);
    }
}

