#version 150

uniform sampler2D p3d_Texture0;
//uniform sampler2D p3d_Texture1;
uniform vec3 pos;
#define maxpoints 83
uniform vec2 polygonpoints[maxpoints];
uniform bool isActive;
uniform sampler2D bakedMap;

// The board rectangle in this card's UV space. The card is deliberately larger
// than the playing surface because it is also the coordinate space the overlay
// polygon is expressed in, so the grass has to be clipped rather than resized.
uniform vec2 boardMin;
uniform vec2 boardMax;

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
// Hoskins hash12: sin() based hashes lose precision on some drivers and leave
// the value-noise lattice visible as a grid of squares.
float hashG(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float noiseG(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = hashG(i);
    float b = hashG(i + vec2(1.0, 0.0));
    float c = hashG(i + vec2(0.0, 1.0));
    float d = hashG(i + vec2(1.0, 1.0));
    // Quintic fade: smoothstep's second derivative jumps at the cell borders,
    // which the contrast boost below turns into visible seams.
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbmG(vec2 p) {
    // Rotating each octave keeps the lattices from stacking up axis-aligned.
    const mat2 rot = mat2(0.80, 0.60, -0.60, 0.80);
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * noiseG(p); p = rot * p * 2.03; a *= 0.5; }
    return v;
}
vec3 battleMat(vec2 uv) {
    vec2 p = uv * 16.0;                          // detail scale across the board

    // A printed gaming mat is close to even. Anything more than a few percent
    // of swing between light and shade reads as camouflage, not as grass.
    float broad = fbmG(p * 0.8);
    vec3 col = mix(vec3(0.31, 0.39, 0.18), vec3(0.41, 0.49, 0.25),
                   smoothstep(0.35, 0.68, broad));

    // Sparse bleached-straw patches, soft edged.
    float straw = smoothstep(0.70, 0.96, fbmG(p * 1.1 + 11.7));
    col = mix(col, vec3(0.50, 0.49, 0.31), straw * 0.14);

    // Worn earth, rarer and weaker than the straw.
    float earth = smoothstep(0.78, 0.99, fbmG(p * 1.6 + 3.1));
    col = mix(col, vec3(0.42, 0.36, 0.24), earth * 0.12);

    // Flock grain. Fine enough to read as fabric rather than as a painted
    // surface, but not so fine that minification averages it flat.
    float fine = noiseG(p * 30.0) - 0.5;
    float mid = noiseG(p * 10.0) - 0.5;
    col *= 1.0 + 0.16 * fine + 0.08 * mid;

    // Matt rather than poster green, but it is still grass: take only the
    // edge off the saturation.
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(lum), col, 0.95);

    // The mat dips into shadow where it meets the table edge.
    vec2 span = max(boardMax - boardMin, vec2(1e-4));
    vec2 e = min(uv - boardMin, boardMax - uv) / span;
    col *= mix(0.80, 1.0, smoothstep(0.0, 0.05, min(e.x, e.y)));

    return clamp(col, 0.0, 1.0);
}

void main() {
    vec2 uv = texcoord;

    // Off the board is tabletop, not battlefield: let it show through.
    if (uv.x < boardMin.x || uv.x > boardMax.x ||
        uv.y < boardMin.y || uv.y > boardMax.y) {
        discard;
    }

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

