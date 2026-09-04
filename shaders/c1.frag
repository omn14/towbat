#version 150

uniform sampler2D p3d_Texture0;
//uniform sampler2D p3d_Texture1;
uniform vec3 pos;
#define maxpoints 83
uniform vec2 polygonpoints[maxpoints];
uniform bool isActive;
uniform vec3 overlayColor;
uniform sampler2D bakedMap;

// The board rectangle in this card's UV space. The card is deliberately larger
// than the playing surface because it is also the coordinate space the overlay
// polygon is expressed in, so the grass has to be clipped rather than resized.
uniform vec2 boardMin;
uniform vec2 boardMax;

// Input from vertex shader
in vec4 color;
in vec2 texcoord;
in vec4 eyePos;


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


// ── Grass battle mat ─────────────────────────────────────────────
// Baked once at startup by bake_mat.frag, which runs the same matGrass the
// terrain uses. Nothing here is recomputed per frame: the mat never changes,
// and its grain is below a pixel at every camera distance the game uses, so
// the mipmap chain fades it exactly as the live grainFade used to.
uniform sampler2D matTex;
#pragma include "shadow.glsl"

vec3 battleMat(vec2 uv) {
    vec3 col = texture(matTex, (uv - boardMin) / (boardMax - boardMin)).rgb;

    // The mat dips into shadow where it meets the table edge. Left live rather
    // than baked: it costs almost nothing, and baking it puts a dark rim in
    // the outermost texels for the mipmaps to smear back inwards.
    vec2 span = max(boardMax - boardMin, vec2(1e-4));
    vec2 e = min(uv - boardMin, boardMax - uv) / span;
    col *= mix(0.80, 1.0, smoothstep(0.0, 0.05, min(e.x, e.y)));

    return col;
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

    // Everything standing on the board drops its shadow here.
    ground = shadeSun(ground, sunShadow(eyePos));

    if (isActive) {
        // Movement / shooting range overlay drawn on top of the grass.
        float d = sdPolygon(uv, polygonpoints);
        vec3 o = (d > 0.0) ? ground : overlayColor;
        o *= 1.0 - exp(-24.0 * abs(d));
        // Bright rim right on the boundary line.
        o = mix(o, vec3(1.0), 1.0 - smoothstep(0.0, 0.015, abs(d) + 0.0075));
        // Keep the indicator a little transparent so the ground shows through.
        p3d_FragColor = vec4(mix(ground, o, 0.9), 1.0);
    } else {
        p3d_FragColor = vec4(ground, 1.0);
    }
}

