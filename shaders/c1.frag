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
    // Normalised, or a threshold does not mean what it looks like: the raw sum
    // tops out at 0.96875 and sits mostly between 0.35 and 0.65.
    return v / 0.96875;
}

// Grain, signed and centred on zero. Three octaves rather than one: a single
// octave of value noise shows its lattice as soft blocks the moment a cell is
// only a few pixels across, which is what made the mat look pixellated.
float grainG(vec2 p) {
    const mat2 rot = mat2(0.80, 0.60, -0.60, 0.80);
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 3; i++) { v += a * noiseG(p); p = rot * p * 2.11; a *= 0.5; }
    return v / 0.875 - 0.5;
}

// Fade a grain layer out before its cells reach pixel size, or it aliases into
// blocks when the camera pulls back.
float grainFade(vec2 p) {
    return 1.0 - smoothstep(0.30, 0.90, max(fwidth(p.x), fwidth(p.y)));
}

// Discrete specks — the flock fibres and little tufts of a printed mat. Value
// noise is smooth at every scale, so on its own it can only ever look like
// cloud; crisp detail has to come from something that is actually discrete.
// One seeded point per cell, with a round falloff so the cells never show.
float specks(vec2 p, float seed, float size) {
    vec2 i = floor(p), f = fract(p);
    float best = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 g = vec2(float(x), float(y));
            vec2 c = i + g + seed;
            vec2 o = vec2(hashG(c), hashG(c + 19.7));
            best = max(best, 1.0 - smoothstep(0.0, size, length(f - g - o)));
        }
    }
    return best;
}
vec3 battleMat(vec2 uv) {
    vec2 p = uv * 16.0;                          // detail scale across the board

    // One base colour with small multiplicative shading. Mixing between two
    // different greens gave big soft blobs; a printed mat has almost no
    // large-scale variation, only fine texture.
    // The base is a warm olive: the reference mat is nearer khaki than grass
    // green, with yellow and brown showing through it.
    vec3 col = vec3(0.400, 0.440, 0.230);
    col *= 1.0 + 0.11 * (fbmG(p * 3.0) - 0.5)
               + 0.08 * (fbmG(p * 9.0) - 0.5);

    // Warm drift: some stretches of the mat read yellow, others stay green.
    float warm = smoothstep(0.42, 0.70, fbmG(p * 1.7 + 5.3));
    col = mix(col, vec3(0.54, 0.49, 0.25), warm * 0.40);

    // Sparse bleached-straw patches, soft edged.
    float straw = smoothstep(0.52, 0.76, fbmG(p * 1.1 + 11.7));
    col = mix(col, vec3(0.62, 0.56, 0.33), straw * 0.38);

    // Worn earth, browner and rarer than the straw.
    float earth = smoothstep(0.58, 0.84, fbmG(p * 1.6 + 3.1));
    col = mix(col, vec3(0.42, 0.31, 0.18), earth * 0.42);

    // Flock grain: what reads as fabric rather than as a painted surface.
    vec2 gp = p * 26.0;
    col *= 1.0 + 0.09 * grainG(gp) * grainFade(gp)
               + 0.05 * grainG(p * 7.0);

    // Fibres and tufts. Without these the mat is all smooth gradient and reads
    // as airbrushed paint; the photograph is full of small discrete detail.
    // Skipped once they have faded out: three speck layers is 54 hashes a
    // pixel, and the branch is coherent because the fade follows the zoom.
    vec2 sp = p * 42.0;
    float fade = grainFade(sp);
    if (fade > 0.01) {
        // Clumped rather than evenly peppered: an even scatter of specks reads
        // as freckles, where real flock gathers and thins.
        float clump = 0.30 + 0.70 * smoothstep(0.28, 0.72, fbmG(p * 5.0 + 31.0));
        float a = fade * clump;
        col = mix(col, vec3(0.31, 0.35, 0.19),
                  specks(sp, 0.0, 0.66) * 0.20 * a);
        col = mix(col, vec3(0.52, 0.51, 0.32),
                  specks(sp * 1.37, 7.1, 0.54) * 0.17 * a);
        // Brown flecks, so the warmth is in the detail and not only in the
        // broad patches.
        col = mix(col, vec3(0.44, 0.35, 0.22),
                  specks(sp * 0.83, 23.9, 0.44) * 0.15 * a);
    }
    // Sparser, larger tufts, which survive further out than the fibres do.
    vec2 tp = p * 15.0;
    float tufts = grainFade(tp);
    if (tufts > 0.01) {
        col = mix(col, vec3(0.33, 0.37, 0.19),
                  specks(tp, 3.4, 0.34) * 0.15 * tufts);
    }

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

