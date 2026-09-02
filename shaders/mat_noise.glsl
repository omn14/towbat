// Shared material vocabulary for the board surface and the terrain that sits
// on it. Included by c1.frag and terrain.frag so a hill is made of the same
// stuff as the mat it stands on, and so a fix lands in one place — the two
// files previously carried separate copies, and the terrain one still had the
// sin() hash, the cubic fade and the unrotated octaves after the ground's copy
// had been fixed.

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
    // which any contrast boost downstream turns into visible seams.
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

// Discrete specks — flock fibres, tufts, individual leaves. Value noise is
// smooth at every scale, so on its own it can only ever look like cloud; crisp
// detail has to come from something that is actually discrete. One seeded
// point per cell, with a round falloff so the cells never show.
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

// The board card spans 100 world units and drives matGrass from uv * 16, so
// anything standing on the board must use this scale or its grain will not
// match the mat underneath it.
const float MAT_WORLD_SCALE = 0.16;

// The grass of the battle mat, in world units. Terrain that is meant to be
// grassy calls this so it matches the board rather than merely coordinating
// with it.
vec3 matGrass(vec2 p) {
    vec3 col = vec3(0.400, 0.440, 0.230);
    col *= 1.0 + 0.11 * (fbmG(p * 3.0) - 0.5)
               + 0.08 * (fbmG(p * 9.0) - 0.5);

    float warm = smoothstep(0.42, 0.70, fbmG(p * 1.7 + 5.3));
    col = mix(col, vec3(0.54, 0.49, 0.25), warm * 0.40);

    float straw = smoothstep(0.52, 0.76, fbmG(p * 1.1 + 11.7));
    col = mix(col, vec3(0.62, 0.56, 0.33), straw * 0.38);

    float earth = smoothstep(0.58, 0.84, fbmG(p * 1.6 + 3.1));
    col = mix(col, vec3(0.42, 0.31, 0.18), earth * 0.42);

    vec2 gp = p * 26.0;
    col *= 1.0 + 0.09 * grainG(gp) * grainFade(gp)
               + 0.05 * grainG(p * 7.0);

    vec2 sp = p * 42.0;
    float fade = grainFade(sp);
    if (fade > 0.01) {
        // Clumped rather than evenly peppered: an even scatter of specks reads
        // as freckles, where real flock gathers and thins.
        float clump = 0.30 + 0.70 * smoothstep(0.28, 0.72, fbmG(p * 5.0 + 31.0));
        float a = fade * clump;
        col = mix(col, vec3(0.31, 0.35, 0.19), specks(sp, 0.0, 0.66) * 0.20 * a);
        col = mix(col, vec3(0.52, 0.51, 0.32),
                  specks(sp * 1.37, 7.1, 0.54) * 0.17 * a);
        col = mix(col, vec3(0.44, 0.35, 0.22),
                  specks(sp * 0.83, 23.9, 0.44) * 0.15 * a);
    }
    vec2 tp = p * 15.0;
    float tufts = grainFade(tp);
    if (tufts > 0.01) {
        col = mix(col, vec3(0.33, 0.37, 0.19), specks(tp, 3.4, 0.34) * 0.15 * tufts);
    }
    return col;
}
