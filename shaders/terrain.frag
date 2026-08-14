#version 150

// terrainType: 0 = forest, 1 = hill, 2 = river, 3 = marsh
uniform int terrainType;
uniform vec4 baseColor;
uniform vec2 pieceSize;   // world-space width/height of this terrain piece
uniform float edgeLevel;  // hill/forest: discard fragments below this field value

// Movement/shooting range overlay (same SDF the ground card draws).
#define MOVE_MAXPTS 83
uniform vec2 movePoints[MOVE_MAXPTS];
uniform bool moveActive;

// Auto-bound by Panda3D; falls back to 0.0 if unavailable (shader stays static).
uniform float osg_FrameTime;

in vec3 worldPos;
in vec3 worldNormal;
in vec2 texcoord;

out vec4 p3d_FragColor;

// ── Cheap value-noise / fbm for procedural surface detail ──────────────────
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p *= 2.0;
        a *= 0.5;
    }
    return v;
}

// Signed distance to the movement/shooting polygon (matches the ground card).
float sdPolygon(in vec2 pt, in vec2[MOVE_MAXPTS] v) {
    float d = dot(pt - v[0], pt - v[0]);
    float s = 1.0;
    for (int i = 0, j = MOVE_MAXPTS - 1; i < MOVE_MAXPTS; j = i, i++) {
        vec2 e = v[j] - v[i];
        vec2 w = pt - v[i];
        vec2 b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        d = min(d, dot(b, b));
        bvec3 cond = bvec3(pt.y >= v[i].y, pt.y < v[j].y, e.x * w.y > e.y * w.x);
        if (all(cond) || all(not(cond))) s = -s;
    }
    return s * sqrt(d);
}

void main() {
    // Hill/forest carry the footprint field in texcoord.x; cut the rim per
    // pixel so the silhouette is smooth regardless of mesh resolution.
    if ((terrainType == 0 || terrainType == 1) && texcoord.x < edgeLevel) {
        discard;
    }

    // Guard against missing/zero normals in the source mesh.
    vec3 N = worldNormal;
    if (dot(N, N) < 0.0001) {
        N = vec3(0.0, 0.0, 1.0);
    }
    N = normalize(N);

    // Simple directional key light + ambient fill.
    float lambert = clamp(dot(N, normalize(vec3(0.4, 0.3, 0.9))), 0.0, 1.0) * 0.7 + 0.3;

    vec2 p = worldPos.xy;
    vec3 col;
    float alpha = 1.0;

    if (terrainType == 0) {
        // Forest — dappled green canopy that drifts slightly over time.
        float canopy = fbm(p * 0.8);
        vec3 dark = vec3(0.05, 0.22, 0.06);
        vec3 lite = vec3(0.15, 0.45, 0.12);
        col = mix(dark, lite, canopy);
        float dapple = fbm(p * 1.6 + osg_FrameTime * 0.05);
        col += vec3(0.06, 0.10, 0.02) * smoothstep(0.55, 0.9, dapple);
    } else if (terrainType == 1) {
        // Hill — grass on the lower slopes blending to rock near the top.
        float h = clamp(worldPos.z * 0.18, 0.0, 1.0);
        vec3 grass = vec3(0.20, 0.42, 0.14);
        vec3 rock = vec3(0.45, 0.40, 0.32);
        col = mix(grass, rock, smoothstep(0.3, 0.9, h));
        col *= mix(0.9, 1.15, fbm(p * 1.2));
    } else if (terrainType == 2) {
        // River — water body, foamy shoreline, wet/sandy banks, soft edge.
        float along  = texcoord.x;
        float across = texcoord.y;
        float d = abs(across - 0.5) * 2.0;         // 0 centre → 1 outer bank edge
        float waterEdge = 0.667;                   // matches mesh bank_scale (1.5)

        float t = osg_FrameTime;
        float flow = t * 0.15;
        vec2 fp = vec2(along, across);

        // Layered scrolling ripples for surface motion.
        float r1 = fbm(fp * vec2(10.0, 5.0) + vec2(flow * 2.0, 0.0));
        float r2 = fbm(fp * vec2(20.0, 9.0) - vec2(flow * 3.0, flow));
        float ripple = r1 * 0.6 + r2 * 0.4;

        // Water: deeper/darker toward the channel centre.
        float wd = clamp(d / waterEdge, 0.0, 1.0);
        vec3 deep    = vec3(0.03, 0.16, 0.30);
        vec3 shallow = vec3(0.16, 0.44, 0.56);
        vec3 water = mix(shallow, deep, 1.0 - wd);
        water += vec3(0.10) * smoothstep(0.55, 0.95, ripple);
        float glint = pow(smoothstep(0.72, 1.0, ripple), 6.0);
        water += vec3(0.9, 0.95, 1.0) * glint * 0.5;

        // Wet-to-dry, noisy bank beyond the waterline.
        float bt = smoothstep(waterEdge, 1.0, d);
        vec3 bankWet = vec3(0.30, 0.25, 0.15);
        vec3 bankDry = vec3(0.42, 0.38, 0.24);
        vec3 bank = mix(bankWet, bankDry, bt);
        bank *= 0.82 + 0.34 * fbm(fp * 18.0);

        // Shoreline blend + foam right at the waterline.
        float shore = smoothstep(waterEdge - 0.08, waterEdge + 0.05, d);
        col = mix(water, bank, shore);
        float foam = (1.0 - smoothstep(0.0, 0.11, abs(d - waterEdge)))
                   * (0.5 + 0.5 * fbm(fp * 40.0 + flow * 4.0));
        col = mix(col, vec3(0.85, 0.90, 0.92), clamp(foam * 0.8, 0.0, 1.0));

        // Irregular, soft fade into the surrounding grass.
        float edgeN = (fbm(fp * 26.0) - 0.5) * 0.10;
        alpha = 1.0 - smoothstep(0.80, 1.0, d + edgeN);
    } else {
        // Marsh — an irregular, soft-edged bog rather than a rectangle.
        float murk = fbm(p * 1.5 + osg_FrameTime * 0.03);
        vec3 mud = vec3(0.22, 0.20, 0.10);
        vec3 scum = vec3(0.18, 0.30, 0.12);
        col = mix(mud, scum, murk);

        vec2 c = (texcoord - 0.5) * 2.0;
        float rr = length(c);
        float wob = (fbm(texcoord * 5.0) - 0.5) * 0.6;
        alpha = smoothstep(1.05, 0.55, rr + wob);
    }

    // Nudge toward the configured base tint so gameplay colours stay readable
    // (skipped for the river so its banks don't turn blue).
    if (terrainType != 2) {
        col = mix(col, baseColor.rgb, 0.15);
    }
    col *= lambert;

    // Movement/shooting range overlay wrapped over the terrain surface. The
    // board card spans world -50..50, so map world XY into the same 0..1 space.
    if (moveActive) {
        vec2 ouv = worldPos.xy * 0.01 + 0.5;
        float d = sdPolygon(ouv, movePoints);
        vec3 o = (d > 0.0) ? col : vec3(0.65, 0.85, 1.0);
        o *= 1.0 - exp(-24.0 * abs(d));
        o = mix(o, vec3(1.0), 1.0 - smoothstep(0.0, 0.015, abs(d) + 0.0075));
        // Keep the indicator a little transparent so terrain shows through.
        col = mix(col, o, 0.9);
    }

    p3d_FragColor = vec4(col, alpha);
}
