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
uniform vec3 moveColor;

// Auto-bound by Panda3D; falls back to 0.0 if unavailable (shader stays static).
uniform float osg_FrameTime;

in vec3 worldPos;
in vec3 worldNormal;
in vec2 texcoord;
in vec4 eyePos;

out vec4 p3d_FragColor;

// ── Shared material vocabulary with the ground card ───────────────────────
#pragma include "mat_noise.glsl"
#pragma include "shadow.glsl"

// The river and marsh were tuned against a 4-octave sum that topped out at
// 0.9375, so they keep that range rather than the normalised one.
float fbm(vec2 p) {
    return fbmG(p) * 0.9375;
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
    float lambert = clamp(dot(N, sunDir), 0.0, 1.0) * 0.7 + 0.3;

    vec2 p = worldPos.xy;
    vec3 col;
    float alpha = 1.0;

    if (terrainType == 0) {
        // Forest floor. The base is the mat's own grass, so the sabot reads as
        // a piece of the table with a wood on it, rather than a green plate
        // set down beside it.
        vec2 mp = p * MAT_WORLD_SCALE;
        col = matGrass(mp);

        // 0 at the rim, 1 well inside. Everything that makes this a wood
        // rather than a lawn is keyed off it, so the edge stays open.
        float inside = smoothstep(edgeLevel, edgeLevel + 0.22, texcoord.x);

        // Shade cast by the canopy, tinted rather than repainted: the grass
        // underneath still shows through. A little of it reaches the rim, or
        // the base stops being visible as a base at all.
        float canopy = fbmG(p * 0.9);
        vec3 shade = mix(vec3(0.44, 0.52, 0.42), vec3(0.66, 0.74, 0.58), canopy);
        col *= mix(vec3(1.0), shade, 0.28 + 0.57 * inside);

        // Bare soil and needle litter where the canopy is thickest and no
        // grass gets the light.
        float soil = smoothstep(0.46, 0.86, fbmG(p * 1.25 + 17.3));
        col = mix(col, vec3(0.23, 0.18, 0.12), soil * 0.60 * inside);

        vec2 np = p * 6.5;
        float nf = grainFade(np);
        if (nf > 0.01) {
            col = mix(col, vec3(0.33, 0.24, 0.14),
                      specks(np, 55.3, 0.42) * 0.34 * nf * inside);
        }

        // Small stones, all the way to the rim: they are part of the ground,
        // not of the wood.
        vec2 sp = p * 3.4;
        float sf = grainFade(sp);
        if (sf > 0.01) {
            col = mix(col, vec3(0.45, 0.43, 0.38),
                      specks(sp, 71.9, 0.30) * 0.26 * sf);
        }

        // Scattered flock over the rim, so the outline breaks up into tufts
        // instead of ending on a clean line.
        vec2 fp = p * 2.4;
        float ff = grainFade(fp);
        if (ff > 0.01) {
            float rim = 1.0 - inside;
            col = mix(col, vec3(0.26, 0.34, 0.18),
                      specks(fp, 101.7, 0.46) * 0.42 * ff * rim);
        }
    } else if (terrainType == 1) {
        // Hill — literally the mat's grass, so the slope reads as the same
        // ground lifted rather than as a different material.
        vec2 mp = p * MAT_WORLD_SCALE;
        col = matGrass(mp);

        // Scattered stone and scree, thickening toward the crown.
        float h = clamp(worldPos.z * 0.18, 0.0, 1.0);
        float crown = smoothstep(0.25, 0.95, h);
        vec2 rp = p * 1.6;
        float rf = grainFade(rp);
        if (rf > 0.01) {
            float rock = specks(rp, 41.0, 0.42) * rf;
            col = mix(col, vec3(0.46, 0.43, 0.36), rock * (0.20 + 0.55 * crown));
        }
        vec2 pp = p * 5.0;
        float pf = grainFade(pp);
        if (pf > 0.01) {
            col = mix(col, vec3(0.40, 0.37, 0.31),
                      specks(pp, 63.7, 0.38) * 0.28 * pf * (0.3 + 0.7 * crown));
        }
        // Worn, drier grass on the exposed top.
        col = mix(col, vec3(0.47, 0.44, 0.26), crown * 0.22);
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

    col = shadeSun(col, sunShadow(eyePos));

    // Movement/shooting range overlay wrapped over the terrain surface. The
    // board card spans world -50..50, so map world XY into the same 0..1 space.
    if (moveActive) {
        vec2 ouv = worldPos.xy * 0.01 + 0.5;
        float d = sdPolygon(ouv, movePoints);
        vec3 o = (d > 0.0) ? col : moveColor;
        o *= 1.0 - exp(-24.0 * abs(d));
        o = mix(o, vec3(1.0), 1.0 - smoothstep(0.0, 0.015, abs(d) + 0.0075));
        // Keep the indicator a little transparent so terrain shows through.
        col = mix(col, o, 0.9);
    }

    p3d_FragColor = vec4(col, alpha);
}
