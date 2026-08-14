#version 150

// terrainType: 0 = forest, 1 = hill, 2 = river, 3 = marsh
uniform int terrainType;
uniform vec4 baseColor;
uniform vec2 pieceSize;   // world-space width/height of this terrain piece

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

void main() {
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
        // River — a flowing ribbon; u runs downstream, v runs bank-to-bank.
        float along  = texcoord.x;
        float across = texcoord.y;                 // 0 = left bank, 1 = right
        float d = abs(across - 0.5) * 2.0;         // 0 centre → 1 bank

        float t = osg_FrameTime;
        float flow = t * 0.15;
        vec2 fp = vec2(along, across);

        // Layered scrolling ripples for surface motion.
        float r1 = fbm(fp * vec2(10.0, 5.0) + vec2(flow * 2.0, 0.0));
        float r2 = fbm(fp * vec2(20.0, 9.0) - vec2(flow * 3.0, flow));
        float ripple = r1 * 0.6 + r2 * 0.4;

        // Darker/deeper toward the middle of the channel.
        float depth = 1.0 - d;
        vec3 deep    = vec3(0.03, 0.16, 0.30);
        vec3 shallow = vec3(0.16, 0.44, 0.56);
        vec3 water = mix(shallow, deep, depth);
        water += vec3(0.10) * smoothstep(0.55, 0.95, ripple);          // sparkle
        float glint = pow(smoothstep(0.72, 1.0, ripple), 6.0);
        water += vec3(0.9, 0.95, 1.0) * glint * 0.5;                   // sun glint

        // Foam churning against the banks.
        float foam = smoothstep(0.6, 0.95, d)
                   * (0.5 + 0.5 * fbm(fp * 40.0 + flow * 4.0));
        water = mix(water, vec3(0.85, 0.90, 0.92), clamp(foam, 0.0, 1.0));

        col = water;
        // Soft edge so the ribbon melts into the ground, not a hard border.
        alpha = 1.0 - smoothstep(0.85, 1.0, d);
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

    // Nudge toward the configured base tint so gameplay colours stay readable.
    col = mix(col, baseColor.rgb, 0.15);
    col *= lambert;

    p3d_FragColor = vec4(col, alpha);
}
